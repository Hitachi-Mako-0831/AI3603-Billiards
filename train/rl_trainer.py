"""
rl_trainer.py - 强化学习训练框架

使用自我对弈 + 价值网络来优化 HybridAgent 的决策

主要组件：
- ValueNetwork: 局面评估神经网络
- StateEncoder: 台球局面特征编码器
- ReplayBuffer: 经验回放缓冲区
- RLTrainer: TD学习训练器
- SelfPlayManager: 自我对弈管理器
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import copy
from collections import deque
from pathlib import Path
import json
from datetime import datetime


class StateEncoder:
    """台球局面特征编码器

    将台球局面编码为固定维度的特征向量
    """

    def __init__(self, max_balls=7):
        """
        参数:
            max_balls: 每方最大球数（实心/条纹各7个）
        """
        self.max_balls = max_balls
        # 特征维度:
        # - 白球位置(2) + 8号球位置(2)
        # - 目标球位置(7*2=14) + 对手球位置(7*2=14)
        # - 各球到最近袋口距离(1+7+7=15)
        # - 局面统计特征(10)
        self.feature_dim = 2 + 2 + 14 + 14 + 15 + 10  # = 57

        # 球桌尺寸（用于归一化）
        self.table_length = 2.24
        self.table_width = 1.12

    def encode(self, balls, my_targets, table, pocket_positions=None):
        """编码局面状态

        参数:
            balls: {ball_id: Ball} 球状态字典
            my_targets: 目标球ID列表
            table: 球桌对象
            pocket_positions: 袋口位置列表

        返回:
            np.array: 特征向量
        """
        features = []

        # 获取袋口位置
        if pocket_positions is None:
            if hasattr(table, 'pockets') and table.pockets:
                pocket_positions = [
                    (float(p.center[0]), float(p.center[1]))
                    for p in table.pockets.values()
                ]
            else:
                L = getattr(table, 'l', 2.24)
                W = getattr(table, 'w', 1.12)
                pocket_positions = [
                    (0.0, 0.0), (L/2, 0.0), (L, 0.0),
                    (0.0, W), (L/2, W), (L, W),
                ]

        # 1. 白球位置 (2维)
        cue_pos = self._get_ball_pos(balls, 'cue')
        features.extend(self._normalize_pos(cue_pos))

        # 2. 8号球位置 (2维)
        eight_pos = self._get_ball_pos(balls, '8')
        features.extend(self._normalize_pos(eight_pos))

        # 3. 目标球位置 (14维)
        target_positions = []
        for bid in my_targets:
            if bid != '8':
                pos = self._get_ball_pos(balls, bid)
                target_positions.append(pos)

        # 填充到max_balls个
        while len(target_positions) < self.max_balls:
            target_positions.append((-1, -1))  # 表示不存在

        for pos in target_positions[:self.max_balls]:
            features.extend(self._normalize_pos(pos))

        # 4. 对手球位置 (14维)
        opponent_positions = []
        for bid, ball in balls.items():
            if bid not in my_targets and bid not in ['cue', '8']:
                if ball.state.s != 4:  # 未进袋
                    pos = (ball.state.rvw[0][0], ball.state.rvw[0][1])
                    opponent_positions.append(pos)

        while len(opponent_positions) < self.max_balls:
            opponent_positions.append((-1, -1))

        for pos in opponent_positions[:self.max_balls]:
            features.extend(self._normalize_pos(pos))

        # 5. 各球到最近袋口的距离 (15维)
        # 白球
        features.append(self._min_dist_to_pocket(cue_pos, pocket_positions))
        # 目标球
        for pos in target_positions[:self.max_balls]:
            features.append(self._min_dist_to_pocket(pos, pocket_positions))
        # 对手球
        for pos in opponent_positions[:self.max_balls]:
            features.append(self._min_dist_to_pocket(pos, pocket_positions))

        # 6. 局面统计特征 (10维)
        stats = self._compute_stats(balls, my_targets, cue_pos, pocket_positions)
        features.extend(stats)

        return np.array(features, dtype=np.float32)

    def _get_ball_pos(self, balls, ball_id):
        """获取球的位置"""
        if ball_id in balls and balls[ball_id].state.s != 4:
            return (balls[ball_id].state.rvw[0][0], balls[ball_id].state.rvw[0][1])
        return (-1, -1)

    def _normalize_pos(self, pos):
        """归一化位置到 [-1, 1]"""
        if pos[0] < 0:
            return [-1.0, -1.0]
        x = (pos[0] / self.table_length) * 2 - 1
        y = (pos[1] / self.table_width) * 2 - 1
        return [np.clip(x, -1, 1), np.clip(y, -1, 1)]

    def _min_dist_to_pocket(self, pos, pockets):
        """计算到最近袋口的距离（归一化）"""
        if pos[0] < 0:
            return 1.0  # 不存在的球返回最大距离

        min_dist = float('inf')
        for pocket in pockets:
            dist = np.sqrt((pos[0] - pocket[0])**2 + (pos[1] - pocket[1])**2)
            min_dist = min(min_dist, dist)

        # 归一化（最大对角线距离约2.5m）
        return min(min_dist / 2.5, 1.0)

    def _compute_stats(self, balls, my_targets, cue_pos, pockets):
        """计算局面统计特征"""
        stats = []

        # 1. 剩余目标球数量（归一化）
        remaining_own = sum(1 for bid in my_targets
                          if bid in balls and balls[bid].state.s != 4 and bid != '8')
        stats.append(remaining_own / 7.0)

        # 2. 剩余对手球数量
        remaining_opp = sum(1 for bid, b in balls.items()
                          if bid not in my_targets and bid not in ['cue', '8']
                          and b.state.s != 4)
        stats.append(remaining_opp / 7.0)

        # 3. 是否在打8号球
        is_targeting_eight = 1.0 if (remaining_own == 0 or my_targets == ['8']) else 0.0
        stats.append(is_targeting_eight)

        # 4. 白球到最近目标球的距离
        min_dist_to_target = 1.0
        for bid in my_targets:
            if bid in balls and balls[bid].state.s != 4:
                pos = (balls[bid].state.rvw[0][0], balls[bid].state.rvw[0][1])
                if cue_pos[0] >= 0:
                    dist = np.sqrt((cue_pos[0] - pos[0])**2 + (cue_pos[1] - pos[1])**2)
                    min_dist_to_target = min(min_dist_to_target, dist / 2.5)
        stats.append(min_dist_to_target)

        # 5. 平均目标球到袋口距离
        avg_dist = 0.0
        count = 0
        for bid in my_targets:
            if bid in balls and balls[bid].state.s != 4:
                pos = (balls[bid].state.rvw[0][0], balls[bid].state.rvw[0][1])
                avg_dist += self._min_dist_to_pocket(pos, pockets)
                count += 1
        stats.append(avg_dist / max(count, 1))

        # 6. 白球位置质量（距离边库）
        if cue_pos[0] >= 0:
            dist_to_cushion = min(
                cue_pos[0], self.table_length - cue_pos[0],
                cue_pos[1], self.table_width - cue_pos[1]
            )
            stats.append(min(dist_to_cushion / 0.3, 1.0))
        else:
            stats.append(0.0)

        # 7-10. 填充
        stats.extend([0.0, 0.0, 0.0, 0.0])

        return stats[:10]


class ValueNetwork(nn.Module):
    """局面价值评估网络

    输入: 局面特征向量
    输出: 局面价值 (预期胜率，范围 [0, 1])
    """

    def __init__(self, input_dim=57, hidden_dims=[256, 128, 64]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # 输出范围 [0, 1]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ShotEvaluator(nn.Module):
    """击球评估网络

    输入: 局面特征 + 击球参数
    输出: 击球后的预期价值
    """

    def __init__(self, state_dim=57, action_dim=5, hidden_dims=[256, 128, 64]):
        super().__init__()

        input_dim = state_dim + action_dim
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, state, action):
        """
        参数:
            state: (batch, state_dim) 局面特征
            action: (batch, action_dim) 击球参数 [V0, phi, theta, a, b]
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, info=None):
        """存储一条经验"""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'info': info or {}
        })

    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states = torch.FloatTensor(np.array([e['state'] for e in batch]))
        actions = torch.FloatTensor(np.array([e['action'] for e in batch]))
        rewards = torch.FloatTensor([e['reward'] for e in batch])
        next_states = torch.FloatTensor(np.array([e['next_state'] for e in batch]))
        dones = torch.FloatTensor([e['done'] for e in batch])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        """保存缓冲区到文件"""
        data = list(self.buffer)
        np.save(path, data, allow_pickle=True)

    def load(self, path):
        """从文件加载缓冲区"""
        data = np.load(path, allow_pickle=True)
        self.buffer = deque(data, maxlen=self.buffer.maxlen)


class RLTrainer:
    """强化学习训练器

    使用TD学习训练价值网络
    """

    def __init__(
        self,
        value_net: ValueNetwork,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.value_net = value_net.to(device)
        self.target_net = copy.deepcopy(value_net).to(device)
        self.device = device
        self.gamma = gamma

        self.optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.95)

        self.train_step = 0
        self.losses = []

    def train_step_td(self, replay_buffer: ReplayBuffer, batch_size: int = 64):
        """执行一步TD学习

        使用 TD(0) 更新: V(s) <- r + gamma * V(s')
        """
        if len(replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = states.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 当前状态价值
        current_values = self.value_net(states).squeeze()

        # 目标价值: r + gamma * V(s') * (1 - done)
        with torch.no_grad():
            next_values = self.target_net(next_states).squeeze()
            target_values = rewards + self.gamma * next_values * (1 - dones)

        # MSE损失
        loss = F.mse_loss(current_values, target_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.train_step += 1
        self.losses.append(loss.item())

        # 软更新目标网络
        if self.train_step % 100 == 0:
            self._soft_update_target(tau=0.01)

        return loss.item()

    def _soft_update_target(self, tau=0.01):
        """软更新目标网络"""
        for target_param, param in zip(self.target_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, path):
        """保存模型"""
        torch.save({
            'value_net': self.value_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_step': self.train_step,
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.value_net.load_state_dict(checkpoint['value_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.train_step = checkpoint['train_step']


class SelfPlayManager:
    """自我对弈管理器

    管理 HybridAgent 之间的对弈，收集训练数据
    """

    def __init__(
        self,
        env_class,
        agent_class,
        state_encoder: StateEncoder,
        value_net: ValueNetwork = None
    ):
        self.env_class = env_class
        self.agent_class = agent_class
        self.state_encoder = state_encoder
        self.value_net = value_net

        self.episode_rewards = []
        self.win_rates = []

    def collect_episode(self, replay_buffer: ReplayBuffer, verbose=False):
        """收集一局对弈数据

        返回:
            winner: 'A' 或 'B'
            episode_length: 回合数
        """
        env = self.env_class(verbose=verbose, record_shots=False)

        # 创建两个agent
        agent_a = self.agent_class()
        agent_b = self.agent_class()

        # 随机选择球型
        target_ball = random.choice(['solid', 'stripe'])
        env.reset(target_ball=target_ball)

        episode_data = []
        shot_count = 0
        max_shots = 60

        while shot_count < max_shots:
            done, info = env.get_done()
            if done:
                break

            player = env.get_curr_player()
            balls, my_targets, table = env.get_observation(player)

            # 编码当前状态
            state = self.state_encoder.encode(balls, my_targets, table)

            # 选择动作
            if player == 'A':
                action_dict = agent_a.decision(balls, my_targets, table)
            else:
                action_dict = agent_b.decision(balls, my_targets, table)

            # 转换动作为向量
            action = self._action_to_vector(action_dict)

            # 执行动作
            step_info = env.take_shot(action_dict)

            # 获取下一状态
            next_balls, next_targets, next_table = env.get_observation(player)
            next_state = self.state_encoder.encode(next_balls, next_targets, next_table)

            # 计算即时奖励
            reward = self._compute_reward(step_info, player)

            # 存储数据
            episode_data.append({
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'player': player
            })

            shot_count += 1

        # 获取最终结果
        done, info = env.get_done()
        winner = info.get('winner', None) if done else None

        # 计算最终奖励并添加到缓冲区
        for i, data in enumerate(episode_data):
            # 终局奖励
            final_reward = 0
            is_done = (i == len(episode_data) - 1)

            if is_done and winner:
                if (winner == 'A' and data['player'] == 'A') or \
                   (winner == 'B' and data['player'] == 'B'):
                    final_reward = 1.0  # 胜利
                else:
                    final_reward = -1.0  # 失败

            total_reward = data['reward'] + final_reward

            replay_buffer.push(
                state=data['state'],
                action=data['action'],
                reward=total_reward,
                next_state=data['next_state'],
                done=is_done
            )

        return winner, shot_count

    def _action_to_vector(self, action_dict):
        """将动作字典转换为向量"""
        return np.array([
            action_dict.get('V0', 3.0) / 8.0,  # 归一化到 [0, 1]
            action_dict.get('phi', 180.0) / 360.0,
            action_dict.get('theta', 0.0) / 90.0,
            (action_dict.get('a', 0.0) + 0.5),  # [-0.5, 0.5] -> [0, 1]
            (action_dict.get('b', 0.0) + 0.5),
        ], dtype=np.float32)

    def _compute_reward(self, step_info, player):
        """计算即时奖励"""
        reward = 0

        # 进球奖励
        pocketed = step_info.get('POCKETED', [])
        for ball_id in pocketed:
            if ball_id == 'cue':
                reward -= 0.5  # 白球进袋
            elif ball_id == '8':
                # 8号球处理在终局奖励中
                pass
            else:
                # 简化处理：任何进球给小奖励
                reward += 0.2

        # 犯规惩罚
        if step_info.get('FOUL_FIRST_HIT'):
            reward -= 0.3
        if step_info.get('NO_POCKET_NO_RAIL'):
            reward -= 0.2
        if step_info.get('NO_HIT'):
            reward -= 0.3

        return reward

    def run_training(
        self,
        trainer: RLTrainer,
        replay_buffer: ReplayBuffer,
        num_episodes: int = 1000,
        train_freq: int = 4,
        batch_size: int = 64,
        save_freq: int = 100,
        save_dir: str = 'checkpoints'
    ):
        """运行训练循环

        参数:
            trainer: RLTrainer 实例
            replay_buffer: ReplayBuffer 实例
            num_episodes: 总对弈局数
            train_freq: 每多少局训练一次
            batch_size: 训练批大小
            save_freq: 保存频率
            save_dir: 保存目录
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)

        wins = {'A': 0, 'B': 0, None: 0}

        print(f"开始训练，共 {num_episodes} 局...")
        print(f"设备: {trainer.device}")

        for episode in range(num_episodes):
            # 收集一局数据
            winner, length = self.collect_episode(replay_buffer, verbose=False)
            wins[winner] += 1

            # 定期训练
            if episode > 0 and episode % train_freq == 0:
                for _ in range(train_freq):
                    loss = trainer.train_step_td(replay_buffer, batch_size)

            # 打印进度
            if (episode + 1) % 10 == 0:
                recent_losses = trainer.losses[-100:] if trainer.losses else [0]
                avg_loss = sum(recent_losses) / len(recent_losses)
                print(f"Episode {episode+1}/{num_episodes} | "
                      f"Buffer: {len(replay_buffer)} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Wins A/B: {wins['A']}/{wins['B']}")

            # 保存检查点
            if (episode + 1) % save_freq == 0:
                trainer.save(save_path / f'value_net_ep{episode+1}.pt')
                replay_buffer.save(save_path / f'replay_buffer_ep{episode+1}.npy')
                print(f"检查点已保存: episode {episode+1}")

        # 保存最终模型
        trainer.save(save_path / 'value_net_final.pt')
        print("训练完成！")

        return wins


def create_rl_agent_wrapper(value_net_path: str):
    """创建带有训练好的价值网络的 HybridAgent

    参数:
        value_net_path: 价值网络模型路径

    返回:
        配置好的 HybridAgent
    """
    from agent import HybridAgent

    agent = HybridAgent(use_value_network=True, value_net_path=value_net_path)
    return agent


# ==================== 训练脚本入口 ====================

def main():
    """主训练函数"""
    import argparse

    parser = argparse.ArgumentParser(description="Train HybridAgent with RL")
    parser.add_argument("--episodes", type=int, default=500, help="训练局数")
    parser.add_argument("--batch-size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--save-dir", type=str, default="checkpoints", help="保存目录")
    parser.add_argument("--resume", type=str, default=None, help="从检查点恢复")
    args = parser.parse_args()

    # 导入环境和agent
    from poolenv import PoolEnv
    from agent import HybridAgent

    # 创建组件
    state_encoder = StateEncoder()
    value_net = ValueNetwork(input_dim=state_encoder.feature_dim)
    replay_buffer = ReplayBuffer(capacity=50000)

    trainer = RLTrainer(
        value_net=value_net,
        learning_rate=args.lr,
        gamma=0.99
    )

    # 恢复检查点
    if args.resume:
        trainer.load(args.resume)
        print(f"从 {args.resume} 恢复训练")

    self_play = SelfPlayManager(
        env_class=PoolEnv,
        agent_class=HybridAgent,
        state_encoder=state_encoder,
        value_net=value_net
    )

    # 开始训练
    wins = self_play.run_training(
        trainer=trainer,
        replay_buffer=replay_buffer,
        num_episodes=args.episodes,
        train_freq=4,
        batch_size=args.batch_size,
        save_dir=args.save_dir
    )

    print(f"\n训练统计:")
    print(f"  Player A 胜场: {wins['A']}")
    print(f"  Player B 胜场: {wins['B']}")
    print(f"  平局/超时: {wins[None]}")


if __name__ == "__main__":
    main()
