"""
PoolEnv Gym Wrapper
将原始 PoolEnv 封装为标准 Gym 接口，并将字典观测转换为向量
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import copy
from poolenv import PoolEnv

class BilliardGymEnv(gym.Env):
    """
    Gymnasium Wrapper for PoolEnv
    
    Observation Space (Box):
        [
         cue_x, cue_y, cue_vx, cue_vy,     (白球)
         ball_1_x, ball_1_y, ball_1_vx, ... (1-15号球)
         target_1_flag, target_2_flag, ...  (1-15号球是否为我的目标, 0/1)
        ]
        
    Action Space (Box):
        [V0, phi, theta, a, b]
        归一化到 [-1, 1] 以便于 RL 训练
    """
    
    def __init__(self, env_config=None, opponent=None):
        self.env = PoolEnv()
        self.config = env_config or {}
        self.opponent = opponent # Opponent Agent instance
        
        # 定义动作空间 (Normalized)
        # 实际范围:
        # V0: [0.5, 8.0]
        # phi: [0, 360]
        # theta: [0, 90]
        # a: [-0.5, 0.5]
        # b: [-0.5, 0.5]
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
            shape=(5,),
            dtype=np.float32
        )
        
        # 定义观测空间
        # 16个球 * 4 (x, y, vx, vy) + 15个目标标记 = 64 + 15 = 79 维
        # 这里简化处理，只取位置和速度，忽略 spin 和 z轴
        self.obs_dim = 16 * 4 + 15
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32
        )
        
        self.my_player_id = 'A' # 默认视角
        self.opponent_player_id = 'B'
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Determine roles and targets based on options
        # options: {'agent_id': 'A' or 'B', 'target_ball': 'solid' or 'stripe' (for Player A)}
        # If agent_id is 'A', we are Player A.
        # If agent_id is 'B', we are Player B.
        
        self.my_player_id = options.get('agent_id', 'A') if options else 'A'
        self.opponent_player_id = 'B' if self.my_player_id == 'A' else 'A'
        
        # target_ball is always set for Player A in env.reset
        target_ball_for_player_A = options.get('target_ball', 'solid') if options else 'solid'
        
        self.env.reset(target_ball=target_ball_for_player_A)
        
        # If we are Player B, the opponent (Player A) moves first.
        if self.my_player_id == 'B':
            self._play_opponent_turn()
            
        obs = self._get_obs()
        info = {}
        return obs, info
    
    def step(self, action):
        # 1. 还原动作 ([-1, 1] -> 实际物理量)
        real_action = self._denormalize_action(action)
        
        # 2. 检查游戏是否结束 (Pre-check)
        done, result = self.env.get_done()
        if done:
            return self._get_obs(), 0.0, True, False, result

        # 3. 执行动作 (My Turn)
        # take_shot 会返回一个 info dict
        shot_info = self.env.take_shot(real_action)
        
        # 4. 计算 Reward (优化版 - 适配新规则)
        reward = 0.0
        
        # 从 shot_info 获取详细事件
        my_pocketed = len(shot_info.get('ME_INTO_POCKET', []))
        enemy_pocketed = len(shot_info.get('ENEMY_INTO_POCKET', []))
        cue_pocketed = shot_info.get('WHITE_BALL_INTO_POCKET', False)
        foul_first_hit = shot_info.get('FOUL_FIRST_HIT', False)
        no_pocket_no_rail = shot_info.get('NO_POCKET_NO_RAIL', False)
        
        # --- A. 进球奖励 (Potting Rewards) ---
        if my_pocketed > 0:
            reward += 10.0 * my_pocketed # 鼓励进球 (原 20)
        
        # --- B. 基础惩罚 (Basic Penalties) ---
        if enemy_pocketed > 0:
            reward -= 5.0 * enemy_pocketed # 误进对手球 (原 10)
        
        # --- C. 犯规惩罚 (Foul Penalties - 关键优化) ---
        if cue_pocketed:
            reward -= 5.0 # 白球进袋 (原 20，因为已经丢了球权，不用罚太重)
        
        if foul_first_hit:
            # 首球犯规
            reward -= 2.0 # (原 15)
            
        if no_pocket_no_rail:
            # 无进球且无碰库
            reward -= 2.0 # (原 15)
            
        # --- D. 过程奖励 (Shaping Rewards) ---
        # 1. 鼓励击打到球（减少空杆）
        if not shot_info.get('NO_HIT', False):
            reward += 0.5  # 只要碰到了球，就给一点奖励 (原 1)
            
        # 2. 鼓励合法击球（没有犯规）
        is_foul = cue_pocketed or foul_first_hit or no_pocket_no_rail or shot_info.get('NO_HIT', False)
        if not is_foul:
            reward += 1.0 # 合法击球奖励 (原 2)
            
            # 如果虽然没进球，但是也没有给对手造成自由球机会（即合法），这本身就是一种成功
            # 可以进一步细化：比如是否让母球停到了安全位置？暂时无法判断
        
        # --- E. 胜负奖励 (Game Outcome) ---
        # 5. 检查游戏是否结束 (Post-check after my shot)
        done, result = self.env.get_done()
        if done:
            if result['winner'] == self.my_player_id:
                reward += 50.0 # 赢球大奖 (原 100)
            elif result['winner'] == 'SAME':
                reward += 0.0
            else: 
                reward -= 20.0 # 输球惩罚 (原 50)
        
        # 6. 如果游戏未结束，且轮到对手，执行对手回合
        if not done and self.env.get_curr_player() == self.opponent_player_id:
             opponent_stats = self._play_opponent_turn()
             
             # 计算对手进球带来的负反馈（如果对手进球，我们应该感到一点压力）
             if opponent_stats['my_pocketed'] > 0:
                 # 对手进球越多，我们扣分越多，鼓励 Agent 尽量不要给对手留机会（防守策略）
                 # 但也不能扣太多，因为对手进球主要取决于对手实力
                 reward -= 0.5 * opponent_stats['my_pocketed']
                 
             # 如果对手犯规了（比如白球进袋），这其实是我们的机会，可以给一点微小的奖励？
             # 不，对手犯规我们已经获得了球权，这已经是奖励了，不要重复奖励。

             # 如果对手回合结束后游戏结束
             done, result = self.env.get_done()
             if done:
                 if result['winner'] == self.my_player_id:
                     # 对手蠢死了，把黑8打进去了，或者白球进袋输了
                     # 这种“躺赢”不应该给太多奖励，否则 Agent 会学会“等着对手自杀”
                     reward += 5.0 # 躺赢给个辛苦费 (原 10)
                 elif result['winner'] == 'SAME':
                     reward += 0.0
                 else:
                     # 对手正常打赢了
                     reward -= 10.0 # 输给对手，惩罚适中 (原 20)

        obs = self._get_obs()
        terminated = done
        truncated = False # 可以根据 hit_count 判断是否超时
        
        return obs, reward, terminated, truncated, shot_info

    def _play_opponent_turn(self):
        """Play opponent's turn until it is my turn again or game over"""
        stats = {'my_pocketed': 0, 'enemy_pocketed': 0, 'cue_pocketed': False}
        
        if self.opponent is None:
            # If no opponent, we can't simulate their turn. 
            # In single player / training mode without opponent, we might just return.
            # But the env logic switches turn.
            # If we don't act, the state remains 'opponent turn'.
            # For now, let's warn and return if no opponent
            return stats

        while True:
            done, _ = self.env.get_done()
            if done:
                break
                
            if self.env.get_curr_player() == self.my_player_id:
                break
                
            # Opponent's turn
            obs = self.env.get_observation(self.opponent_player_id)
            try:
                # BasicAgent expects (balls, my_targets, table)
                # env.get_observation returns exactly that tuple
                action = self.opponent.decision(*obs)
                shot_info = self.env.take_shot(action)
                
                # Accumulate stats for opponent turn (Note: 'ME' here refers to Opponent perspective)
                # But shot_info keys are relative to the shooter?
                # Assuming shot_info keys: 'ME_INTO_POCKET', 'ENEMY_INTO_POCKET' are relative to current shooter.
                stats['my_pocketed'] += len(shot_info.get('ME_INTO_POCKET', []))
                stats['enemy_pocketed'] += len(shot_info.get('ENEMY_INTO_POCKET', []))
                if shot_info.get('WHITE_BALL_INTO_POCKET', False):
                    stats['cue_pocketed'] = True
                    
            except Exception as e:
                print(f"[Wrapper] Opponent decision failed: {e}")
                break
                
        return stats


    def _get_obs(self):
        """将字典观测扁平化"""
        balls, my_targets, table = self.env.get_observation(self.my_player_id)
        
        # 1. 处理球 (0-15)
        # 顺序: cue, 1, 2, ... 15
        ball_ids = ['cue'] + [str(i) for i in range(1, 16)]
        
        obs_vec = []
        
        for bid in ball_ids:
            if bid in balls:
                ball = balls[bid]
                # x, y, vx, vy
                obs_vec.extend([
                    ball.state.rvw[0][0], # x
                    ball.state.rvw[0][1], # y
                    ball.state.rvw[1][0], # vx
                    ball.state.rvw[1][1]  # vy
                ])
            else:
                # 球进袋了或者不存在，填0
                obs_vec.extend([0, 0, 0, 0])
                
        # 2. 处理目标 (One-hot 类似)
        # 1-15号球，如果是我的目标，为1，否则为0
        target_vec = []
        for i in range(1, 16):
            bid = str(i)
            if bid in my_targets:
                target_vec.append(1.0)
            else:
                target_vec.append(0.0)
                
        obs_vec.extend(target_vec)
        
        return np.array(obs_vec, dtype=np.float32)

    def _denormalize_action(self, action):
        """[-1, 1] -> Real Value"""
        # V0: [0.5, 8.0]
        v0 = 0.5 + (action[0] + 1) * 0.5 * (8.0 - 0.5)
        
        # phi: [0, 360]
        phi = 0 + (action[1] + 1) * 0.5 * 360
        
        # theta: [0, 90]
        theta = 0 + (action[2] + 1) * 0.5 * 90
        
        # a: [-0.5, 0.5]
        a = -0.5 + (action[3] + 1) * 0.5 * 1.0
        
        # b: [-0.5, 0.5]
        b = -0.5 + (action[4] + 1) * 0.5 * 1.0
        
        return {'V0': float(v0), 'phi': float(phi), 'theta': float(theta), 'a': float(a), 'b': float(b)}

