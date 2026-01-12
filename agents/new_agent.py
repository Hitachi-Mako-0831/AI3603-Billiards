import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
import os
import copy
from .agent import Agent

# --- Helper Functions & Classes (Moved from algorithms/sac/) ---

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256):
        super(QNetwork, self).__init__()
        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256, action_space=None):
        super(GaussianPolicy, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # Action rescaling
        # We assume action space is always [-1, 1] for our normalized environment
        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class ReplayBuffer:
    def __init__(self, capacity, seed, state_dim, action_dim):
        self.capacity = capacity
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        self.buffer = []
        self.position = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # HER Config
        self.her_ratio = 0.8 # 80% 的概率把进错球的数据转化为成功数据
        
        # Obs 结构硬编码 (需与 wrapper.py 保持一致)
        # 16球 * 4 (x,y,vx,vy) = 64
        # 15球 target flags = 15 (index 64-78)
        self.target_flags_start_idx = 64
        self.ball_ids_map = [str(i) for i in range(1, 16)] # '1' to '15'
    
    def push(self, state, action, reward, next_state, done, info=None):
        """
        存储 transition。支持 HER 扩展。
        """
        # 1. Store original transition
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
        # 2. HER Logic (Hindsight Experience Replay)
        # 只有在提供了 info 且启用了 HER 时才尝试
        if info and self.her_ratio > 0:
            self._try_her_store(state, action, next_state, info)

    def _try_her_store(self, state, action, next_state, info):
        """
        尝试生成 HER 数据
        策略：如果我不小心打进了非目标球（对手球），假装那个球就是我的目标。
        """
        enemy_pocketed = info.get('ENEMY_INTO_POCKET', [])
        
        # 如果没有误进球，目前暂时不做 HER (未来可以做定点 HER)
        if not enemy_pocketed:
            return
            
        # 随机决定是否保存 HER 数据
        if random.random() > self.her_ratio:
            return
            
        # 针对每一个误进的球，生成一条 HER 数据
        for ball_id in enemy_pocketed:
            if ball_id not in self.ball_ids_map:
                continue # 忽略黑8或异常ID
                
            ball_idx = self.ball_ids_map.index(ball_id)
            flag_idx = self.target_flags_start_idx + ball_idx
            
            # 检查原始 obs 里这个球是不是已经在 targets 里了 (理论上 info['ENEMY_INTO_POCKET'] 保证了它不是)
            # 但双重检查一下
            if state[flag_idx] > 0.5:
                continue
                
            # --- 构造 HER Transition ---
            
            # 1. 修改 State: 把该球标记为目标
            her_state = state.copy()
            her_state[flag_idx] = 1.0
            
            # 2. 修改 Next State: 也要标记为目标
            her_next_state = next_state.copy()
            her_next_state[flag_idx] = 1.0
            
            # 3. 修改 Reward: 既然进了“目标球”，给奖励
            # 参考 wrapper.py: reward += 10.0 * my_pocketed (原20)
            # 同时要移除原本的惩罚 (原本可能是 -5.0 * enemy_pocketed)
            # 这里简化处理：直接给一个进球的正奖励，加上基础的合法击球奖励
            # 假设: HER 场景下，这被视为一次完美的单球进攻
            her_reward = 10.0 + 1.0 
            
            # 4. 修改 Done: 这里比较棘手。
            # 如果这球进了，游戏结束了吗？
            # 很难判断。通常 HER 不改变 Done，除非是 Goal-Based Env。
            # 这里我们保持 Done 为 False (假设打进一个球通常不会立即结束，除非是黑8)
            her_done = False 
            
            # Store HER transition
            if len(self.buffer) < self.capacity:
                self.buffer.append(None)
            self.buffer[self.position] = (her_state, action, her_reward, her_next_state, her_done)
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        import pickle
        print(f"[Buffer] Saving replay buffer to {path} (Size: {len(self.buffer)})...")
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'buffer': self.buffer,
                    'position': self.position
                }, f)
            print("[Buffer] Saved successfully.")
        except Exception as e:
            print(f"[Buffer] Save failed: {e}")

    def load(self, path):
        import pickle
        import os
        if not os.path.exists(path):
            print(f"[Buffer] No buffer file found at {path}")
            return
            
        print(f"[Buffer] Loading replay buffer from {path}...")
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.buffer = data['buffer']
                self.position = data['position']
            print(f"[Buffer] Loaded {len(self.buffer)} transitions.")
        except Exception as e:
            print(f"[Buffer] Load failed: {e}")


class NewAgent(Agent):
    """
    SAC Agent implementation (Single File)
    继承自 Agent 以兼容 evaluate.py
    """
    def __init__(self, cfg=None):
        super().__init__()
        
        # Default Config if not provided (e.g. for evaluate.py)
        if cfg is None:
            # Default hyperparameters
            cfg = {
                'gamma': 0.99,
                'tau': 0.005,
                'alpha': 0.2,
                'batch_size': 512,
                'hidden_size': 256,
                'lr': 0.0003,
                'start_steps': 1000, # Reduce for eval safety/quick start
                'updates_per_step': 1,
                'target_update_interval': 1,
                'replay_size': 1000000,
                'seed': 42,
                'reward_scale': 0.1,
                'automatic_entropy_tuning': True
            }
        
        self.cfg = cfg
        self.gamma = cfg.get('gamma', 0.99)
        self.tau = cfg.get('tau', 0.005)
        self.alpha = cfg.get('alpha', 0.2)
        self.automatic_entropy_tuning = cfg.get('automatic_entropy_tuning', True)
        self.batch_size = cfg.get('batch_size', 512)
        self.hidden_size = cfg.get('hidden_size', 512)
        self.lr = cfg.get('lr', 0.0003)
        self.start_steps = cfg.get('start_steps', 20000)
        self.updates_per_step = cfg.get('updates_per_step', 1)
        self.target_update_interval = cfg.get('target_update_interval', 1)
        self.reward_scale = cfg.get('reward_scale', 0.1)
        self.replay_size = cfg.get('replay_size', 1000000)
        self.seed = cfg.get('seed', 42)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dimensions (Fixed for Billiards)
        self.state_dim = 79
        self.action_dim = 5
        
        # 1. Initialize Critic
        self.critic = QNetwork(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)
        
        self.critic_target = QNetwork(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 2. Initialize Actor
        self.policy = GaussianPolicy(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.lr)

        # 3. Automatic Entropy Tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor((self.action_dim,)).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha_optim = None
        
        # 4. Replay Buffer
        self.memory = ReplayBuffer(self.replay_size, self.seed, self.state_dim, self.action_dim)
        
        self.total_numsteps = 0
        print(f"[NewAgent/SAC] Initialized on {self.device}. HER Enabled.")
        
        # Try to load default model for evaluation
        self._try_load_default()

    def _try_load_default(self):
        # Auto-load logic: check for 'checkpoints/sac_final.pth'
        default_path = "checkpoints/sac_final.pth"
        if os.path.exists(default_path):
            print(f"[NewAgent] Found default model at {default_path}, loading...")
            self.load(default_path)

    def decision(self, balls, my_targets, table):
        """
        Evaluate.py 调用的接口
        参数:
            balls: dict of pooltool.objects.Ball
            my_targets: list of str
            table: pooltool.objects.Table
        返回:
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        # 1. Convert to vector
        state_vec = self._obs_to_vector(balls, my_targets)
        
        # 2. Select action (evaluate=True)
        action_norm = self.select_action(state_vec, evaluate=True)
        
        # 3. Denormalize
        return self._denormalize_action(action_norm)

    def select_action(self, state, evaluate=False):
        self.total_numsteps += 1
        
        # Warmup: Random actions
        if self.total_numsteps < self.start_steps and not evaluate:
            action = np.random.uniform(-1, 1, self.action_dim)
            return action

        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            _, _, action = self.policy.sample(state)
        else:
            action, _, _ = self.policy.sample(state)
            
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, batch_size):
        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        mask_batch = 1 - mask_batch # 1 if NOT done

        # Scale rewards
        reward_batch = reward_batch * self.reward_scale

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
            
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)

        # Soft Update
        if self.total_numsteps % self.target_update_interval == 0:
            self.soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def store_transition(self, obs, action, reward, next_obs, done, info=None):
        self.memory.push(obs, action, reward, next_obs, done, info)

    def train(self, replay_buffer=None, batch_size=None):
        if self.total_numsteps < self.start_steps:
            return {}

        if len(self.memory) < self.batch_size:
            return {}
            
        bs = batch_size if batch_size else self.batch_size
        
        q1_loss, q2_loss, pi_loss, alpha_loss, alpha_val = 0, 0, 0, 0, 0
        
        for _ in range(self.updates_per_step):
            q1, q2, pi, a_loss, a_val = self.update_parameters(bs)
            q1_loss += q1
            q2_loss += q2
            pi_loss += pi
            alpha_loss += a_loss
            alpha_val += a_val
            
        return {
            'loss': (q1_loss + q2_loss + pi_loss) / self.updates_per_step,
            'actor_loss': pi_loss / self.updates_per_step,
            'critic_loss': (q1_loss + q2_loss) / 2 / self.updates_per_step,
            'alpha_loss': alpha_loss / self.updates_per_step,
            'alpha': alpha_val / self.updates_per_step
        }

    def save(self, path, extra_info=None):
        # 1. Save Buffer (separate file)
        # Assuming path ends with .pth, we replace it for buffer
        buffer_path = path.replace('.pth', '_buffer.pkl')
        self.memory.save(buffer_path)
        
        # 2. Save Model
        state = {
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'policy_optim': self.policy_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'total_numsteps': self.total_numsteps,
        }
        if self.automatic_entropy_tuning:
            state['alpha_optim'] = self.alpha_optim.state_dict()
            state['log_alpha'] = self.log_alpha
            
        if extra_info:
            state.update(extra_info)
            
        torch.save(state, path)
        print(f"[NewAgent] Saved model to {path}")

    def load(self, path):
        # 1. Load Buffer
        buffer_path = path.replace('.pth', '_buffer.pkl')
        self.memory.load(buffer_path)
        
        # 2. Load Model
        if not os.path.exists(path):
            print(f"[NewAgent] No model file found at {path}")
            return
            
        print(f"[NewAgent] Loading model from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.policy_optim.load_state_dict(checkpoint['policy_optim'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim'])
        self.total_numsteps = checkpoint.get('total_numsteps', 0)
        
        if self.automatic_entropy_tuning and 'alpha_optim' in checkpoint:
            self.alpha_optim.load_state_dict(checkpoint['alpha_optim'])
            self.log_alpha = checkpoint['log_alpha']
            
        print(f"[NewAgent] Loaded model. Total Steps: {self.total_numsteps}")

    def _obs_to_vector(self, balls, my_targets):
        """Replicates BilliardGymEnv._get_obs logic"""
        ball_ids = ['cue'] + [str(i) for i in range(1, 16)]
        obs_vec = []
        for bid in ball_ids:
            if bid in balls:
                ball = balls[bid]
                # x, y, vx, vy
                obs_vec.extend([
                    ball.state.rvw[0][0],
                    ball.state.rvw[0][1],
                    ball.state.rvw[1][0],
                    ball.state.rvw[1][1]
                ])
            else:
                obs_vec.extend([0, 0, 0, 0])
                
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
        """Replicates BilliardGymEnv._denormalize_action logic"""
        v0 = 0.5 + (action[0] + 1) * 0.5 * (8.0 - 0.5)
        phi = 0 + (action[1] + 1) * 0.5 * 360
        theta = 0 + (action[2] + 1) * 0.5 * 90
        a = -0.5 + (action[3] + 1) * 0.5 * 1.0
        b = -0.5 + (action[4] + 1) * 0.5 * 1.0
        return {'V0': float(v0), 'phi': float(phi), 'theta': float(theta), 'a': float(a), 'b': float(b)}
