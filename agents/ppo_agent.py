import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from .agent import Agent


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1.0)
        torch.nn.init.constant_(m.bias, 0.0)


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, log_std_min: float, log_std_max: float):
        super().__init__()
        self.actor_body = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_log_std = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor):
        h = self.actor_body(state)
        mean = self.actor_mean(h)
        log_std = torch.clamp(self.actor_log_std(h), self.log_std_min, self.log_std_max)
        value = self.critic(state)
        return mean, log_std, value

    def act(self, state: torch.Tensor, deterministic: bool):
        mean, log_std, value = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        if deterministic:
            raw_action = mean
        else:
            raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return action, log_prob, entropy, value

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor):
        mean, log_std, value = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = torch.clamp(action, -1.0 + 1e-6, 1.0 - 1e-6)
        raw_action = 0.5 * (torch.log1p(action) - torch.log1p(-action))
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        entropy = dist.entropy().sum(-1, keepdim=True)
        return log_prob, entropy, value


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def __len__(self):
        return len(self.rewards)


class PPOAgent(Agent):
    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            cfg = {
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "clip_range": 0.2,
                "entropy_coef": 0.0,
                "value_coef": 0.5,
                "max_grad_norm": 0.5,
                "lr": 3e-4,
                "hidden_size": 256,
                "update_epochs": 10,
                "minibatch_size": 256,
                "log_std_min": -5.0,
                "log_std_max": 2.0,
                "seed": 42,
                "device": "auto",
            }

        self.cfg = cfg
        self.gamma = float(cfg["gamma"])
        self.gae_lambda = float(cfg["gae_lambda"])
        self.clip_range = float(cfg["clip_range"])
        self.entropy_coef = float(cfg["entropy_coef"])
        self.value_coef = float(cfg["value_coef"])
        self.max_grad_norm = float(cfg["max_grad_norm"])
        self.lr = float(cfg["lr"])
        self.hidden_size = int(cfg["hidden_size"])
        self.update_epochs = int(cfg["update_epochs"])
        self.minibatch_size = int(cfg["minibatch_size"])
        self.log_std_min = float(cfg["log_std_min"])
        self.log_std_max = float(cfg["log_std_max"])
        self.seed = int(cfg["seed"])

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if cfg.get("device", "auto") == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg["device"])

        self.state_dim = 79
        self.action_dim = 5

        self.model = ActorCritic(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_size=self.hidden_size,
            log_std_min=self.log_std_min,
            log_std_max=self.log_std_max,
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.buffer = RolloutBuffer()
        self.total_numsteps = 0

        self._try_load_default()

    def _try_load_default(self):
        default_path = "checkpoints/ppo_final.pth"
        if os.path.exists(default_path):
            self.load(default_path)

    def _obs_to_vector(self, balls, my_targets):
        ball_ids = ["cue"] + [str(i) for i in range(1, 16)]
        obs_vec = []
        for bid in ball_ids:
            if bid in balls:
                ball = balls[bid]
                obs_vec.extend(
                    [
                        ball.state.rvw[0][0],
                        ball.state.rvw[0][1],
                        ball.state.rvw[1][0],
                        ball.state.rvw[1][1],
                    ]
                )
            else:
                obs_vec.extend([0, 0, 0, 0])

        target_vec = [1.0 if str(i) in my_targets else 0.0 for i in range(1, 16)]
        obs_vec.extend(target_vec)
        return np.asarray(obs_vec, dtype=np.float32)

    def _denormalize_action(self, action):
        v0 = 0.5 + (action[0] + 1.0) * 0.5 * (8.0 - 0.5)
        phi = (action[1] + 1.0) * 0.5 * 360.0
        theta = (action[2] + 1.0) * 0.5 * 90.0
        a = -0.5 + (action[3] + 1.0) * 0.5 * 1.0
        b = -0.5 + (action[4] + 1.0) * 0.5 * 1.0
        return {"V0": float(v0), "phi": float(phi), "theta": float(theta), "a": float(a), "b": float(b)}

    @torch.no_grad()
    def select_action(self, state_vec: np.ndarray, evaluate: bool):
        state = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        action, log_prob, _, value = self.model.act(state, deterministic=evaluate)
        action = action.squeeze(0).cpu().numpy()
        log_prob = log_prob.item()
        value = value.item()
        return action, log_prob, value

    def decision(self, balls, my_targets, table):
        state_vec = self._obs_to_vector(balls, my_targets)
        action_norm, _, _ = self.select_action(state_vec, evaluate=True)
        return self._denormalize_action(action_norm)

    def store_transition(self, obs_vec, action_norm, log_prob, reward, done, value):
        self.buffer.add(obs_vec, action_norm, log_prob, reward, done, value)
        self.total_numsteps += 1

    def _compute_gae_and_returns(self, last_value: float):
        rewards = np.asarray(self.buffer.rewards, dtype=np.float32)
        dones = np.asarray(self.buffer.dones, dtype=np.float32)
        values = np.asarray(self.buffer.values, dtype=np.float32)
        next_values = np.concatenate([values[1:], np.asarray([last_value], dtype=np.float32)], axis=0)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (1.0 - dones[t]) * next_values[t] - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values
        return advantages, returns

    def update(self, last_obs_vec=None, last_done: bool = True):
        if len(self.buffer) == 0:
            return {}

        if last_obs_vec is not None and not last_done:
            _, _, last_value = self.select_action(last_obs_vec, evaluate=True)
        else:
            last_value = 0.0

        advantages, returns = self._compute_gae_and_returns(last_value=last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.as_tensor(np.asarray(self.buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(self.buffer.actions), dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(np.asarray(self.buffer.log_probs), dtype=torch.float32, device=self.device).unsqueeze(-1)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device).unsqueeze(-1)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(-1)

        n = states.shape[0]
        indices = np.arange(n)

        last_losses = {}
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                mb_idx = indices[start : start + self.minibatch_size]
                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_advantages = advantages_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                new_log_probs, entropy, values = self.model.evaluate_actions(mb_states, mb_actions)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = 0.5 * (mb_returns - values).pow(2).mean()
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                last_losses = {
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.mean().item()),
                }

        self.buffer.reset()
        return last_losses

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "total_numsteps": self.total_numsteps,
                "cfg": dict(self.cfg),
            },
            path,
        )

    def load(self, path: str):
        if not os.path.exists(path):
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_numsteps = int(checkpoint.get("total_numsteps", 0))

