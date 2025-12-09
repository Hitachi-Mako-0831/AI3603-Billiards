import torch
import torch.nn as nn
import numpy as np
from ..base import BasePolicy
from ..registry import register_algorithm
from .model import ActorCritic

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

@register_algorithm("ppo")
class PPOAgent(BasePolicy):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        # Hyperparameters
        self.lr = cfg.get('lr', 3e-4)
        self.gamma = cfg.get('gamma', 0.99)
        self.eps_clip = cfg.get('eps_clip', 0.2)
        self.K_epochs = cfg.get('k_epochs', 4)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        self.state_dim = 79 # Fixed from wrapper
        self.action_dim = 5 # Fixed from wrapper
        
        self.policy = ActorCritic(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        self.policy_old = ActorCritic(self.state_dim, self.action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer = RolloutBuffer()
        self.mse_loss = nn.MSELoss()
        
        print(f"PPO Agent initialized on {self.device}")

    def select_action(self, state, evaluate=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy_old.act(state)
        
        if not evaluate:
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)
            
        return action.cpu().data.numpy().flatten()

    def store_transition(self, reward, is_terminal):
        """
        Helper to store reward and terminal flag after step()
        """
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(is_terminal)

    def train(self, replay_buffer=None, batch_size=None):
        # PPO uses its own internal buffer, ignoring external replay_buffer arg for now
        # Calculate Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        if rewards.numel() > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # Convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(state_values, rewards) - 0.01 * dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Clear buffer
        self.buffer.clear()
        
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
