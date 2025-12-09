import torch
import torch.nn as nn
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared features or separate? Let's use separate for simplicity
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh() # Output range [-1, 1] for mean
        )
        
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_mean = self.actor(state)
        action_std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        
        return action, action_logprob
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        action_std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        action_logprobs = dist.log_prob(action).sum(dim=-1)
        dist_entropy = dist.entropy().sum(dim=-1)
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy
