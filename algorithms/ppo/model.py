import torch
import torch.nn as nn
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(ActorCritic, self).__init__()
        
        # Optimized Network Architecture:
        # 1. Increased width (256 -> 512) for better representation
        # 2. Increased depth (2 -> 3 hidden layers)
        # 3. Orthogonal initialization (standard practice in PPO)
        
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), # Added extra layer
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
            nn.Linear(hidden_dim, hidden_dim), # Added extra layer
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Orthogonal Initialization
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        
        # Special init for Actor output (makes initial policy close to random/zero mean)
        # The last layer of actor determines the mean action. 
        # Low gain (0.01) makes initial actions close to 0, promoting exploration via std
        nn.init.orthogonal_(self.actor[-2].weight, gain=0.01)
        
        # Special init for Critic output
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
        
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
