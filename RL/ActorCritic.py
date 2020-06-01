import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

device = torch.device("cuda:0")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, 40),
                nn.LeakyReLU(),
                nn.Linear(40, 28),
                nn.LeakyReLU(),
                nn.Linear(28, 14),
                nn.LeakyReLU(),
                nn.Linear(14, action_dim),
                nn.Hardtanh(min_val=-3.14, max_val=3.14)
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 40),
                nn.LeakyReLU(),
                nn.Linear(40, 28),
                nn.LeakyReLU(),
                nn.Linear(28, 10),
                nn.LeakyReLU(),
                nn.Linear(10, 5),
                nn.LeakyReLU(),
                nn.Linear(5, 1),
            )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)
        
        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy