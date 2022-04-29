# -*- coding: utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()
        state_dim = int(state_dim);hidden_dim = int(hidden_dim);action_dim = int(action_dim);
        # actor
        self.actor = nn.Sequential(
                        nn.Linear(in_features = state_dim, out_features = hidden_dim),
                        nn.Tanh(),
                        nn.Linear(in_features = hidden_dim, out_features = hidden_dim),
                        nn.Tanh(),
                        nn.Linear(in_features = hidden_dim, out_features = action_dim),
                        nn.Softmax(dim=-1)
                    )

        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(in_features = state_dim, out_features = hidden_dim),
                        nn.Tanh(),
                        nn.Linear(in_features = hidden_dim, out_features = hidden_dim),
                        nn.Tanh(),
                        nn.Linear(in_features = hidden_dim, out_features = 1),
                    )

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

    def to(self, device):
        return super(ActorCritic, self).to(device)