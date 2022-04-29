"""#Soft Actor Critic Demystified - [Explained](https://towardsdatascience.com/entropy-in-soft-actor-critic-part-1-92c2cd3a3515)"""

# Commented out IPython magic to ensure Python compatibility.
import math
import random
import os
import gym
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

"""##Utils Functions"""

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = int((self.position + 1) % self.capacity)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

class NormalizedActions(gym.ActionWrapper):
    def action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def reverse_action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return actions

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

"""##Network Definitions

###Critic Network
"""

class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size):
        super(SoftQNetwork, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1))
        self.q2 = nn.Sequential(
            nn.Linear(num_inputs + num_actions, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1))

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        return self.q1(state_action), self.q2(state_action)

"""###Policy Network"""

LOG_STD_MIN = -20
LOG_STD_MAX = 2

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3,
                 device=torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")):
        super(GaussianPolicy, self).__init__()

        self.device = device
        self.log_std_min = LOG_STD_MIN # log standard deviation
        self.log_std_max = LOG_STD_MAX

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        # For reparameterization trick (mean + std * N(0,1))
        z = normal.rsample()
        action = torch.tanh(z)
        action = action * self.action_scale + self.action_bias

        # Enforcing Action Bound
        log_prob = normal.log_prob(z) - torch.log(self.action_scale - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)

DIS_STD_MIN = -0.25
DIS_STD_MAX = 0.25

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3,
                 device=torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")):
        super(DeterministicPolicy, self).__init__()

        self.device = device
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.noise = torch.Tensor(num_actions)

        self.action_scale = torch.tensor(1.)
        self.action_bias = torch.tensor(0.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean_linear(x)) * self.action_scale + self.action_bias
        return mean

    def evaluate(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(DIS_STD_MIN, DIS_STD_MAX)
        action = mean + noise
        return action, torch.tensor(0.), mean
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)

"""##SAC Agent - [Pytorch SAC](https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py) - Building an algo for finding the best π(at|st)."""

soft_q_lr = 3e-4
policy_lr = 3e-3
ent_coef_lr = 3e-4

class SoftActorCritic(object):
    def __init__(self, policy, state_dim, action_dim, replay_buffer, hidden_dim=256,
                 device=torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")):

        self.device = device
        # Setup the network
        if policy == 'Gaussian':
            self.policy_net = GaussianPolicy(state_dim, action_dim, hidden_dim).to(device)
        elif policy == 'Deterministic':
            self.policy_net = DeterministicPolicy(state_dim, action_dim, hidden_dim).to(device)

        self.soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        self.checkpoint_dir = "./saved_models/"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Entropy coeff
        self.target_entropy = -action_dim
        self.log_ent_coef = torch.zeros(1, requires_grad=True, device=device)

        # set the losses
        self.soft_q_criterion = nn.MSELoss()

        # set the optimizers
        self.soft_q_optimizer = optim.Adam(self.soft_q_net.parameters(), lr=soft_q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
        self.ent_coef_optimizer = optim.Adam([self.log_ent_coef], lr=ent_coef_lr)

        hard_update(self.target_soft_q_net, self.soft_q_net)        

        # reference the replay buffer
        self.replay_buffer = replay_buffer

        self.log = {'entropy_loss': [], 'q_value_loss': [], 'policy_loss': []}

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action, _, _ = self.policy_net.evaluate(state)
        action = action.detach().cpu().numpy()
        action = action[0].tolist()
        action = action.index(max(action))
        return action

    def soft_q_update(self, batch_size, gamma=0.99, soft_tau=5e-3):
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        # Training Q function
        ent_coef = torch.exp(self.log_ent_coef.detach())
        with torch.no_grad():
            new_action, log_prob, mean = self.policy_net.evaluate(next_state)
            target_q1_value, target_q2_value = self.target_soft_q_net(next_state, new_action)
            # The Soft State-Value Function
            min_q_value = torch.min(target_q1_value, target_q2_value) - ent_coef * log_prob
            # The Soft Action-Value Function
            target_q_value = reward + (1 - done) * gamma * min_q_value

        predicted_q1_value, predicted_q2_value = self.soft_q_net(state, action)  # Two Q-functions to mitigate positive bias in the policy improvement step
        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        q1_loss = self.soft_q_criterion(predicted_q1_value, target_q_value.detach())
        q2_loss = self.soft_q_criterion(predicted_q2_value, target_q_value.detach())
        q_value_loss = q1_loss + q2_loss

        self.soft_q_optimizer.zero_grad()
        q_value_loss.backward()
        self.soft_q_optimizer.step()

        # Training Policy Function
        pi, log_pi, _ = self.policy_net.evaluate(state)
        expected_new_q1_pi, expected_new_q2_pi = self.soft_q_net(state, pi)

        expected_new_q_pi = torch.min(expected_new_q1_pi, expected_new_q2_pi)
        # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
        policy_loss = (ent_coef * log_pi - expected_new_q_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Entropy Adjusment for Maximum Entropy RL
        ent_loss = (self.log_ent_coef * (-log_pi - self.target_entropy).detach()).mean()
        self.ent_coef_optimizer.zero_grad()
        ent_loss.backward()
        self.ent_coef_optimizer.step()

        soft_update(self.target_soft_q_net, self.soft_q_net, soft_tau)

        self.log['q_value_loss'].append(q_value_loss.item())
        self.log['entropy_loss'].append(ent_loss.item())
        self.log['policy_loss'].append(policy_loss.item())

        # ent_coef = torch.exp(self.log_ent_coef.detach())
        # ent_coef_tlogs = ent_coef.clone() # For TensorboardX logs

    # Save model parameters
    def save_checkpoint(self,ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = self.checkpoint_dir + "/checkpoint.pkl"

        torch.save({'policy_state_dict': self.policy_net.state_dict(),
                    'critic_state_dict': self.soft_q_net.state_dict(),
                    'critic_target_state_dict': self.target_soft_q_net.state_dict(),
                    'critic_optimizer_state_dict': self.soft_q_optimizer.state_dict(),
                    'policy_optimizer_state_dict': self.policy_optimizer.state_dict()}, ckpt_path)

    # Load model parameters
    def load_checkpoint(self, ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = self.checkpoint_dir
        if not os.path.exists(ckpt_path):
            return
        
        print('Loading models from {}'.format(ckpt_path))
        checkpoint = torch.load(ckpt_path)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.soft_q_net.load_state_dict(checkpoint['critic_state_dict'])
        self.target_soft_q_net.load_state_dict(checkpoint['critic_target_state_dict'])
        self.soft_q_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])

        self.policy_net.evaluate()
        self.soft_q_net.evaluate()
        self.target_soft_q_net.evaluate()
       