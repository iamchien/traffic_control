# -*- coding: utf-8 -*-
import math
import random
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import torch
from torch import nn
import copy
from collections import deque
import random
import gym
from tqdm import tqdm
import numpy as np

class DQN_Agent:
    def __init__(self, seed, layer_sizes, lr, sync_freq, exp_replay_size,
                 device=torch.device( "cuda:0" if torch.cuda.is_available() else "cpu")):
        self.device = device
        layer_sizes = [int(x) for x in layer_sizes]
        exp_replay_size = int(exp_replay_size); sync_freq = int(sync_freq); seed = int(seed);
        torch.manual_seed(seed)
        self.q_net = self.build_nn(layer_sizes).to(device)
        self.target_net = copy.deepcopy(self.q_net).to(device)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.gamma = torch.FloatTensor(0.95).to(device)
        self.experience_replay = deque(maxlen=exp_replay_size)
        return

    def build_nn(self, layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
            act = nn.Tanh() if index < len(layer_sizes) - 2 else nn.Identity()
            layers += (linear, act)
        return nn.Sequential(*layers)

    def get_action(self, state, action_space_len, epsilon):
        # We do not require gradient at this point, because this function will be used either
        # during experience collection or during inference
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            Qp = self.q_net(state)

        Q, A = torch.max(Qp, axis=0)
        A = A if torch.rand(1, ).item() > epsilon else torch.randint(0, action_space_len, (1,))
        return A.item()

    def get_q_next(self, state):
        with torch.no_grad():
            qp = self.target_net(state)
        q, _ = torch.max(qp, axis=1)
        return q

    def collect_experience(self, experience):
        self.experience_replay.append(experience)
        return

    def sample_from_experience(self, sample_size):
        if len(self.experience_replay) < sample_size:
            sample_size = len(self.experience_replay)
        sample = random.sample(self.experience_replay, sample_size)
        s = torch.FloatTensor(np.array([exp[0] for exp in sample])).to(self.device)
        a = torch.FloatTensor(np.array([exp[1] for exp in sample])).to(self.device)
        rn = torch.FloatTensor(np.array([exp[2] for exp in sample])).to(self.device)
        sn = torch.FloatTensor(np.array([exp[3] for exp in sample])).to(self.device)
        return s, a, rn, sn

    def train(self, batch_size):
        s, a, rn, sn = self.sample_from_experience(sample_size=batch_size)
        if self.network_sync_counter == self.network_sync_freq:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.network_sync_counter = 0

        # predict expected return of current state using main network
        qp = self.q_net(s.to(self.device))
        pred_return, _ = torch.max(qp, axis=1)

        # get target return using target network
        q_next = self.get_q_next(sn.to(self.device))
        target_return = rn.to(self.device) + self.gamma * q_next

        loss = self.loss_fn(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer.step()

        self.network_sync_counter += 1
        return loss.item()