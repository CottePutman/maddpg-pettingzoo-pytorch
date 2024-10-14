from copy import deepcopy
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import Adam


class Agent:
    """Agent that can interact with environment from pettingzoo"""

    def __init__(self, obs_dim, act_dim, global_obs_dim, actor_lr, critic_lr, device, act_type='discrete'):
        self.device = device

        # critic input all the observations and actions
        # if there are 3 agents for example, the input for critic is (obs1, obs2, obs3, act1, act2, act3)
        # Critic应当学习代理所有的动作和观察值来生成一个预测的Q值
        self.actor = MLPNetwork(np.prod(obs_dim), np.prod(act_dim)).to(self.device)
        self.critic = MLPNetwork(global_obs_dim, 1).to(self.device)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor = deepcopy(self.actor)
        self.target_critic = deepcopy(self.critic)
        self.act_type = act_type

        # 退火算法
        # 用于控制过于鬼屎的初始权重导致的类似one-hot的动作向量
        self.softmax_temp = 2.0     # 从高温开始
        self.min_temp = 1.0         # 降温
        self.anneal_rate = 0.995    # Multiplicative factor for annealing

    @staticmethod
    def gumbel_softmax(self, logits, tau=1.0, eps=1e-20):
        # NOTE that there is a function like this implemented in PyTorch(torch.nn.functional.gumbel_softmax),
        # but as mention in the doc, it may be removed in the future, so i implement it myself
        # tau是温度参数，用于控制softmax得到的值的均匀程度，或称为平滑程度
        # tau越大分布越温和，越小则越尖锐
        epsilon = torch.rand_like(logits, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        logits += -torch.log(-torch.log(epsilon + eps) + eps)
        return F.softmax(logits / tau, dim=-1)

    def action(self, obs, model_out=False):
        # this method is called in the following two cases:
        # a) interact with the environment
        # b) calculate action when update actor, where input(obs) is sampled from replay buffer with size:
        # torch.Size([batch_size, state_dim])

        obs = obs.to(self.device)
        logits = self.actor(obs)  # torch.Size([batch_size, action_size])
        # action = self.gumbel_softmax(logits)
        if self.act_type == 'discrete':
            action = F.gumbel_softmax(logits / self.softmax_temp, hard=True)
        elif self.act_type == 'continue':
            action = F.softmax(logits / self.softmax_temp, dim=-1)

        if model_out:
            return action, logits
        return action.round(decimals=4)

    def target_action(self, obs):
        # when calculate target critic value in MADDPG,
        # we use target actor to get next action given next states,
        # which is sampled from replay buffer with size torch.Size([batch_size, state_dim])

        obs = obs.to(self.device)
        logits = self.target_actor(obs)  # torch.Size([batch_size, action_size])
        # action = self.gumbel_softmax(logits)
        if self.act_type == 'discrete':
            action = F.gumbel_softmax(logits / self.softmax_temp, hard=True)
        elif self.act_type == 'continue':
            action = F.softmax(logits / self.softmax_temp, dim=-1)
        
        return action.squeeze(0).detach().round(decimals=4)

    def critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        state_list = [s.to(self.device) for s in state_list]    # 移动state_list到GPU
        act_list = [a.to(self.device) for a in act_list]        # 移动act_list到GPU
        # print([s.device for s in state_list], [a.device for a in act_list])
        x = torch.cat(state_list + act_list, 1)
        return self.critic(x).squeeze(1)                        # tensor with a given length

    def target_critic_value(self, state_list: List[Tensor], act_list: List[Tensor]):
        state_list = [s.to(self.device) for s in state_list]    # 移动state_list到GPU
        act_list = [a.to(self.device) for a in act_list]        # 移动act_list到GPU
        x = torch.cat(state_list + act_list, 1)
        return self.target_critic(x).squeeze(1)  # tensor with a given length

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        self.softmax_temp = max(self.softmax_temp * self.anneal_rate, self.min_temp)

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        self.softmax_temp = max(self.softmax_temp * self.anneal_rate, self.min_temp)


class MLPNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, non_linear=nn.ReLU()):
        super(MLPNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, hidden_dim),
            non_linear,
            nn.Linear(hidden_dim, out_dim),
        ).apply(self.init)

    @staticmethod
    def init(m):
        """init parameter of the module"""
        gain = nn.init.calculate_gain('relu')
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        # 展平张量
        x = x.view(x.size(0), -1)
        return self.net(x)
