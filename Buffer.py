import numpy as np
import torch


class Buffer:
    """replay buffer for each agent"""

    def __init__(self, capacity, obs_shape, act_shape, device):
        self.capacity = capacity

        # 多维向量需要展开成单维（注意双括号）
        # 针对动作向量，则需要进行独热编码
        self.obs = np.zeros((capacity, np.prod(obs_shape)))
        self.action = np.zeros((capacity, np.prod(act_shape)))
        self.reward = np.zeros(capacity)
        self.next_obs = np.zeros((capacity, np.prod(obs_shape)))
        self.terminations = np.zeros(capacity, dtype=bool)
        self.truncations = np.zeros(capacity, dtype=bool)

        self._index = 0
        self._size = 0

        self.device = device

    def add(self, obs, action, reward, next_obs, truncation, termination):
        """ add an experience to the memory """
        # 将所有传入的动作、观察值展开
        self.obs[self._index] = obs.flatten()
        self.action[self._index] = action.flatten()
        self.reward[self._index] = reward
        self.next_obs[self._index] = next_obs.flatten()
        self.truncations[self._index] = truncation
        self.terminations[self._index] = termination

        self._index = (self._index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, indices):
        # retrieve data, Note that the data stored is ndarray
        obs = self.obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        next_obs = self.next_obs[indices]
        truncation = self.truncations[indices]
        termination = self.terminations[indices]

        # NOTE that `obs`, `action`, `next_obs` will be passed to network(nn.Module),
        # so the first dimension should be `batch_size`
        obs = torch.from_numpy(obs).float().to(self.device)  # torch.Size([batch_size, state_dim])
        action = torch.from_numpy(action).float().to(self.device)  # torch.Size([batch_size, action_dim])
        reward = torch.from_numpy(reward).float().to(self.device)  # just a tensor with length: batch_size
        # reward = (reward - reward.mean()) / (reward.std() + 1e-7)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)  # Size([batch_size, state_dim])
        truncation = torch.from_numpy(truncation).float().to(self.device)  # just a tensor with length: batch_size
        termination = torch.from_numpy(termination).float().to(self.device)

        return obs, action, reward, next_obs, truncation, termination

    def __len__(self):
        return self._size
