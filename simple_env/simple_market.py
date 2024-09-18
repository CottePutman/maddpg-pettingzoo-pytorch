import functools
import pickle
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, Box
from simple_stock import SimpleStock

from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers, parallel_to_aec


def load_simple_data() -> list:
    with open('data/test/SH600015_100.pkl', 'rb') as f:
        price = pickle.load(f)

    data = {'$close0': [i for i in price]}
    return data


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    """元数据储存环境的常量。

    "name"元数据可以让环境优雅地被显示。
    """

    metadata = {
        "render_modes": ["human"],
        "name": "simple_market",
    }


    # def __init__(self, stock: SimpleStock):
    #     """初始化交易环境。

    #     其应当明确股票数据集位置
    #     """
    #     ##TODO 实现多支股票
    #     self.stock_data = stock
    #     self.num_steps = len(stock)
    #     self.timestep = None
    #     self.possible_agents = ["trader", "invalid"]
    #     self.trader_amount = None
    #     self.trader_price = None
    #     self.rewards = None
    #     self._cumulative_rewards = None


    def __init__(self, render_mode=None):
        """初始化交易环境。

        只加载简单的一百条价格数据
        """
        ##TODO 实现多支股票
        self.stock_data = load_simple_data()
        self.num_steps = len(self.stock_data["$close0"])
        self.timestep = None
        self.render_mode = render_mode
        
        self.possible_agents = ["trader", "invalid"]
        self.trader_amount = None
        self.trader_price = None
        self.rewards = None
        self._cumulative_rewards = None


    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # 定义金融数据的三个维度：最高价、最低价、成交量
        # return Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        # 观察值包含三个维度：当前价格、自身持有量、自身购入价、盈利情况
        return Box(low=0, high=np.inf, shape=(4,), dtype=np.float32)
    

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        定义动作空间: 0表示跳过, 1表示买入, 2表示平仓。
        """
        return Discrete(3)
    
    
    def reset(self, seed=None, options=None):
        """将环境重置到初始状态。

        应当初始化下述属性：
        - 代理 
        - 时间步（每天的开市时间是确定的，原则上每天的时间步总长度也是确定的）
        - 观测值
        - 信息

        """
        ##TODO 随机选择股票
        self.agents = self.possible_agents[:]
        self.timestep = 0
        self.trader_amount = 0
        self.trader_price = 0
        self.rewards = { a: 0 for a in self.agents }
        self._cumulative_rewards = { a: 0 for a in self.agents }

        # 获取当前股市开盘位置信息
        observation = { a: self._get_observation() for a in self.agents }

        observations = {
            a :{
                "observation": observation,
                "action_mask": [1, 1, 0]        # 初始不允许平仓
            }
            for a in self.agents
        }

        # 获取虚拟信息。对parallel_to_aec而言是必须的
        infos = {a: {} for a in self.agents}

        return observations, infos


    #TODO 非常粗糙
    def step(self, actions):
        """处理当前代理做出的动作 (通过agent_selection指定)。

        需要更新以下内容:
        - 代理
        - 时间步
        - 观测值
        - 信息

        以及其他被observe()和render()用到的数据
        """
        # 执行动作
        action = actions["trader"]
        step_reward = {a: 0 for a in self.agents}

        #TODO 仿照Qlib对交易行为以及带来的奖励进行处理
        if action == 0:     # 跳过
            pass
        elif action == 1 and self.trader_amount == 0:   # 买入
            self.trader_amount = 100                    # 暂时固定为买入100单位，TODO
            self.trader_price = self._get_observation()[0]
        elif action == 2 and self.trader_amount > 0:    # 平仓
            earn = (self._get_observation()[0] - self.trader_price) * self.trader_amount
            self._cumulative_rewards["trader"] += earn
            step_reward["trader"] = earn
            self.trader_amount = 0
            self.trader_price = 0

        trader_action_mask = np.ones(3, dtype=np.int8)
        if self.trader_amount == 0:
            trader_action_mask[2] = 0   # 禁止平仓
        elif self.trader_amount > 0:
            trader_action_mask[1] = 0   # 禁止买入

        observation = { a: self._get_observation() for a in self.agents }
        observations = {
            a: {
                "observation": observation,
                "action_mask": trader_action_mask
            }
            for a in self.agents
        }

        terminations = {a: False for a in self.agents}
        infos = {a: {} for a in self.agents}

        truncations = {a: False for a in self.agents}
        self.timestep += 1
        if self.timestep >= self.num_steps:
            truncations = {a: True for a in self.agents}
            self.agents = []

        return observations, step_reward, terminations, truncations, infos


    def render(self):
        """渲染当前的环境状态。
        """
        print(f"Step: {self.timestep}, Data: {self._get_observation()}")

    
    def _get_observation(self):
        """获取当前时间步的观测值（如最高价、最低价和成交量）。
        """
        cur_price = self.stock_data["$close0"][self.timestep]  # 假设数据为NumPy数组，形状为(num_steps, 3)
        obs = (cur_price, self.trader_amount, self.trader_price, self._cumulative_rewards["trader"])
        return obs