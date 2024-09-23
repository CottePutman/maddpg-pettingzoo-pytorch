import functools
import pickle
from copy import copy

import re
import numpy as np
import pandas as pd
from gymnasium.spaces import Discrete, Box, Dict

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn

data_path = "data/S&P500/individual_stocks_5yr/individual_stocks_5yr"
stock_list = ['AAPL', 'ATVI', 'CMCSA', 'COST', 'CSX', 'DISH', 'EA', 'EBAY', 'FB', 'GOOGL']


def load_simple_data() -> list:
    with open('data/test/SH600015_100.pkl', 'rb') as f:
        price = pickle.load(f)

    data = {'$close0': [i for i in price]}
    return data


def load_sp500_data() -> pd.DataFrame:
    with open('data/S&P500/all_stocks_5yr.csv', 'rb') as f:
        data = pd.read_csv(f)
    
    return data


def load_sp10_data(data_path, stock_list) -> list[pd.DataFrame]:
    data = []
    for stock in stock_list:
        with open(f'{data_path}/{stock}_data.csv', 'rb') as f:
            data.append(pd.read_csv(f))

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

parallel_env = parallel_wrapper_fn(env)

class raw_env(AECEnv):
    """元数据储存环境的常量。

    "name"元数据可以让环境优雅地被显示。
    """

    metadata = {
        "render_modes": ["human"],
        "name": "simple_market",
        "is_parallelizable": True,
    }


    def __init__(self, 
                 render_mode=None, 
                 num_agents=5,
                 num_steps=10e2):
        """初始化交易环境。

        加载十支不同的股票
        """
        self.stock_data = load_sp10_data(data_path, stock_list)
        # self.num_steps = len(self.stock_data[0]["date"])        # 取APPLE的日期为总步数
        self.num_steps = num_steps
        self.timestep = 0
        
        self.agent_num = num_agents
        self.agents = [f"trader_{i}" for i in range(self.agent_num)]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.kill_list = []
        self.dead_agents = []
        
        #TODO 实现做多、做空
        self.action_spaces = {a: Discrete(3) for a in self.agents}      # 动作空间即买入、卖出、跳过
        # self.observation_spaces = {
        #     a: Dict(
        #         {  
        #             # 观察空间为一个三维向量，分别为当前价格、买入量、买入价格
        #             "observation": Box(low=0, high=np.inf, shape=(3,), dtype=np.float32),
        #             # 防止代理做出非法动作
        #             "action_mask": Box(low=0, high=1, shape=(9,), dtype=np.int8)
        #         }
        #     )
        #     for a in self.agents
        # }
        
        # 与PettingZoo的原生API保持一致
        obs_space = Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        # self.observation_spaces = dict(
        #     zip(
        #         self.agents,
        #         [obs_space for _ in enumerate(self.agents)]
        #     )
        # )
        self.observation_spaces = {a: obs_space for a in self.agents}

        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.trader_amounts = {a: 0 for a in self.agents}
        self.trader_prices = {a: 0 for a in self.agents}
        self.trader_sum_earn = {a: 0 for a in self.agents}

        self.render_mode = render_mode


    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    
    def reset(self, seed=None, options=None):
        """将环境重置到初始状态。

        应当初始化下述属性：
        - 代理 
        - 时间步（每天的开市时间是确定的，原则上每天的时间步总长度也是确定的）
        - 观测值
        - 信息

        """
        #TODO 随机选择股票
        #TODO 累计收益
        self.timestep = 0

        self.agents = self.possible_agents[:]
        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.kill_list = []
        self.dead_agents = []

        # 基于self.agents的必须在self.agent完成初始化后再初始化
        self.trader_amounts = {a: 0 for a in self.agents}
        self.trader_prices = {a: 0 for a in self.agents}
        self.trader_sum_earn = {a: 0 for a in self.agents}

        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()


    #TODO 非常粗糙
    def step(self, action):
        """处理当前代理做出的动作 (通过agent_selection指定)。

        需要更新以下内容:
        - 代理
        - 时间步
        - 观测值
        - 信息

        以及其他被observe()和render()用到的数据
        """ 
        agent = self.agent_selection      
        if (
            self.terminations[agent]
            or self.truncations[agent]
        ):
            # 该方法会从各类数据中将已死亡代理的信息删除
            return self._was_dead_step(action)

        # 需要依靠_cumulative_rewards来判断是否破产了
        # 每轮agent_select完成后会返回所有代理的累计奖励
        # 故上一轮的遗留数据需要被清零
        self._cumulative_rewards[agent] = 0
        self._clear_rewards()

        # 执行动作
        #TODO 仿照Qlib对交易行为以及带来的奖励进行处理
        if action == 0:     # 跳过
            pass
        elif action == 1 and self.trader_amounts[agent] == 0:        # 买入
            self.trader_amounts[agent] = 1                           # 暂时固定为买入1单位，TODO
            self.trader_prices[agent] = self._get_last_day_price(agent)
        elif action == 2 and self.trader_amounts[agent] > 0:         # 平仓
            earn = (self._get_last_day_price(agent) - self.trader_prices[agent]) * self.trader_amounts[agent]
            self.rewards[agent] += earn
            self.trader_sum_earn[agent] += earn
            self.trader_amounts[agent] = 0
            self.trader_prices[agent] = 0

        # TODO 实现保证金机制
        # 奖励小于特定值的时候破产
        # 在处理完当前代理的所有动作后暂时存入kill_list
        # 待到所有代理都处理完后再移除该代理
        if self.trader_sum_earn[agent] < -5:
            self.kill_list.append(agent)
            
        # 每轮循环结束后
        # manage the kill list
        if self._agent_selector.is_last():
            self.timestep += 1
            # start iterating on only the living agents
            _live_agents = self.agents[:]
            for k in self.kill_list:
                # kill the agent
                _live_agents.remove(k)
                # set the termination for this agent for one round
                self.terminations[k] = True
                # add that we know this guy is dead
                self.dead_agents.append(k)

            # reset the kill list
            self.kill_list = []

            # reinit the agent selector with existing agents
            self._agent_selector.reinit(_live_agents)

        # if there still exist agents, get the next one
        if len(self._agent_selector.agent_order):
            self.agent_selection = self._agent_selector.next()

        # 超时检测
        truncate = self.timestep >= self.num_steps
        self.truncations = {a: truncate for a in self.agents}

        self._accumulate_rewards()
        self._deads_step_first()


    def render(self):
        """
        渲染当前的环境状态。
        """
        print(f"Step: {self.timestep}")

    
    def observe(self, agent):
        """
        获取当前时间步的观测值（如最高价、最低价和成交量）。
        """
        # TODO self.trader_amounts会莫名其妙全空
        cur_price = self._get_last_day_price(agent)       # 假设数据为NumPy数组，形状为(num_steps, 3)
        obs = [cur_price, self.trader_amounts[agent], self.trader_prices[agent]]

        return np.array(obs)
    

    def _get_last_day_price(self, agent):
        # 利用正则匹配寻找代理对应的股票下标
        index = int(re.compile(r'\d+').search(agent).group())
        return self.stock_data[index]["close"][self.timestep]