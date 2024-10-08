import functools
import pprint
import matplotlib as plt

import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn

from utils.data import index_to_date
from utils.common import get_history_and_abb

from simple_env.data_gen_sim import DataGenerator, PortfolioSim
from simple_env.data_gen_sim import max_drawdown, sharpe

from tpg.TPG import TemporalPortfolioGraph


# 用于防止0成为除数的微小量
eps = 1e-8


def env(render_mode=None, **kwargs):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    target_history, target_stocks = get_history_and_abb()
    
    # 注意**kwargs的用法
    kwargs['render_mode'] = internal_render_mode
    kwargs['history'] = target_history
    kwargs['abbreviation'] = target_stocks
    env = raw_env(**kwargs)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # 适用于连续动作空间的包装器
    env = wrappers.ClipOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv):
    """
    An environment for financial portfolio management.
    
    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.
    
    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059) 
    and is modified based on [Yang 2023](https://doi.org/10.1016/j.knosys.2023.110905)
    """
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "simple_market",
        "is_parallelizable": True,
    }

    # 所有的必要参数都应该被放置在有默认值的参数之前
    def __init__(self, 
                 history,
                 abbreviation,
                 steps=730,   # 2 years
                 render_mode=None, 
                 num_agents=2,
                 trading_cost=0.0025,
                 time_cost=0.00,
                 window_length=1,
                 start_idx=0,
                 sample_start_date=None,
                 embedding_dim=10):
        """
        An environment for financial portfolio management.
        
        Args:
        -   steps: steps in episode
        -   scale: scale data and each episode (except return)
        -   augment: fraction to randomly shift data by
        -   trading_cost: cost of trade as a fraction
        -   time_cost: cost of holding as a fraction
        -   window_length: how many past observations to return
        -   start_idx: The number of days from '2012-08-13' of the dataset
        -   sample_start_date: The start date sampling from the history
        -   embedding_dim: 最终的嵌入Z_t维度，也是观察空间的维度
        """
        if window_length != 1: raise RuntimeWarning("Window length should be 1.")
        
        self.window_length = window_length
        self.num_asset = history.shape[0]
        self.start_idx = start_idx

        # TODO 模仿MultiModel，为每个代理分配不同的DataGenerator与Sim
        # TODO 每个代理应该有自己的TPG，或者是所有资产在一起共用一张TPG？
        # 目前的多代理仍然是共用同一个Sim
        self.src = DataGenerator(history, 
                                 abbreviation, 
                                 trade_steps=steps, 
                                 window_length=window_length,
                                 start_idx=start_idx,
                                 start_date=sample_start_date)
        
        self.sim = PortfolioSim(asset_names=abbreviation,
                                trading_cost=trading_cost,
                                time_cost=time_cost,
                                steps=steps)
        
        self.tpg = TemporalPortfolioGraph(output_dim=embedding_dim)
    
        self.agent_num = num_agents
        self.agents = [f"trader_{i}" for i in range(self.agent_num)]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.kill_list = []
        self.dead_agents = []
        
        # 动作即为每个资产所占的比重
        # 初始化要用到np.ndarray
        # 动作空间的第一个维度表示不投资，把钱以现金的形式抓在手上
        action_space = Box(low=0, 
                           high=1, 
                           shape=(len(self.src.asset_names) + 1,),  # 注意双逗号的意义
                           dtype=np.float32)
        self.action_spaces = {a: action_space for a in self.agents}

        # 状态空间
        # 根据论文中的数据，似乎并没有观察窗口这一说法
        # 所有的state都是当前时间的各类数据
        # 修改后的state形状为(num_asset, 1, [open, close, high, low, volume])
        # 以这种形式暂时保留观察窗口的存在
        state_space = Box(low=0, 
                          high=np.inf, 
                          shape=(len(abbreviation), window_length, history.shape[-1]),
                          dtype = np.float32)
        self.state_spaces = {a: state_space for a in self.agents}

        # 观察空间
        # 按照论文，整个模型的观察空间即一个256维的联合嵌入向量Z_t
        obs_space = Box(low=-np.inf,
                        high=np.inf,
                        shape=(embedding_dim, 1),
                        dtype=np.float32)
        self.observation_spaces = {a: obs_space for a in self.agents}

        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: [] for a in self.agents}

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
        # TODO 考虑更多的重置可能性，如随机化股票
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: [] for a in self.agents}
        self.kill_list = []
        self.dead_agents = []

        self.sim.reset()
        self.src.reset()
        self.tpg.reset(self.num_asset, [self.src.observe(), self.src.next_observe()])

        # 基于self.agents的必须在self.agent完成初始化后再初始化
        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()

    def step(self, action):
        """
        Step the env.
        
        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight from 0 to 1. The first is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        - 需要注意，由于本环境为连续动作环境，会直接以Agent类的Actor最后一层logits输出作为action，那么就需要在某处加上softmax，否则会直接被clip_out_of_bounds裁切
        """
        agent = self.agent_selection      
        if (
            self.terminations[agent]
            or self.truncations[agent]
        ):
            # 该方法会从各类数据中将已死亡代理的信息删除
            # 注意return的位置
            self._was_dead_step(action)
            return

        # 每轮agent_select完成后会返回所有代理的累计奖励
        # 故上一轮的遗留数据需要被清零
        self._cumulative_rewards[agent] = 0
        self._clear_rewards()

        np.testing.assert_almost_equal(
            action.shape,
            (len(self.sim.asset_names) + 1)
        )

        # 规范化，以防万一
        action = np.clip(action, 0, 1)
        
        weights = action    # [w0, w1...]
        weights /= (weights.sum() + eps)
        weights[0] += np.clip(1 - weights.sum(), 0, 1)      # 若动作全为0那就allin第一个资产

        # and操作符不可用于array，此处使用*操作符
        # 检查action/weights中的所有值都符合要求
        assert ((action >= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
        np.testing.assert_almost_equal(
            np.sum(weights), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights)

        # TODO 还是得实现每个代理使用不同的Sim或者DG
        # 下述的observation等和实际传递给policy的observation并不是一个东西
        # 由于PettingZoo默认期待所有代理在一个cycle中都要执行动作
        # 而若只使用一个DataGenerator就会导致当前一个代理已经将其step推到底后
        # 下一个代理仍然会想继续访问数据，这就导致数据异常
        # 必须要每个代理使用不同的Generator或保证其只更新一次
        # 仅在为第一个代理时才置forward为True
        # ground_truth_obs似乎没有作用
        observation, truncation, ground_truth_obs = self.src.step(forward = self._agent_selector.is_first())

        # concatenate observation with ones
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)

        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)

        # relative price vector of last observation day (close/open)
        close_price_vector = observation[:, -1, 3]
        open_price_vector = observation[:, -1, 0]
        y1 = close_price_vector / open_price_vector
        pf_reward, rewards, info, termination = self.sim.step(weights, y1)
  
        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos[agent] + [info]])[-1]
        # add dates
        info['date'] = index_to_date(self.start_idx + self.src.idx + self.src.step_count)
        info['steps'] = self.src.step_count
        info['next_obs'] = ground_truth_obs

        # TODO 还要为每个资产分配奖励！
        # 代理的总奖励等于所有资产的奖励之和（收益）
        self.rewards[agent] = pf_reward

        # TODO 多代理情况
        # 更新TPG
        obs = self.src.observe()
        act = action
        rwd = rewards
        next_obs = self.src.next_observe()
        node_features = [obs, act, rwd, next_obs]
        self.tpg.update(node_features)

        # 资产为0时死亡
        # 在处理完当前代理的所有动作后暂时存入kill_list
        # 待到所有代理都处理完后再移除该代理
        if termination:
            self.kill_list.append(agent)
            
        # 每轮循环结束后：
        # 1. 处理破产代理
        if self._agent_selector.is_last():
            # 仅从存活代理开始迭代
            # 管理死亡代理列表
            _live_agents = self.agents[:]
            for k in self.kill_list:
                # 杀死代理
                _live_agents.remove(k)
                # 标记终止符
                self.terminations[k] = True
                # 将其添加入死亡列表，方便直接访问
                self.dead_agents.append(k)

            # 重置每轮的死亡代理列表
            self.kill_list = []

            # 若发生超时，则一定是所有代理同时超时
            self.truncations = {a: truncation for a in self.agents}

            # 从存活代理上重新初始化代理选择器
            self._agent_selector.reinit(_live_agents)

        # 若仍有存活的代理，那么获取下一个
        if len(self._agent_selector.agent_order):
            self.agent_selection = self._agent_selector.next()

        # TODO info无限增长，暂时没什么用不启用了
        # self.infos[agent].append(info)

        self._accumulate_rewards()
        self._deads_step_first()

    def render(self, close=False):
        """
        渲染当前的环境状态。
        """
        if close:
            return
        if self.render_mode == 'ansi':
            for a in self.agents:
                pprint(self.infos[a][-1])
        elif self.render_mode == 'human':
            for a in self.agents:
                self.render(a)
    
    def observe(self, agent):
        """
        从TPG的GCN模型获得当前时间步的嵌入
        """
        return self.tpg.observe()
    
    def plot(self, agent):
        # show a plot of portfolio vs mean market performance
        df_info = pd.DataFrame(self.infos[agent])
        df_info['date'] = pd.to_datetime(df_info['date'], format='%Y-%m-%d')
        df_info.set_index('date', inplace=True)
        mdd = max_drawdown(df_info.rate_of_return + 1)
        sharpe_ratio = sharpe(df_info.rate_of_return)
        title = 'max_drawdown={: 2.2%} sharpe_ratio={: 2.4f}'.format(mdd, sharpe_ratio)
        df_info[["portfolio_value", "market_value"]].plot(title=title, fig=plt.gcf(), rot=30)

