import functools
import pprint
import matplotlib as plt

import re
import numpy as np
import pandas as pd
from gymnasium.spaces import Box

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn

from utils.data import date_to_index, index_to_date
from utils.common import get_history_and_abb

data_path = "data/S&P500/individual_stocks_5yr/individual_stocks_5yr"
stock_list = ['AAPL', 'ATVI', 'CMCSA', 'COST', 'CSX', 'DISH', 'EA', 'EBAY', 'FB', 'GOOGL']
eps = 1e-8

def load_sp10_data(data_path, stock_list) -> list[pd.DataFrame]:
    data = []
    for stock in stock_list:
        with open(f'{data_path}/{stock}_data.csv', 'rb') as f:
            data.append(pd.read_csv(f))

    return data


def random_shift(x, fraction):
    """ Apply a random shift to a pandas series. """
    min_x, max_x = np.min(x), np.max(x)
    m = np.random.uniform(-fraction, fraction, size=x.shape) + 1
    return np.clip(x * m, min_x, max_x)


def scale_to_start(x):
    """ Scale pandas series so that it starts at one. """
    x = (x + eps) / (x[0] + eps)
    return x


def sharpe(returns, freq=30, rfr=0):
    """ Given a set of returns, calculates naive (rfr=0) sharpe (eq 28). """
    return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)


def max_drawdown(returns):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    peak = returns.max()
    trough = returns[returns.argmax():].min()
    return (trough - peak) / (peak + eps)


class DataGenerator(object):
    """Acts as data provider for each new episode."""

    def __init__(self, history, abbreviation, steps=730, window_length=50, start_idx=0, start_date=None):
        """

        Args:
            history: (num_stocks, timestamp, 5) open, high, low, close, volume
            abbreviation: a list of length num_stocks with assets name
            steps: the total number of steps to simulate, default is 2 years
            window_length: observation window, must be less than 50
            start_date: the date to start. Default is None and random pick one.
                        It should be a string e.g. '2012-08-13'
        """
        assert history.shape[0] == len(abbreviation), 'Number of stock is not consistent'
        import copy

        self.steps = steps + 1
        self.window_length = window_length
        self.start_idx = start_idx
        self.start_date = start_date

        # make immutable class
        self._data = history.copy()  # all data
        self.asset_names = copy.copy(abbreviation)

    def step(self):
        # get observation matrix from history, exclude volume, maybe volume is useful as it
        # indicates how market total investment changes. Normalize could be critical here
        self.step_count += 1
        obs = self.data[:, self.step_count:self.step_count + self.window_length, :].copy()
        # normalize obs with open price

        # used for compute optimal action and sanity check
        ground_truth_obs = self.data[:, self.step_count + self.window_length:self.step_count + self.window_length + 1, :].copy()

        done = self.step_count >= self.steps
        return obs, done, ground_truth_obs

    def reset(self):
        self.step_count = 0

        # get data for this episode, each episode might be different.
        if self.start_date is None:
            self.idx = np.random.randint(
                low=self.window_length, high=self._data.shape[1] - self.steps)
        else:
            # compute index corresponding to start_date for repeatable sequence
            self.idx = date_to_index(self.start_date) - self.start_idx
            assert self.idx >= self.window_length and self.idx <= self._data.shape[1] - self.steps, \
                'Invalid start date, must be window_length day after start date and simulation steps day before end date'
        # print('Start date: {}'.format(index_to_date(self.idx)))
        data = self._data[:, self.idx - self.window_length:self.idx + self.steps + 1, :4]
        # apply augmentation?
        self.data = data
        return self.data[:, self.step_count:self.step_count + self.window_length, :].copy(), \
               self.data[:, self.step_count + self.window_length:self.step_count + self.window_length + 1, :].copy()
    
    # 允许不进行step而获得观察值
    def observe(self):
        return self.data[:, self.step_count:self.step_count + self.window_length, :].copy()


class PortfolioSim(object):
    """
    Portfolio management sim.
    Params:
    - cost e.g. 0.0025 is max in Poliniex
    Based of [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    def __init__(self, asset_names=list(), steps=730, trading_cost=0.0025, time_cost=0.0):
        self.asset_names = asset_names
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps

    def _step(self, w1, y1):
        """
        Step.
        w1 - new action of portfolio weights - e.g. [0.1,0.9,0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1]
        Numbered equations are from https://arxiv.org/abs/1706.10059
        """
        assert w1.shape == y1.shape, 'w1 and y1 must have the same shape'
        assert y1[0] == 1.0, 'y1[0] must be 1'

        p0 = self.p0

        dw1 = (y1 * w1) / (np.dot(y1, w1) + eps)  # (eq7) weights evolve into

        mu1 = self.cost * (np.abs(dw1 - w1)).sum()  # (eq16) cost to change portfolio

        assert mu1 < 1.0, 'Cost is larger than current holding'

        p1 = p0 * (1 - mu1) * np.dot(y1, w1)  # (eq11) final portfolio value

        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding

        rho1 = p1 / p0 - 1  # rate of returns
        r1 = np.log((p1 + eps) / (p0 + eps))  # log rate of return
        reward = r1 / self.steps * 1000.  # (22) average logarithmic accumulated return
        # remember for next step
        self.p0 = p1

        # if we run out of money, we're done (losing all the money)
        done = p1 == 0

        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": mu1,
        }
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.p0 = 1.0


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    target_history, target_stocks = get_history_and_abb()
    env = raw_env(render_mode=internal_render_mode, 
                  history=target_history,
                  stock_abbreviation=target_stocks)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    # env = wrappers.AssertOutOfBoundsWrapper(env)
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
    """

    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "simple_market",
        "is_parallelizable": True,
    }

    # 所有的必要参数都应该被放置在有默认值的参数之前
    def __init__(self, 
                 history,
                 stock_abbreviation,
                 steps=730,   # 2 years
                 render_mode=None, 
                 num_agents=1,
                 num_steps=10e2,
                 trading_cost=0.0025,
                 time_cost=0.00,
                 window_length=50,
                 start_idx=0,
                 sample_start_date=None):
        """
        An environment for financial portfolio management.
        
        Params:
            steps - steps in episode
            scale - scale data and each episode (except return)
            augment - fraction to randomly shift data by
            trading_cost - cost of trade as a fraction
            time_cost - cost of holding as a fraction
            window_length - how many past observations to return
            start_idx - The number of days from '2012-08-13' of the dataset
            sample_start_date - The start date sampling from the history
        """
        self.window_length = window_length
        self.num_stocks = history.shape[0]
        self.start_idx = start_idx

        self.src = DataGenerator(history, 
                                 stock_abbreviation, 
                                 steps=steps, 
                                 window_length=window_length,
                                 start_idx=start_idx,
                                 start_date=sample_start_date)
        
        self.sim = PortfolioSim(asset_names=stock_abbreviation,
                                trading_cost=trading_cost,
                                time_cost=time_cost,
                                steps=steps)

        self.stock_data = load_sp10_data(data_path, stock_list)
        self.num_steps = num_steps
        self.timestep = 0
        
        self.agent_num = num_agents
        self.agents = [f"trader_{i}" for i in range(self.agent_num)]
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self.kill_list = []
        self.dead_agents = []
        
        # 动作即为每个资产所占的比重
        # 初始化要用到np.ndarray
        action_space = Box(low=0, 
                           high=1, 
                           shape=(len(self.src.asset_names) + 1,),  # 注意双逗号的意义
                           dtype=np.float32)
        self.action_spaces = {a: action_space for a in self.agents}

        # 观察空间
        obs_space = Box(low=-np.inf, 
                        high=np.inf, 
                        shape=(len(stock_abbreviation), window_length, history.shape[-1]),
                        dtype = np.float32)
        self.observation_spaces = {a: obs_space for a in self.agents}

        self.rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: [] for a in self.agents}
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
        self.timestep = 0

        self.agents = self.possible_agents[:]
        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: [] for a in self.agents}
        self.kill_list = []
        self.dead_agents = []

        # 基于self.agents的必须在self.agent完成初始化后再初始化
        self.trader_amounts = {a: 0 for a in self.agents}
        self.trader_prices = {a: 0 for a in self.agents}
        self.trader_sum_earn = {a: 0 for a in self.agents}

        self._agent_selector.reinit(self.agents)
        self._agent_selector.reset()
        self.agent_selection = self._agent_selector.reset()

        self.sim.reset()
        _, _ = self.src.reset()

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
            return self._was_dead_step(action)

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

        observation, done1, ground_truth_obs = self.src.step()

        # concatenate observation with ones
        cash_observation = np.ones((1, self.window_length, observation.shape[2]))
        observation = np.concatenate((cash_observation, observation), axis=0)

        cash_ground_truth = np.ones((1, 1, ground_truth_obs.shape[2]))
        ground_truth_obs = np.concatenate((cash_ground_truth, ground_truth_obs), axis=0)

        # relative price vector of last observation day (close/open)
        close_price_vector = observation[:, -1, 3]
        open_price_vector = observation[:, -1, 0]
        y1 = close_price_vector / open_price_vector
        reward, info, done2 = self.sim._step(weights, y1)

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos[agent] + [info]])[-1]
        # add dates
        info['date'] = index_to_date(self.start_idx + self.src.idx + self.src.step_count)
        info['steps'] = self.src.step_count
        info['next_obs'] = ground_truth_obs

        # TODO 此处瞎写的
        self.rewards[agent] = reward
        self.terminations[agent] = done1 or done2
        # self.truncations[agent] = done2
        # TODO info无限增长，暂时没什么用不启用了
        # self.infos[agent].append(info)

        # TODO 实现保证金机制
        # 奖励小于特定值的时候破产
        # 在处理完当前代理的所有动作后暂时存入kill_list
        # 待到所有代理都处理完后再移除该代理
        # if self.trader_sum_earn[agent] < -5:
        #     self.kill_list.append(agent)
            
        # 每轮循环结束后：
        if self._agent_selector.is_last():
            self.timestep += 1
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

            # 从存活代理上重新初始化代理选择器
            self._agent_selector.reinit(_live_agents)

        # 若仍有存活的代理，那么获取下一个
        if len(self._agent_selector.agent_order):
            self.agent_selection = self._agent_selector.next()

        # TODO 超时检测的实现机制暂不清楚
        truncate = self.timestep >= self.num_steps
        self.truncations = {a: truncate for a in self.agents}

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
        从self.src获取当前时间步的观测值
        """
        return self.src.observe()
    
    def _get_last_day_price(self, agent):
        # 利用正则匹配寻找代理对应的股票下标
        index = int(re.compile(r'\d+').search(agent).group())
        return self.stock_data[index]["close"][self.timestep]
    
    def plot(self, agent):
        # show a plot of portfolio vs mean market performance
        df_info = pd.DataFrame(self.infos[agent])
        df_info['date'] = pd.to_datetime(df_info['date'], format='%Y-%m-%d')
        df_info.set_index('date', inplace=True)
        mdd = max_drawdown(df_info.rate_of_return + 1)
        sharpe_ratio = sharpe(df_info.rate_of_return)
        title = 'max_drawdown={: 2.2%} sharpe_ratio={: 2.4f}'.format(mdd, sharpe_ratio)
        df_info[["portfolio_value", "market_value"]].plot(title=title, fig=plt.gcf(), rot=30)

