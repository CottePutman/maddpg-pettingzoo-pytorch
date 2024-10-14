import numpy as np
from utils.data import date_to_index


eps = 1e-8


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

    def __init__(self, 
                 history, 
                 abbreviation, 
                 trade_steps=730, 
                 window_length=1, 
                 start_idx=0, 
                 start_date=None):
        """

        Args:
            history: (num_stocks, timestamp, 5: (open, high, low, close, volume))
            abbreviation: a list of length num_stocks with assets name
            steps: the total number of steps to simulate, default is 2 years
            window_length: observation window, must be less than 50
            start_date: the date to start. Default is None and random pick one.
                        It should be a string e.g. '2012-08-13'
        """
        assert(history.shape[0] == len(abbreviation)), 'Number of stock is not consistent'
        import copy

        self.num_steps = trade_steps + 1
        self.window_length = window_length
        self.start_idx = start_idx
        self.start_date = start_date

        # make immutable class
        self._data = history.copy()  # all data
        self.asset_names = copy.copy(abbreviation)

        self.is_reset = False

    def step(self, forward=True):
        # get observation matrix from history, exclude volume, maybe volume is useful as it
        # indicates how market total investment changes. Normalize could be critical here
        if forward:
            self.step_count += 1
        obs = self.data[:, self.step_count:self.step_count + self.window_length, :].copy()
        # normalize obs with open price

        # used for compute optimal action and sanity check
        ground_truth_obs = self.data[:, self.step_count + self.window_length:self.step_count + self.window_length + 1, :].copy()

        # 最大可模拟范围超时检测
        truncation = self.step_count >= self.num_steps

        return obs, truncation, ground_truth_obs

    def reset(self):
        self.step_count = 0

        # get data for this episode, each episode might be different.
        if self.start_date is None:
            self.idx = np.random.randint(
                low=self.window_length, high=self._data.shape[1] - self.num_steps)
        else:
            # compute index corresponding to start_date for repeatable sequence
            self.idx = date_to_index(self.start_date) - self.start_idx
            assert self.idx >= self.window_length and self.idx <= self._data.shape[1] - self.num_steps, \
                'Invalid start date, must be window_length day after start date and simulation steps day before end date'
        # print('Start date: {}'.format(index_to_date(self.idx)))
        data = self._data[:, self.idx - self.window_length:self.idx + self.num_steps + 1, :]
        # apply augmentation?
        self.data = data

        self.is_reset = True
        return self.data[:, self.step_count:self.step_count + self.window_length, :].copy(), \
               self.data[:, self.step_count + self.window_length:self.step_count + self.window_length + 1, :].copy()
    
    # 允许不进行step而获得观察值
    # 此处就是仅返回obs，别瞎几把改
    def observe(self):
        obs = self.data[:, self.step_count:self.step_count + self.window_length, :].copy()
        return obs
    
    def next_observe(self):
        if self.step_count >= self.num_steps:
            return self.observe()
        
        obs = self.data[:, self.step_count + 1:self.step_count + self.window_length + 1, :].copy()
        return obs


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
        self.infos = []
        self.p0 = 1.0   # Total portfolio value

    def step(self, w1, y1):
        """
        Step.
        
        w1: new action of portfolio weights - e.g. [0.1,0.9,0.0] (weights for each asset)
        y1: price relative vector also called return, e.g. [1.0, 0.9, 1.1] (price relative change for each asset)

        Returns:
        - rewards: A reward for each asset based on its performance
        - info: Dictionary of portfolio information
        - terminate: Boolean indicating if the portfolio is bankrupt
        
        Numbered equations are from [this paper](https://arxiv.org/abs/1706.10059).
        """
        # The first asset in the portfolio is special, that it is the quoted currency, 
        # referred to as the cash for the rest of the article.
        # 动作w1的第一个dim实际表示不投资，把钱抓在手上
        # 这也是为什么y1[0]永远是1，暂不考虑通胀
        assert w1.shape == y1.shape, 'w1 and y1 must have the same shape'
        assert y1[0] == 1.0, 'y1[0] must be 1'

        p0 = self.p0

        dw1 = (y1 * w1) / (np.dot(y1, w1) + eps)  # (eq7) weights evolve into

        mu1 = self.cost * (np.abs(dw1 - w1)).sum()  # (eq16) cost to change portfolio

        assert mu1 < 1.0, 'Cost is larger than current holding'

        p1 = p0 * (1 - mu1) * np.dot(y1, w1)  # (eq11) final portfolio value

        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding

        rho1 = y1 - 1  # rate of returns for each asset
        r1 = np.log((y1 + eps) / 1)  # log rate of return
        
        # Calculate portfolio-level reward, including transcation cost
        portfolio_reward = np.log((p1 + eps) / (p0 + eps)) / self.steps * 1000.

        rewards = r1 / self.steps * 1000.  # (22) average logarithmic accumulated return
        
        # remember for next step
        self.p0 = p1

        # if we run out of money, we're done (losing all the money)
        terminate = p1 <= 0

        info = {
            "reward": rewards,
            "log_return": r1,
            "portfolio_value": p1,
            "return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": mu1,
        }
        # self.infos.append(info)
        
        # TODO 惩罚风险行为
        # 此处暂时以方差代替
        variance = np.var(w1[1:])
        if variance > 0.1:
            portfolio_reward -= variance

        return portfolio_reward, rewards, info, terminate

    def reset(self):
        self.infos = []
        self.p0 = 1.0