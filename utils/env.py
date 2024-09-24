from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3
from pettingzoo.butterfly import knights_archers_zombies_v10
from portfolio import PortfolioEnv
from simple_env import simple_aec_market, simple_pm


def get_env(env_name, ep_len=25, act_type: dict = None, softmax: list = None):
    """
    create environment and get observation and action dimension of each agent in this environment
    """
    new_env = None
    if env_name == 'zombie':
        # render_mode="human"会大幅拉低训练速度
        new_env = knights_archers_zombies_v10.parallel_env(render_mode="human",
                                                           num_archers=2,
                                                           num_knights=2)
    elif env_name == 'simple_adversary_v2':
        new_env = simple_adversary_v3.parallel_env(render_mode="rgb_array")
    elif env_name == 'simple_spread_v2':
        new_env = simple_spread_v3.parallel_env(max_cycles=ep_len)
    elif env_name == 'simple_tag_v2':
        new_env = simple_tag_v3.parallel_env(render_mode="rgb_array",
                                             max_cycles=ep_len)
    elif env_name == 'port':
        new_env = PortfolioEnv()
    elif env_name == 'market':
        new_env = simple_aec_market.parallel_env(render_mode='human')
    elif env_name == 'pm':
        new_env = simple_pm.parallel_env(render_mode='human')

    new_env.reset(seed=42)
    _dim_info = {}
    for agent_id in new_env.agents:
        _dim_info[agent_id] = []  # [(obs_shape), (act_shape), type, softmax]
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape or new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).shape or new_env.action_space(agent_id).n)
        # act_type为None则默认所有代理均为离散动作
        # 若all关键字存在，则所有都按照all的类型进行设置
        # 若all不存在，则根据config中为每个代理单独设置的类型进行设置
        # TODO 应当需要实现批量化操作，思路是命名的规范化与正则化匹配
        if act_type is None:
            _dim_info[agent_id].append('discrete')
        elif 'all' in act_type.keys():
            _dim_info[agent_id].append(act_type['all'])
        else:
            _dim_info[agent_id].append(act_type[agent_id])
            
        _dim_info[agent_id].append(softmax)

    return new_env, _dim_info