import argparse
import yaml
import os
import matplotlib.pyplot as plt
import numpy as np
from MADDPG import MADDPG
from utils.env import get_env


def load_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', dest="config_path", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config_path)
    
    config_env = config['environment']
    env_name = config_env['env_name']
    act_type = config_env['act_type'] or None
    
    config_train = config['training']
    episode_num = config_train['episode_num']
    episode_length = config_train['episode_length']
    learn_interval = config_train['learn_interval']
    random_steps = config_train['random_steps']
    tau = config_train['tau']
    gamma = config_train['gamma']
    buffer_capacity = config_train['buffer_capacity']
    batch_size = config_train['batch_size']
    actor_lr = config_train['actor_lr']
    critic_lr = config_train['critic_lr']

    # create folder to save result
    env_dir = os.path.join('./results', env_name)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    result_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(result_dir)

    # MADDPG似乎不能支持二维观察值的初始化
    env, dim_info = get_env(env_name, episode_length, config_env)
    maddpg = MADDPG(dim_info,
                    act_type, 
                    buffer_capacity, 
                    batch_size, 
                    actor_lr, 
                    critic_lr,
                    result_dir)

    step = 0  # global step counter
    # agent_num = env.num_agents
    # reward of each episode of each agent
    episode_rewards = {agent_id: np.zeros(episode_num) for agent_id in env.agents}
    for episode in range(episode_num):
        observations, infos = env.reset()
        agent_reward = {agent_id: 0 for agent_id in env.agents}  # agent reward of the current episode
        while env.agents:  # interact with the env for an episode
            step += 1      
            # 最开始是进行随机决策，到达一定步数之后才换为maddpg的决策  
            if step < random_steps:
                actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            else:
                actions = maddpg.select_action(observations)

            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            # 启用terminations和truncations
            maddpg.add(observations, actions, rewards, next_observations, truncations, terminations) 

            for agent_id, r in rewards.items():  # update reward
                agent_reward[agent_id] += r

            if step >= random_steps and step % learn_interval == 0:  # learn every few steps
                maddpg.learn(batch_size, gamma)
                maddpg.update_target(tau)

            observations = next_observations

        # episode finishes
        for agent_id, r in agent_reward.items():  # record reward
            episode_rewards[agent_id][episode] = r

        if (episode + 1) % 10 == 0:  # print info every few episodes
            message = f'episode {episode + 1}, '
            sum_reward = 0
            for agent_id, r in agent_reward.items():  # record reward
                message += f'{agent_id}: {r:>4f}; '
                sum_reward += r
            message += f'sum reward: {sum_reward}'
            print(message)

    maddpg.save(episode_rewards)  # save model


    def get_running_reward(arr: np.ndarray, window=100):
        """calculate the running reward, i.e. average of last `window` elements from rewards"""
        running_reward = np.zeros_like(arr)
        for i in range(window - 1):
            running_reward[i] = np.mean(arr[:i + 1])
        for i in range(window - 1, len(arr)):
            running_reward[i] = np.mean(arr[i - window + 1:i + 1])
        return running_reward


    # training finishes, plot reward
    fig, ax = plt.subplots()
    x = range(1, episode_num + 1)
    for agent_id, reward in episode_rewards.items():
        ax.plot(x, reward, label=agent_id)
        ax.plot(x, get_running_reward(reward))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maddpg solve {env_name}'
    ax.set_title(title)
    plt.savefig(os.path.join(result_dir, title))
