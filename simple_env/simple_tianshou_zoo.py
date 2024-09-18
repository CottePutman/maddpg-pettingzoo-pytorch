"""This is a minimal example of using Tianshou with MARL to train agents.

Author: Will (https://github.com/WillDudley)

Python version used: 3.8.10

Requirements:
pettingzoo == 1.22.0
git+https://github.com/thu-ml/tianshou
"""

import os
from typing import Optional, Tuple

import gymnasium
import numpy as np
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import BasePolicy, DQNPolicy, MultiAgentPolicyManager, RandomPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.net.common import Net

import simple_aec_market
import paper_rock
from pettingzoo.classic import tictactoe_v3
from pettingzoo.butterfly import knights_archers_zombies_v10


def _get_agents(
    num_agents = 10,
    agent_learn: Optional[BasePolicy] = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = _get_env()
    observation_space = (
        env.observation_space["observation"]
        if isinstance(env.observation_space, gymnasium.spaces.Dict)
        else env.observation_space
    )

    # 存储所有代理的策略
    policies = {}
    for i in range(num_agents):
        agent_name = f"trader_{i}"
        
        if agent_learn is None:
            # 为每个代理创建单独的网络
            net = Net(
                state_shape=observation_space.shape or observation_space.n,
                action_shape=env.action_space.n,
                hidden_sizes=[128, 128, 128, 128],
                device="cuda" if torch.cuda.is_available() else "cpu",
            ).to("cuda" if torch.cuda.is_available() else "cpu")

            # 初始化优化器（可共享）
            if optim is None:
                optim = torch.optim.Adam(net.parameters(), lr=1e-4)
            
            # 为每个代理创建一个DQN策略
            agent_policy = DQNPolicy(
                model=net,
                optim=optim,
                discount_factor=0.9,
                estimation_step=3,
                target_update_freq=320,
                action_space=env.action_space,
            )
        else:
            agent_policy = agent_learn

        policies[agent_name] = agent_policy  # 存储代理策略

    # 使用 MultiAgentPolicyManager 管理所有代理的策略
    policy = MultiAgentPolicyManager(policies=policies, env=env)
    
    return policy, optim, env.agents


def _get_env():
    """This function is needed to provide callables for DummyVectorEnv."""
    # return PettingZooEnv(knights_archers_zombies_v10.env())
    return PettingZooEnv(simple_aec_market.env())
    # return PettingZooEnv(tictactoe_v3.env())
    # return PettingZooEnv(paper_rock.env())


if __name__ == "__main__":
    # ======== Step 1: Environment setup =========
    train_envs = DummyVectorEnv([_get_env for _ in range(1)])
    test_envs = DummyVectorEnv([_get_env for _ in range(1)])

    # seed
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    # ======== Step 2: Agent setup =========
    policy, optim, agents = _get_agents()

    # ======== Step 3: Collector setup =========
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(20_000, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=64 * 10, reset_before_collect=True)  # batch size * training_num

    # ======== Step 4: Callback functions setup =========
    def save_best_fn(policy):
        model_save_path = os.path.join("log", "ttt", "dqn", "policy.pth")
        os.makedirs(os.path.join("log", "ttt", "dqn"), exist_ok=True)
        torch.save(policy.policies[agents[1]].state_dict(), model_save_path)

    def stop_fn(mean_rewards):
        return mean_rewards >= 0.6

    def train_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.1)

    def test_fn(epoch, env_step):
        policy.policies[agents[1]].set_eps(0.05)

    def reward_metric(rews):
        return rews[:, 1]

    # ======== Step 5: Run the trainer =========
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=50,
        step_per_epoch=1000,
        step_per_collect=50,
        episode_per_test=10,
        batch_size=64,
        train_fn=train_fn,
        test_fn=test_fn,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        update_per_step=0.1,
        test_in_train=False,
        reward_metric=reward_metric,
    ).run()

    # return result, policy.policies[agents[1]]
    print(f"\n==========Result==========\n{result}")
    print("\n(the trained policy can be accessed via policy.policies[agents[1]])")