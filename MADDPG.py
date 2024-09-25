import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

from Agent import Agent
from Buffer import Buffer
from utils.common import softmax_and_mapping


def setup_logger(filename):
    """ set up logger with filename. """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(filename, mode='w')
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s--%(levelname)s--%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger


class MADDPG:
    """A MADDPG(Multi Agent Deep Deterministic Policy Gradient) agent"""

    def __init__(self, dim_info, act_type, capacity, batch_size, actor_lr, critic_lr, res_dir):
        # sum all the dims of each agent to get input dim for critic
        # 调用np.prod()对多维值进行相乘操作,val[0]是观察空间，val[1]是动作空间
        global_obs_act_dim = sum(np.prod(val[0]) for val in dim_info.values()) + sum(np.prod(val[1]) for val in dim_info.values())
        # create Agent(actor-critic) and replay buffer for each agent
        self.agents = {}
        self.buffers = {}
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = 'cpu'

        for agent_id, (obs_dim, act_dim, act_type, softmax) in dim_info.items():
            # TODO 根据dim_info的act_type进行网络的选择
            self.agents[agent_id] = Agent(obs_dim, act_dim, global_obs_act_dim, actor_lr, critic_lr, self.device, act_type)
            self.buffers[agent_id] = Buffer(capacity, obs_dim, act_dim, self.device)
        self.dim_info = dim_info
        self.act_type = act_type
        self.softmax = softmax

        self.batch_size = batch_size
        self.res_dir = res_dir  # directory to save the training result
        self.logger = setup_logger(os.path.join(res_dir, 'maddpg.log'))


    def add(self, observations, actions, rewards, next_observations, truncations, terminations):
        # NOTE that the experience is a dict with agent name as its key
        for agent_id in next_observations.keys():
            # 终止/超时检测
            # 代理死亡时，其信息不会再被包含在env.reset()的返回值中，
            # 而obs本质上是上一步的观察值，即代理死亡前的观察值，因而仍然包含有相关信息
            # 但actions, rewards, next_observations, termination如何对np.array进行softmax
            # if agent_id not in terminations.keys() or agent_id not in truncations.keys():
            #     # self.buffers[agent_id].add(obs, 0, 0, obs, True, True)
            #     continue
            
            obs = observations[agent_id]
            action = actions[agent_id]

            # 若是连续型的动作则无视该步骤
            # 此处将离散值转换为独热编码的操作非常重要，关乎后续的maddpg.learn函数
            # TODO 此处仅仅非常粗糙地一律将离散动作以int32型进行独热编码
            # Ensure action is always treated as an int64, even if int is returned
            if isinstance(action, int) or isinstance(action, np.int64):
                # the action from env.action_space.sample() is int, we have to convert it to onehot
                # Convert action to int64 before converting to one-hot
                action = np.eye(self.dim_info[agent_id][1])[np.int32(action)]
            
            reward = rewards[agent_id]
            next_obs = next_observations[agent_id]
            truncation = truncations[agent_id]
            termination = terminations[agent_id]
            self.buffers[agent_id].add(obs, action, reward, next_obs, truncation, termination)


    def sample(self, batch_size):
        """sample experience from all the agents' buffers, and collect data for network input"""
        # get the total num of transitions, these buffers should have same number of transitions
        total_num = len(self.buffers[next(iter(self.agents))])
        indices = np.random.choice(total_num, size=batch_size, replace=False)

        # NOTE that in MADDPG, we need the obs and actions of all agents
        # but only the reward and done of the current agent is needed in the calculation
        obs, act, reward, next_obs, trunction, termination, next_act = {}, {}, {}, {}, {}, {}, {}
        for agent_id, buffer in self.buffers.items():
            o, a, r, n_o, tru, ter = buffer.sample(indices)
            obs[agent_id] = o.to(self.device)
            act[agent_id] = a.to(self.device)
            reward[agent_id] = r.to(self.device)
            next_obs[agent_id] = n_o.to(self.device)
            trunction[agent_id] = tru.to(self.device)
            termination[agent_id] = ter.to(self.device)
            # calculate next_action using target_network and next_state
            next_act[agent_id] = self.agents[agent_id].target_action(n_o).to(self.device)

        return obs, act, reward, next_obs, trunction, termination, next_act

    def select_action(self, obs):
        actions = {}
        for agent, o in obs.items():
            o = torch.from_numpy(o).unsqueeze(0).float().to(self.device)
            a = self.agents[agent].action(o)  # torch.Size([1, action_size])
            # NOTE that the output is a tensor, convert it to int before input to the environment
            # 支持自动判断连续、离散动作的输出
            if self.act_type == 'discrete':
                actions[agent] = a.squeeze(0).argmax().item()
            elif self.act_type == 'continue':
                actions[agent] = a.squeeze(0).detach().cpu().numpy()
                # 连续值的情况下却不设置softmax值，需要警报用户
                if self.softmax is None:
                    raise UserWarning(f"Agent {agent}'s actions are continuous but softmax is not set.")
                # 对softmax进行规范化的步骤已经在get_env中进行处理过
                actions[agent] = softmax_and_mapping(actions[agent], self.softmax)
            else:
                raise ValueError(f"Unsupported action space type: {self.act_type}")
            
            self.logger.info(f'{agent} action: {actions[agent]}')
        return actions

    def learn(self, batch_size, gamma):
        for agent_id, agent in self.agents.items():
            obs, act, reward, next_obs, truncation, termination, next_act = self.sample(batch_size)
            # update critic
            critic_value = agent.critic_value(list(obs.values()), list(act.values()))

            # calculate target critic value
            next_target_critic_value = agent.target_critic_value(list(next_obs.values()),
                                                                 list(next_act.values()))
            target_value = reward[agent_id] + gamma * next_target_critic_value * (1 - truncation[agent_id]) * (1 - termination[agent_id])

            critic_loss = F.mse_loss(critic_value, target_value.detach(), reduction='mean')
            agent.update_critic(critic_loss)

            # update actor
            # action of the current agent is calculated using its actor
            action, logits = agent.action(obs[agent_id], model_out=True)
            act[agent_id] = action
            actor_loss = -agent.critic_value(list(obs.values()), list(act.values())).mean()
            actor_loss_pse = torch.pow(logits, 2).mean()
            agent.update_actor(actor_loss + 1e-3 * actor_loss_pse)
            # self.logger.info(f'agent{i}: critic loss: {critic_loss.item()}, actor loss: {actor_loss.item()}')

    def update_target(self, tau):
        def soft_update(from_network, to_network):
            """ copy the parameters of `from_network` to `to_network` with a proportion of tau"""
            for from_p, to_p in zip(from_network.parameters(), to_network.parameters()):
                to_p.data.copy_(tau * from_p.data + (1.0 - tau) * to_p.data)

        for agent in self.agents.values():
            soft_update(agent.actor, agent.target_actor)
            soft_update(agent.critic, agent.target_critic)

    def save(self, reward):
        """save actor parameters of all agents and training reward to `res_dir`"""
        torch.save(
            {name: agent.actor.state_dict() for name, agent in self.agents.items()},  # actor parameter
            os.path.join(self.res_dir, 'model.pt')
        )
        with open(os.path.join(self.res_dir, 'rewards.pkl'), 'wb') as f:  # save training data
            pickle.dump({'rewards': reward}, f)

    @classmethod
    def load(cls, dim_info, file):
        """init maddpg using the model saved in `file`"""
        instance = cls(dim_info, 0, 0, 0, 0, os.path.dirname(file))
        data = torch.load(file)
        for agent_id, agent in instance.agents.items():
            agent.actor.load_state_dict(data[agent_id])
        return instance
