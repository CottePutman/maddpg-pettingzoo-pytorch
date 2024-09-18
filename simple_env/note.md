```
Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
Spaces should be defined in the action_space() and observation_space() methods.
```
至少对于Parallel环境来说，使用对应的函数来定义观察和动作空间。

<br/>

```
{'agent_id': 'player_1', 'obs': array([[[0, 0],
        [0, 0],
        [0, 0]],

       [[0, 0],
        [0, 0],
        [0, 0]],

       [[0, 0],
        [0, 0],
        [0, 0]]], dtype=int8), 'mask': [True, True, True, True, True, True, True, True, True]}
```
传入Tianshou Batch的数据应该是以代理id作为下标的数组，字典的键信息放在外面。如上，但这是ACE环境，不知道Parallel环境该如何操作？先改为ACE试试。

<br/>

坏了，Tianshou默认所有代理的观察/动作空间都必须相等：
```python
def __init__(self, env: BaseWrapper):
        super().__init__()
        self.env = env
        # agent idx list
        self.agents = self.env.possible_agents
        self.agent_idx = {}
        for i, agent_id in enumerate(self.agents):
            self.agent_idx[agent_id] = i

        self.rewards = [0] * len(self.agents)

        # Get first observation space, assuming all agents have equal space
        self.observation_space: Any = self.env.observation_space(self.agents[0])

        # Get first action space, assuming all agents have equal space
        self.action_space: Any = self.env.action_space(self.agents[0])

        assert all(
            self.env.observation_space(agent) == self.observation_space for agent in self.agents
        ), (
            "Observation spaces for all agents must be identical. Perhaps "
            "SuperSuit's pad_observations wrapper can help (useage: "
            "`supersuit.aec_wrappers.pad_observations(env)`"
        )

        assert all(self.env.action_space(agent) == self.action_space for agent in self.agents), (
            "Action spaces for all agents must be identical. Perhaps "
            "SuperSuit's pad_action_space wrapper can help (useage: "
            "`supersuit.aec_wrappers.pad_action_space(env)`"
        )

        self.reset()
```
如果要仿照MSPM的思路，则必须放弃使用Tianshou框架了。