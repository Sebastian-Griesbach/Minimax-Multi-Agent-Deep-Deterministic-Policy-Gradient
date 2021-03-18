import torch
import numpy as np
from cpprb import ReplayBuffer

class Multiagent_Replay_buffer():
    def __init__(self, observation_dims, action_dims, num_agents, return_device, max_size=100000, dtype=torch.float32):

        self.observation_dims = observation_dims
        self.action_dims = action_dims
        self.num_agents = num_agents

        observation_dims_cumsum = np.cumsum(observation_dims)
        action_dims_cumsum = np.cumsum(action_dims)
        
        self.joint_observation_dim = observation_dims_cumsum[-1]
        self.joint_action_dim = action_dims_cumsum[-1]

        #TODO create dict entry for each agent

        self.replay_buffer = ReplayBuffer(max_size,
            env_dict ={"obs": {"shape": self.joint_observation_dim},
            "act": {"shape": self.joint_action_dim},
            "rew": {"shape": self.num_agents},
            "next_obs": {"shape": self.joint_observation_dim},
            "done": {}})

        self.observation_slices, self.actions_slices,  = [], []
        for i in range(num_agents):
            self.observation_slices.append(slice(observation_dims_cumsum[i],observation_dims_cumsum[i+1]))
            self.actions_slices.append(slice(action_dims_cumsum[i],action_dims_cumsum[i+1]))

        self.return_device = return_device
        self.dtype = dtype

    def add_transition(self, observations, actions, rewards, next_observations, done):
        joint_observation = np.hstack(observations)
        joint_actions = np.hstack(actions)
        rewards = np.array(rewards)
        joint_next_observations = np.hstack(next_observations)

        self.replay_buffer.add(obs=joint_observation, act=joint_actions, rew=rewards, next_obs=joint_next_observations, done=done)

    def sample(self, batch_size):
        sample = self.replay_buffer.sample(batch_size)

        joint_observations = torch.tensor(sample["obs"], dtype=self.dtype).to(self.return_device)
        joint_actions = torch.tensor(sample["act"], dtype=self.dtype).to(self.return_device)
        joint_next_observations = torch.tensor(sample["next_obs"], dtype=self.dtype).to(self.return_device)
        rewards = torch.tensor(sample["rew"], dtype=self.dtype).to(self.return_device)
        dones = torch.tensor(sample["done"], dtype=self.dtype).to(self.return_device)

        observations =  [joint_observations[:,slice_] for slice_ in self.observation_slices]
        actions = [joint_actions[:,slice_] for slice_ in self.observation_slices]
        next_observations = [joint_next_observations[:,slice_] for slice_ in self.observation_slices]

        return observations, actions, rewards, next_observations, dones

    