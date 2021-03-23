import torch
import numpy as np
from cpprb import ReplayBuffer
from operator import itemgetter

class Multiagent_replay_buffer():

    STATE_KEY = "state"
    NEXT_STATE_KEY = "next_state"
    OBS_PREFIX = "obs"
    ACT_PREFIX = "act"
    REW_PREFIX = "rew"
    NEXT_OBS_PREFIX = "next_obs"
    DONE_KEY = "done"

    def __init__(self, state_shape, observation_shapes, action_shapes, num_agents, return_device, max_size=100000, dtype=torch.float32):

        self.state_shape = state_shape
        self.observation_shapes = observation_shapes
        self.action_shapes = action_shapes
        self.num_agents = num_agents

        self.obs_keys, self.act_keys, self.rew_keys, self.next_obs_keys = [], [], [], []

        env_dict = {}
        env_dict[self.STATE_KEY] = {"shape" : self.state_shape}
        env_dict[self.NEXT_STATE_KEY] = {"shape" : self.state_shape}

        for i in range(num_agents):
            self.obs_keys.append(f"{self.OBS_PREFIX}{i}")
            self.act_keys.append(f"{self.ACT_PREFIX}{i}")
            self.rew_keys.append(f"{self.REW_PREFIX}{i}")
            self.next_obs_keys.append(f"{self.NEXT_OBS_PREFIX}{i}")

            env_dict[self.obs_keys[i]] = {"shape" : self.observation_shapes[i]}
            env_dict[self.act_keys[i]] = {"shape" : self.action_shapes[i]}
            env_dict[self.rew_keys[i]] = {}
            env_dict[self.next_obs_keys[i]] = {"shape" : self.observation_shapes[i]}

        env_dict[self.DONE_KEY] = {}

        self.concatinated_keys =  [self.STATE_KEY] + [self.NEXT_STATE_KEY] + self.obs_keys + self.act_keys + self.rew_keys + self.next_obs_keys + [self.DONE_KEY]

        self.replay_buffer = ReplayBuffer(max_size, env_dict)

        self.return_device = return_device
        self.dtype = dtype

    def add_transition(self, state, next_state, observations, actions, rewards, next_observations, done):

        concatinated_data = [state] + [next_state] + observations + actions + rewards + next_observations + [done]
        param_dict = dict(zip(self.concatinated_keys, concatinated_data))

        self.replay_buffer.add(**param_dict)

    def sample(self, batch_size):
        sample = self.replay_buffer.sample(batch_size)

        states = self._numpy_to_tensor(sample[self.STATE_KEY])
        next_states = self._numpy_to_tensor(sample[self.NEXT_STATE_KEY])
        observations = self._get_subset_from_sample(self.obs_keys, sample)
        actions = self._get_subset_from_sample(self.act_keys, sample)
        rewards = self._get_subset_from_sample(self.rew_keys, sample)
        next_observations = self._get_subset_from_sample(self.next_obs_keys, sample)
        dones = self._numpy_to_tensor(sample[self.DONE_KEY])

        return states, next_states, observations, actions, rewards, next_observations, dones

    def _get_subset_from_sample(self, keys, sample):
        return list(map(self._numpy_to_tensor, list(itemgetter(*keys)(sample))))

    def _numpy_to_tensor(self, np_array):
        return torch.tensor(np_array, dtype=self.dtype).to(self.return_device)

    