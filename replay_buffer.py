#Uses https://pypi.org/project/cpprb/

from typing import Dict, List, Tuple
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

    def __init__(self, state_shape: Tuple[int, ...], observation_shapes: List[Tuple[int, ...]], action_shapes: List[Tuple[int, ...]], num_agents: int, return_device: str, max_size: int = 100000, dtype: torch.dtype = torch.float32) -> None:
        """initialising replay buffer

        Args:
            state_shape (Tuple[int, ...]): shape of environment state
            observation_shapes (List[Tuple[int, ...]]): List of shapes of observations according to order of agent
            action_shapes (List[Tuple[int, ...]]): List of shapes of actions according to order of agents
            num_agents (int): Number of agents
            return_device (str): cuda device on which batches will be returned on
            max_size (int, optional): Maximum number of timesteps the replay buffer will hold. If reached the oldest entries will be overwritten. Defaults to 100000.
            dtype (torch.dtype, optional): Datatype to return batches in. Defaults to torch.float32.
        """

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

    def add_transition(self, state: np.array, next_state: np.array, observations: List[np.array], actions: List[np.array], rewards: List[float], next_observations: List[np.array], done: bool) -> None:
        """Adds a transition of the environment to the replay buffer

        Args:
            state (np.array): old state of the environment
            next_state (np.array): new state of the environment
            observations (List[np.array]): List of old observations of agents, according to order of agents.
            actions (List[np.array]): List of actions taken be the agents, accodring to order of agents.
            rewards (List[float]): List of rewards received according to order of agents.
            next_observations (List[np.array]): List of new observations of the agents after the actions have been perfomred. Accodring to order of agents.
            done (bool): signial whether the episode has terminated
        """

        concatinated_data = [state] + [next_state] + observations + actions + rewards + next_observations + [done]
        param_dict = dict(zip(self.concatinated_keys, concatinated_data))

        self.replay_buffer.add(**param_dict)

    def sample(self, batch_size: int) -> Tuple[torch.tensor, torch.tensor, List[torch.tensor], List[torch.tensor], List[torch.tensor], List[torch.tensor], torch.tensor]:
        """Sample a batch of the replay buffer

        Args:
            batch_size (int): Number of how many transitions the batch will contain

        Returns:
            Tuple[torch.tensor, torch.tensor, List[torch.tensor], List[torch.tensor], List[torch.tensor], List[torch.tensor], torch.tensor]: sampeled batch of the replay buffer retuned as a tuple with following entries (states, next_states, observations, actions, rewards, next_observations, dones)
        """
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
        """Helper function to extract specific entries from a dictionary and convert them into a torch.tensor

        Args:
            keys: List of Keys to extract
            sample: Raw dictionary sample

        Returns:
            List[torch.tensor]: List of torch.tensor containing all entries of according to the keys. Order of the keys determins order of the tensors in the List.
        """
        return list(map(self._numpy_to_tensor, list(itemgetter(*keys)(sample))))

    def _numpy_to_tensor(self, np_array):
        """Converts numpy arrays to torch tensors setting dtype and device.

        Args:
            np_array: numpy array to convert

        Returns:
            torch.tensor: converted numpy array
        """
        return torch.tensor(np_array, dtype=self.dtype, requires_grad=False).to(self.return_device)

    