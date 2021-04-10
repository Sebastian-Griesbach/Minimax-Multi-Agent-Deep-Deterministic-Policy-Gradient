import gym
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Union

class Multiagent_wrapper(ABC, gym.Wrapper):
    def __init__(self, env: gym.Env, state_space: gym.Space, num_agents: int, action_spaces: List[gym.Space], observation_spaces: List[gym.Space]) -> None:
        """initalizes Multiagent wrapper

        Args:
            env (gym.Env): underlying gym environment
            state_space (gym.Space): Observation space of the underlying environment
            num_agents (int): number of agents participating in this environment
            action_spaces (List[gym.Space]): List of action spaces of agents. According to order of agents.
            observation_spaces (List[gym.Space]): List of observation spaces of agents. Accoding to order of agents.
        """
        gym.Wrapper.__init__(self, env)

        self.state_space = state_space
        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.observation_spaces = observation_spaces

    def step(self, actions: List[np.array]) -> Tuple[np.array, List[np.array], List[float], List[bool], Union[None, Dict]]:
        """Performe one timestep

        Args:
            actions (List[numpy.array]): List of actions to performe. According to order of agents.

        Returns:
            Tuple[numpy.array, List[numpy.array], List[float], List[bool], Union[None, Dict]]: Tuple containing (state, observations, rewards, done, info) where observations and rewards are list according to the order of agents.
        """
        joint_action = self._build_joint_action(actions)
        state, reward, done, info = self.env.step(joint_action)
        observations = self._build_observations(state)
        rewards = self._build_rewards(state, reward, info)
        return state, observations, rewards, done, info

    def reset(self) -> Tuple[np.array, List[np.array]]:
        """Reset the environment

        Returns:
            Tuple[np.array, List[np.array]]: Tuple containing (state. observations) where observations is a list of observations according to the order of agents.
        """
        state = self.env.reset()
        observations = self._build_observations(state)
        return state, observations

    @abstractmethod
    def _build_joint_action(self, actions: List[np.array]) -> np.array:
        """This function needs to merge the actions together such that the underlying environment can process them.

        Args:
            actions (List[numpy.array]): List of actions according to the order of agents.

        Returns:
            numpy.array: merged actions such that the underlying environment can process them.
        """
        pass

    @abstractmethod
    def _build_observations(self, state: np.array) -> List[np.array]:
        """This function needs to split up the state into the observations of the single agents.

        Args:
            state (numpy.array): total state of the environment

        Returns:
            List[numpy.array]: List of observations according to order of agents.
        """
        pass

    @abstractmethod
    def _build_rewards(self, state: np.array, reward: float, info: Union[None, Dict]) -> List[float]:
        """This function needs to determin the rewards for all agents

        Args:
            state (numpy.array): Total state of the underlying environment
            reward (float): reward of the underlying environment
            info (Union[None, Dict])): info of the underlying environment

        Returns:
            List[float]: List of rewards according to order of agents
        """
        pass