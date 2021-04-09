import gym
import numpy as np
from abc import ABC, abstractmethod
from typing import List

class Multiagent_wrapper(ABC, gym.Wrapper):
    def __init__(self, env, state_space, num_agents, action_spaces, observation_spaces):
        """initalizes Multiagent wrapper

        Args:
            env (gym.env): underlying gym environment
            state_space ([type]): Observation space of the underlying environment
            num_agents ([type]): number of agents participating in this environment
            action_spaces ([type]): List of action spaces of agents. According to order of agents.
            observation_spaces ([type]): List of observation spaces of agents. Accoding to order of agents.
        """
        gym.Wrapper.__init__(self, env)

        self.state_space = state_space
        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.observation_spaces = observation_spaces

    def step(self, actions):
        """Performe one timestep

        Args:
            actions ([type]): List of actions to performe. According to order of agents.

        Returns:
            [type]: Tuple containing (state, observations, rewards, done, info) where observations and rewards are list according to the order of agents.
        """
        joint_action = self._build_joint_action(actions)
        state, reward, done, info = self.env.step(joint_action)
        observations = self._build_observations(state)
        rewards = self._build_rewards(state, reward, info)
        return state, observations, rewards, done, info

    def reset(self):
        """Reset the environment

        Returns:
            [type]: Tuple containing (state. observations) where observations is a list of observations according to the order of agents.
        """
        state = self.env.reset()
        observations = self._build_observations(state)
        return state, observations

    @abstractmethod
    def _build_joint_action(self, actions) -> np.array:
        """This function needs to merge the actions together such that the underlying environment can process them.

        Args:
            actions ([type]): List of actions according to the order of agents.

        Returns:
            np.array: merged actions such that the underlying environment can process them.
        """
        pass

    @abstractmethod
    def _build_observations(self, state) -> List[np.array]:
        """This function needs to split up the state into the observations of the single agents.

        Args:
            state ([type]): total state of the environment

        Returns:
            List[numpy.array]: List of observations according to order of agents.
        """
        pass

    @abstractmethod
    def _build_rewards(self, state, reward, info) -> list:
        """This function needs to determin the rewards for all agents

        Args:
            state ([type]): Total state of the underlying environment
            reward ([type]): reward of the underlying environment
            info ([type]): info of the underlying environment

        Returns:
            list: List of rewards according to order of agents
        """
        pass