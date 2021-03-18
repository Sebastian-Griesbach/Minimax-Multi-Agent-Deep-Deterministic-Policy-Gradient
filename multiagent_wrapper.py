import gym
import numpy as np
from abc import ABC, abstractmethod

class Multiagent_wrapper(ABC, gym.Wrapper):
    def __init__(self, env, num_agents, action_dims, observation_dims):
        gym.Wrapper.__init__(env)

        self.num_agents = num_agents
        self.action_spaces = action_dims
        self.observation_space = observation_dims

    def step(self, actions):
        joint_action = self._build_joint_action(actions)
        observation, reward, done, info = self.env.step(joint_action)
        observations = self._split_observation(observation)
        rewards = self._calculate_rewards(observation, reward, info)
        return observations, rewards, done, info

    def reset(self):
        observation = self.reset()
        return self._build_observations(observation)

    def _build_joint_action(self, actions):
        return np.hstack(actions)

    @abstractmethod
    def _build_observations(self, observation, info):
        pass

    @abstractmethod
    def _build_rewards(self, observation, reward, info):
        pass