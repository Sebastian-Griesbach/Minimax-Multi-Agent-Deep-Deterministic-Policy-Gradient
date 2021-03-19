import gym
import numpy as np
from abc import ABC, abstractmethod

class Multiagent_wrapper(ABC, gym.Wrapper):
    def __init__(self, env, state_space, num_agents, action_spaces, observation_spaces):
        gym.Wrapper.__init__(env)

        self.state_space = state_space
        self.num_agents = num_agents
        self.action_spaces = action_spaces
        self.observation_spaces = observation_spaces

    def step(self, actions):
        joint_action = self._build_joint_action(actions)
        state, reward, done, info = self.env.step(joint_action)
        observations = self._build_observations(state, info)
        rewards = self._build_rewards(state, reward, info)
        return state, observations, rewards, done, info

    def reset(self):
        state = self.reset()
        observations = self._build_observations(state)
        return state, observations

    @abstractmethod
    def _build_joint_action(self, actions):
        pass

    @abstractmethod
    def _build_observations(self, state, info):
        pass

    @abstractmethod
    def _build_rewards(self, state, reward, info):
        pass