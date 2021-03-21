from ..multiagent_wrapper import Multiagent_wrapper
from ..multiagent_critic import Multiagent_critic
import laserhockey.hockey_env as h_env
import gym
import torch
import numpy as np

def main():
    

class Multiagent_laserhockey_wrapper(Multiagent_wrapper):
    def __init__(self):
        env = h_env.HockeyEnv()
        state_space = env.observation_space
        num_agents = 2
        action_spaces = [gym.spaces.Box(-1.0, 1.0, [4], np.float32)]*2
        observation_spaces = [env.observation_space]*2
        
        super().__init__(env, state_space, num_agents, action_spaces, observation_spaces)

    def _build_joint_action(self, actions):
        return np.hstack(actions)

    def _build_observations(self, state):
        return state

    def _build_rewards(self, state, reward, info):
        pure_reward_p1 = reward - info["reward_closeness_to_puck"]
        reward_p1 = pure_reward_p1
        reward_p2 = -pure_reward_p1
        return [reward_p1, reward_p2]

class Multiagent_laserhockey_critic(Multiagent_critic):
    def __init__(self, model):
        super().__init__(model)

    def _build_input(self, state, *actions) -> torch.tensor:
        return torch.hstack(state, *actions)

if __name__ == "__main__":
    main()