from torch.nn.modules.linear import Linear
from multiagent_wrapper import Multiagent_wrapper
from multiagent_critic import Multiagent_critic
from m3ddpg import M3DDPG
import laserhockey.hockey_env as h_env
import gym
import torch
from torch import nn
import numpy as np


def main():
    env = Multiagent_laserhockey_wrapper()

    actor1 =  HockeyActorNet(in_dim=env.observation_spaces[0].shape[0], out_dim=env.action_spaces[0].shape[0], min_value=min(env.action_spaces[0].low), max_value=max(env.action_spaces[0].high))
    actor2 =  HockeyActorNet(in_dim=env.observation_spaces[1].shape[0], out_dim=env.action_spaces[1].shape[0], min_value=min(env.action_spaces[1].low), max_value=max(env.action_spaces[1].high))

    critic1 = HockeyCriticNet(in_dim=env.state_space.shape[0]+env.action_space.shape[0])
    critic2 = HockeyCriticNet(in_dim=env.state_space.shape[0]+env.action_space.shape[0])

    m3ddpg = M3DDPG(env= env, 
                actor_models = [actor1, actor2],
                critic_models = [critic1, critic2],
                actor_learning_rates = [0.0001, 0.0001],
                critic_learning_rates = [0.0001, 0.0001],
                device = "cuda",
                discounts = [0.99, 0.99],
                taus = [0.05, 0.05],
                noise_levels = [0.2, 0.2],
                critic_noise_levels = [0.02, 0.02],
                noise_clips = [1.,1.],
                epsilons = [0.2, 0.2],
                batch_size=512,
                burnin_steps=100000,
                max_replay_buffer_size = 200000,
                update_target_nets_fequency = 2)

    return m3ddpg
    #m3ddpg.train(1000)

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
        return [state, self.env.obs_agent_two()]

    def _build_rewards(self, state, reward, info):
        pure_reward_p1 = reward - info["reward_closeness_to_puck"]
        reward_p1 = max(0., pure_reward_p1)
        reward_p2 = max(0., -pure_reward_p1)
        return [reward_p1, reward_p2]

class HockeyActorNet(nn.Module):
    def __init__(self, in_dim, out_dim, min_value, max_value):
        super(HockeyActorNet, self).__init__()

        self.layers = nn.Sequential(
          nn.Linear(in_dim,128),
          nn.ReLU(),
          nn.Linear(128, out_dim)
        )

        self.register_buffer('min_value', torch.tensor(min_value, requires_grad=False, dtype=torch.float32))
        self.register_buffer('max_value', torch.tensor(max_value, requires_grad=False, dtype=torch.float32))
        
    def forward(self, x):
        return torch.clip(self.layers(x), self.min_value, self.max_value)

class HockeyCriticNet(Multiagent_critic):
    def __init__(self, in_dim):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(in_dim,256),
          nn.ReLU(),
          nn.Linear(256,1)
        )

    def forward(self, state, *actions) -> torch.tensor:
        combined = torch.hstack([state,*actions])
        return self.layers(combined)

if __name__ == "__main__":
    main()