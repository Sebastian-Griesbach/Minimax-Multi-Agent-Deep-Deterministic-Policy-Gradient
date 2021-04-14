from numpy.core.defchararray import index
from torch.nn.modules.linear import Linear
from multiagent_wrapper import Multiagent_wrapper
from multiagent_critic import Multiagent_critic
from m3ddpg import M3DDPG
import gym
import torch
from torch import nn
import numpy as np

def main():
    env = Multiagent_debugEnv_wrapper()

    actor1 =  DebugActorNet(in_dim=env.observation_spaces[0].shape[0], out_dim=env.action_spaces[0].shape[0], min_value=min(env.action_spaces[0].low), max_value=max(env.action_spaces[0].high), index_value=1)
    actor2 =  DebugActorNet(in_dim=env.observation_spaces[1].shape[0], out_dim=env.action_spaces[1].shape[0], min_value=min(env.action_spaces[1].low), max_value=max(env.action_spaces[1].high), index_value=2)

    critic1 = DebugCriticNet(in_dim=env.state_space.shape[0]+env.action_space.shape[0], index_value=1)
    critic2 = DebugCriticNet(in_dim=env.state_space.shape[0]+env.action_space.shape[0], index_value=2)

    m3ddpg = M3DDPG(env= env, 
                actor_models = [actor1, actor2],
                critic_models = [critic1, critic2],
                actor_learning_rates = [0.01, 0.01],
                critic_learning_rates = [0.01, 0.01],
                device = "cpu",
                discounts = [0.99, 0.99],
                taus = [0.05, 0.05],
                noise_levels = [0.1, 0.1],
                critic_noise_levels = [0.1, 0.1],
                noise_clips = [0.1,0.1],
                epsilons = [0.5, 0.5],
                batch_size=8,
                burnin_steps=0,
                max_replay_buffer_size = 10000,
                update_target_nets_frequency = 2)

    #return env, m3ddpg
    m3ddpg.train(100)

class DebugActorNet(nn.Module):
    def __init__(self, in_dim, out_dim, min_value, max_value, index_value):
        super(DebugActorNet, self).__init__()

        self.index_value = index_value

        self.layers = nn.Sequential(
          nn.Linear(in_dim,out_dim),
        )

        self.register_buffer('min_value', torch.tensor(min_value, requires_grad=False, dtype=torch.float32))
        self.register_buffer('max_value', torch.tensor(max_value, requires_grad=False, dtype=torch.float32))
        
    def forward(self, x):
        out = torch.clip(self.layers(x), self.min_value, self.max_value)
        if(len(out.shape) > 1):
            out[:,0] = self.index_value
        else:
            out[0] = self.index_value
        return out

class DebugCriticNet(Multiagent_critic):
    def __init__(self, in_dim, index_value):
        super().__init__()
        self.layers = nn.Sequential(
          nn.Linear(in_dim,1),
        )
        self.index_value = index_value

    def forward(self, state, actions) -> torch.tensor:
        combined = torch.hstack([state,*actions])
        out = self.layers(combined)
        out = torch.clip(out, -0.5, +0.5)
        out += self.index_value
        return out

class DebugEnv(gym.Env):

    def __init__(self):
        super(DebugEnv, self).__init__()
        self.action_space = gym.spaces.Box(-2.0, 2.0, [4], np.float32)
        self.observation_space = gym.spaces.Box(-2.0, 2.0, [5], np.float32)
        self.obs = np.zeros(3)
        self.num_steps = 0
        self.max_num_steps = 1000.

    def step(self, action):
        self.num_steps += 1
        self.obs = np.hstack([action, self.num_steps/self.max_num_steps])
        reward = 1
        done =  self.max_num_steps < self.num_steps
        return self.obs, reward, done, None

    def reset(self):
        self.obs = np.hstack([1.,0.,2.,0., self.num_steps/self.max_num_steps])
        self.num_steps = 0
        return self.obs

    def render(self, mode='human'):
        pass

class Multiagent_debugEnv_wrapper(Multiagent_wrapper):
    def __init__(self):
        env = DebugEnv()
        state_space = env.observation_space
        num_agents = 2
        action_spaces = [gym.spaces.Box(-2.0, 2.0, [2], np.float32)]*2
        observation_spaces = [env.observation_space]*2
        
        super().__init__(env, state_space, num_agents, action_spaces, observation_spaces)

    def _build_joint_action(self, actions):
        return np.hstack(actions)

    def _build_observations(self, state):
        obs_1 = state
        obs_2 = np.hstack([state[2:4],state[0:2],state[4]])
        return [obs_1, obs_2]

    def _build_rewards(self, state, reward, info):
        return [1+self.num_steps/self.max_num_steps,2+self.num_steps/self.max_num_steps]

if __name__ == "__main__":
    main()