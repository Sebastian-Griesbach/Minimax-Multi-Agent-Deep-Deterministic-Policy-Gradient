from abc import ABC, abstractmethod
from typing import Dict
import torch
import gym
import copy
from tqdm import tqdm
import numpy as np
import copy

class Actor_Critic(ABC):
    def __init__(self,
                environment: gym.env,
                actor: torch.nn.module,
                critic: torch.nn.module,
                replay_buffer,
                device: str,
                hyper_parameter_dict: Dict,
                disable_tqdm: bool = False,
                dtype: torch.dtype = torch.float32) -> None:

        self.environment = environment
        self.dtype = dtype
        self.actor = actor.train()
        self.critic = critic.train()
        self.target_actor = copy.deepcopy(self.actor).eval()
        self.target_critic = copy.deepcopy(self.target_critic).eval()
        self.replay_buffer = replay_buffer
        self.device = device
        self.hyper_parameter_dict = hyper_parameter_dict
        self.disable_tqdm = disable_tqdm
        self.dtype = dtype

        self.env_done = True
        self.episode_rewards = []
        self.episode_reward = 0
        self.total_train_steps = 0

    def train(self, num_train_steps):

        if self.env_done:
            observation = self.environment.reset()
            self.env_done = False

        for update_step in tqdm(range(num_train_steps)):

            if self.env_done:
                observation = self.environment.reset()
                self.episode_rewards.append(self.episode_reward)

            with torch.no_grad():
                action = self.select_action(self.numpy_to_torch(observation))

            new_observation, reward, self.env_done, info = self.environment.step(action)
            action = self.torch_to_numpy(action)
            self.replay_buffer.add_transition(observation, action, reward, new_observation, self.env_done)

            self.episode_reward += reward

            observation = new_observation

            states, actions, rewards, next_states, terminals = self.replay_buffer.sample(self.batch_size)
    
            self.update_critice(self.target_actor, self.critic, self.target_critic, states, actions, rewards, next_states, terminals)

            self.update_actor(self.actor, self.critic, states)

            self.update_target_net()

        return self.episode_rewards

    def numpy_to_torch(self, numpy_array):
        return torch.tensor(numpy_array, dtype=self.dtype).to(self.device)

    def torch_to_numpy(self, torch_tensor):
        return torch_tensor.detach().cpu().numpy()

    @abstractmethod
    def update_critice(self, target_actor: torch.nn.module,
                        critic: torch.nn.module,
                        target_critic: torch.nn.module,
                        states: torch.tensor, 
                        actions: torch.tensor,
                        rewards: torch.tensor,
                        next_states: torch.tensor,
                        terminals: torch.tensor,
                        *args,
                        **kwargs):
        ...

    @abstractmethod
    def update_actor(self, 
                        actor: torch.nn.module, 
                        critic: torch.nn.module, 
                        states: torch.tensor, 
                        *args, 
                        **kwargs):
        ...

    @abstractmethod
    def select_action(self, 
                        states: torch.tensor, 
                        *args, 
                        **kwargs):
        ...

    @abstractmethod
    def update_target_net(self, *args, **kwargs):
        ...

    @abstractmethod
    def check_hyper_parameter_dict(self, *args, **kwargs):
        ...


class DDPG(Actor_Critic):
    def __init__(self, 
                environment, 
                actor_model,
                critic_model,
                actor_optimizer,
                critic_optimizer,
                replay_buffer,
                noise_generator,
                device,
                discount=0.99,
                tau=0.001,
                noise_level=0.2,
                noise_clip=1, 
                noise_decay=1,
                batch_size=64,
                max_episode_length=500,
                dtype=torch.float32
                ):

        super(DDPG, self).__init__( replay_buffer=replay_buffer,
                                    environment=environment,
                                    max_episode_length=max_episode_length,
                                    device=device,
                                    batch_size=batch_size,
                                    dtype=dtype)

        self.actor = actor_model.train()
        self.target_actor = copy.deepcopy(self.actor).eval()
        self.critic = critic_model.train()
        self.target_critic = copy.deepcopy(self.critic).eval()

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.device = device

        self.noise = noise_generator
        self.noise_level = noise_level
        self.noise_clip = noise_clip
        self.noise_decay = noise_decay

        self.action_high = torch.tensor(self.environment.action_space.high, dtype = self.dtype, requires_grad=False, device=self.device)
        self.action_low = torch.tensor(self.environment.action_space.low, dtype = self.dtype, requires_grad=False, device=self.device)

        self.MSE = torch.nn.MSELoss()
        
        self.discount = discount
        self.tau = tau

    def update_critice(self, states, actions, rewards, next_states, terminals):
        q_values = self.critic(torch.hstack([states, actions]))
        with torch.no_grad(): 
            next_actions = self.select_action(states)
            targets = rewards + (1-terminals) * self.discount * self.target_critic(torch.hstack([next_states, next_actions]))

        self.critic_optimizer.zero_grad()
        critic_loss = self.MSE(q_values, targets)
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, states):
        self.actor_optimizer.zero_grad()
        actor_loss =  -self.critic(torch.hstack([states, self.actor(states)])).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_target_net(self):
        for netpair in [(self.target_actor, self.actor),
                        (self.target_critic, self.critic)]:
            for target_params, true_params in zip(netpair[0].parameters(), netpair[1].parameters()):
                target_params.data.copy_(target_params.data * (1.0 - self.tau) + true_params.data * self.tau)

    def select_action(self, observation):
        noise = self.noise.generate(self.noise_level, self.noise_clip).to(self.device)
        with torch.no_grad():
            action = self.actor(observation.to(self.device))
        return torch.max(torch.min(action + noise,self.action_high),self.action_low)

    def pre_episode_routine(self):
        pass