from typing import Dict
from numpy import random
from numpy.lib.function_base import append, select
import torch
import gym
import copy
from torch.serialization import save
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import os
from multiagent_wrapper import Multiagent_wrapper
from replay_buffer import Multiagent_replay_buffer

class M3DDPG():
    def __init__(self, 
                env: Multiagent_wrapper, 
                actor_models,
                critic_models,
                actor_learning_rates,
                critic_learning_rates,
                device,
                discounts,
                taus,
                noise_levels,
                noise_clips,
                epsilons,
                batch_size=64,
                max_episode_length=500,
                max_replay_buffer_size = 100000,
                update_actor_frequency = 2):

        """
        save_path_models="./models/",
        save_path_data="./data/"
        """

        self.env = env
        self.device = device

        self.dtype = torch.float32

        self.discounts = discounts
        self.taus = taus
        self.noise_levels = noise_levels
        self.noise_clips = noise_clips
        self.epsilons = epsilons
        self.batch_size = batch_size
        self.max_episode_length = max_episode_length
        self.update_actor_frequency = update_actor_frequency

        self.state_shape = env.state_space.shape
        self.action_shapes = [action_space.shape for action_space in env.action_spaces]
        self.observation_shapes = [observation_shape.shape for observation_shape in env.observation_shapes]

        self.action_highs = [action_space.high for action_space in env.action_spaces]
        self.action_lows = [action_space.low for action_space in env.action_spaces]
        
        self.num_agents = env.num_agents

        self.actors, self.critics, self.target_actors, self.target_critics, self.actor_optimizers, self.critic_optimizers  = [], [], [], [], [], []

        for i in range(len(actor_models)):
            self.actors.append(actor_models[i].train().to(self.device))
            self.critics.append(critic_models[i].train().to(self.device))

            self.target_actors.append(copy.copy.deepcopy(actor_models[i]).eval().to(self.device))
            self.target_critics.append(copy.copy.deepcopy(critic_models[i]).eval().to(self.device))

            self.actor_optimizers.append(optim.Adam(self.actors[i].parameters(), lr = actor_learning_rates[i]))
            self.critic_optimizers.append(optim.Adam(self.critics[i].parameters(), lr = critic_learning_rates[i]))

        self.replay_buffer = Multiagent_replay_buffer(self.state_shape,
                                                        self.observation_shapes, 
                                                        self.action_shapes, 
                                                        self.num_agents, 
                                                        self.device, 
                                                        max_size=max_replay_buffer_size, 
                                                        dtype=self.dtype)


        #self.save_path_models = save_path_models
        #self.save_path_data = save_path_data

        self.loss = torch.nn.MSELoss()


        self.total_iterations = 0

        self.env_done = True
        self.rewards_histroy = np.zeros(1,self.env.num_agents)
        self.episode_rewards = np.zeros(self.env.num_agents)
        self.total_train_steps = 0

    def train(self, num_train_steps):

        if self.env_done:
            state, observations = self.env.reset()
            self.env_done = False

        for train_step in tqdm(range(num_train_steps)):

            if self.env_done:
                state, observations = self.env.reset()
                self.rewards_histroy.append(self.episode_rewards, axis=0)

            actions = []
            for actor_id in range(self.env.num_agents):
                with torch.no_grad():
                    actions.append(self.select_action(actor_id, observations[actor_id]))

            next_state, new_observations, rewards, self.env_done, _ = self.env.step(actions)

            self.replay_buffer.add_transition(state, next_state, observations, actions, rewards, new_observations, self.env_done)

            self.episode_rewards = np.sum([self.episode_rewards,rewards], axis=1)

            observations = new_observations
            state = next_state

            states_batch, next_states_batch, observations_batch, actions_batch, rewards_batch, next_observations_batch, done_batch = self.replay_buffer.sample(self.batch_size)
    
            self.update_critice(states_batch, next_states_batch, observations_batch, actions_batch, rewards_batch, next_observations_batch, done_batch)

            if self.total_train_steps % self.update_actor_frequency == 0:
                self.update_actor(states_batch, observations_batch, actions_batch)
                self.update_target_nets()

        return self.episode_rewards


        left_episode_rewards = []
        right_episode_rewards = []

        for _ in tqdm(range(num_iterations)):

            observation = self.env.reset()
            done = False
            episode_step = 0
            left_episode_reward = 0
            right_episode_reward = 0
            self.total_iterations += 1

            while (not done) and (episode_step <= self.max_episode_length):
                episode_step += 1

                actions = self.select_action(observation, self.epsilon)

                new_observation, rewards, done, _ = self.env.step(actions)
                self.replay_buffer.add_transition(observation, actions, rewards, new_observation, done)
                left_episode_reward += rewards[0]
                right_episode_reward += rewards[1]
                observation = new_observation

                #load from replay buffer
                states, actions, rewards, next_states, terminals = self.replay_buffer.sample(self.batch_size)
                
                #critic update
                q_values_1, q_values_2 = self.critics(torch.hstack([states, actions]))
                with torch.no_grad(): 
                    next_actions = self.add_noise_to_action(torch.hstack(self.target_actors(next_states)), noise_level = self.noise_level)
                    #next_actions = torch.hstack(self.target_actors(next_states))
                    next_q_value_1, next_q_value_2 = self.target_critics(torch.hstack([next_states, next_actions]))
                    targets_1 = (rewards[:,0].reshape(-1, 1) + (1-terminals) * self.discount * next_q_value_1).detach()
                    targets_2 = (rewards[:,1].reshape(-1, 1) + (1-terminals) * self.discount * next_q_value_2).detach()

                self.critic_1_optimizer.zero_grad()
                critic_1_loss = self.MSE(q_values_1, targets_1)
                critic_1_loss.backward()
                self.critic_1_optimizer.step()

                self.critic_2_optimizer.zero_grad()
                critic_2_loss = self.MSE(q_values_2, targets_2)
                critic_2_loss.backward()
                self.critic_2_optimizer.step()

                #actor update
                if episode_step % self.update_actor_frequency == 0:
                    actions_1, actions_2 = self.actors(states)
                    replay_actions_1, replay_actions_2 = (actions[:,:self.single_action_size], actions[:,self.single_action_size:])

                    self.actor_1_optimizer.zero_grad()
                    actor_1_loss = -self.critics.network_1(torch.hstack([states, torch.hstack([actions_1, replay_actions_2])])).mean()
                    actor_1_loss.backward()
                    self.actor_1_optimizer.step()

                    self.actor_2_optimizer.zero_grad()
                    actor_2_loss = -self.critics.network_2(torch.hstack([states, torch.hstack([replay_actions_1, actions_2])])).mean()
                    actor_2_loss.backward()
                    self.actor_2_optimizer.step()

                    #update target model
                    self.update_target_net(self.target_actors, self.actors, self.tau)
                    self.update_target_net(self.target_critics, self.critics, self.tau)

            left_episode_rewards.append(left_episode_reward)
            right_episode_rewards.append(right_episode_reward)

        return left_episode_rewards, right_episode_rewards

    def update_critice(self, states_batch, next_states_batch, observations_batch, actions_batch, rewards_batch, next_observations_batch, done_batch):
        with torch.no_grad(): 
            #next_actions = map(self.actors, next_observations_batch)

            #next_actions = self.add_noise_to_action(torch.hstack(self.target_actors(next_states)), noise_level = self.noise_level)
            #next_actions = torch.hstack(self.target_actors(next_states))
            next_actions_batch = _execute_function_param_pairs(self.target_actors, next_observations_batch, False)
            next_q_values_batch = _execute_function_param_pairs(self.target_critics, zip(next_states_batch, next_actions_batch), True)
            #TODO continue here
            next_q_value_1, next_q_value_2 = self.target_critics(torch.hstack([next_states, next_actions]))
            targets_1 = (rewards[:,0].reshape(-1, 1) + (1-terminals) * self.discount * next_q_value_1).detach()
            targets_2 = (rewards[:,1].reshape(-1, 1) + (1-terminals) * self.discount * next_q_value_2).detach()

        #q_values_1, q_values_2 = self.critics(torch.hstack([states, actions]))
        q_values = _execute_function_param_pairs(self.critics, zip(states_batch, actions_batch), True)
        q_values = map(self.critics, states_batch, actions_batch)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss = self.MSE(q_values_1, targets_1)
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss = self.MSE(q_values_2, targets_2)
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

    def update_target_net(self, target_net, true_net, tau):
        for target_params, true_params in zip(target_net.parameters(), true_net.parameters()):
            target_params.data.copy_(target_params.data * (1.0 - tau) + true_params.data * tau)

    def add_noise_to_action(self, action, noise_level):
        noise = (torch.randn_like(action) * noise_level).clamp(-self.noise_clip, self.noise_clip)
        #noise = self.noise.generate(self.noise_level, self.noise_clip).to(self.device)
        return torch.clip(action + noise,self.action_low, self.action_high)

    def select_action(self, observation, epsilon):
        random_actions = self.env.action_space.sample().reshape(-1, self.single_action_size)
        with torch.no_grad():
            actions = self.add_noise_to_action(torch.hstack(self.actors(torch.FloatTensor(observation).to(self.device))), noise_level=self.noise_level)
        greedy_actions = actions.cpu().numpy().reshape(-1, self.single_action_size)

        rand = np.random.uniform(size=self.num_agents).reshape(-1,1)

        take_random = rand <= epsilon
        take_greedy = np.invert(take_random)

        actions = take_random * random_actions + take_greedy * greedy_actions
        return actions.flatten()

    def get_policy(self):
        policy = copy.deepcopy(self.actors).eval().cpu()
        def act(state):
            with torch.no_grad():
                tensor_state = torch.tensor(state, dtype=self.dtype, requires_grad=False)
                action = torch.hstack(policy(tensor_state))
            return action.numpy()

        return act

    def save_status(self,Prefix="MADDPG"):
        actor_file_name = f'{Prefix}_actor_{self.total_iterations}its.pt'
        self.save_model(self.actors, self.save_path_models, actor_file_name)
        critic_file_name = f'{Prefix}_critic_{self.total_iterations}its.pt'
        self.save_model(self.critics, self.save_path_models, critic_file_name)

        replay_buffer_file_name = f'{Prefix}_replay_buffer_{self.total_iterations}its.pt'
        save_path = os.path.join(self.save_path_data,replay_buffer_file_name)
        self.replay_buffer.save(save_path)

    def load_status(self, actor_file_name, critice_file_name, replay_buffer_file_name = None):
        self.actors.load_state_dict(self.load_model(self.save_path_models, actor_file_name))
        self.critics.load_state_dict(self.load_model(self.save_path_models, critice_file_name))
        self.target_actors = copy.deepcopy(self.actors).eval().to(self.device)
        self.target_critics = copy.deepcopy(self.critics).eval().to(self.device)

        if(replay_buffer_file_name != None):
            load_path = os.path.join(self.save_path_data,replay_buffer_file_name)
            self.replay_buffer.load(load_path)


    def save_model(self, model, path, filename):
        save_path = os.path.join(path,filename)
        torch.save(model.state_dict(), save_path)

    def load_model(self, path, filename):
        load_path = os.path.join(path,filename)
        return torch.load(load_path)

def _execute_function_param_pairs(function_list, params, multiparam):
    execute = (lambda func, args: func(*args)) if multiparam else (lambda func, args: func(args))
    return list(map(execute, function_list, params))