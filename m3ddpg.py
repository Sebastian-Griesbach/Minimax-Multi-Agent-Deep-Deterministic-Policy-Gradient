from typing import Dict
from numpy import random
from numpy.lib.function_base import append, select
import torch
import gym
import copy
from torch.optim import optimizer
from torch.serialization import save
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import os
from collections import deque
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
                critic_noise_levels,
                epsilons,
                max_replay_buffer_size,
                burnin_steps = 10000,
                update_target_nets_frequency = 2,
                batch_size=64):
        """
        Args:
            env (Multiagent_wrapper): gym Multiagent_wrapper of training environment
            actor_models ([type]): List of actor models according to order of agents
            critic_models ([type]): List of critics models according to order of agents
            actor_learning_rates ([type]): List of actor learning rates according to order of agents
            critic_learning_rates ([type]): List of critic learning rates according to order of agents
            device ([type]): cuda device to use for model training
            discounts ([type]): List of discount factors for future rewards according to order of agents
            taus ([type]): List of tau factors used to update the target net, both for critics and actors, according to order of agents
            noise_levels ([type]): List of factors to multiply standart normal noise with for which is used as exploration noise, according to order of agents
            noise_clips ([type]): List of values the exploration noise will be clipped to. Only one value per agent, according to order of agents.
            critic_noise_levels ([type]): List of factors multiplied with standart normal noise. This noise is added to the target_values of the critics during training. According to order of agents
            epsilons ([type]): List of probability values to take an entirly random action (sampeled from the actionspace) at each time step, for more exploration additionally to exploration noise. According to order of agents
            max_replay_buffer_size ([type]): Maximal number of time steps the Replaybuffer will hold. When this number is reached the oldest entries will be overwritten.
            burnin_steps (int, optional): number of timesteps taken befor the training routine starts. During this time random actions sampeled from the actionspace will be performed. Defaults to 10000.
            update_target_nets_frequency (int, optional): Frequency of timesteps at which the targents as well as the original actor net ist updated. Defaults to 2.
            batch_size (int, optional): Number of timesteps that are take into account at each update steps. Defaults to 64.
        """

        self.env = env
        self.device = device

        self.dtype = torch.float32

        self.discounts = discounts
        self.taus = taus
        self.noise_levels = noise_levels
        self.noise_clips = noise_clips
        self.critic_noise_levels = critic_noise_levels
        self.epsilons = epsilons
        self.batch_size = batch_size
        self.update_target_nets_frequency = update_target_nets_frequency
        self.burnin_steps = burnin_steps

        self.state_shape = env.state_space.shape
        self.action_shapes = [action_space.shape for action_space in env.action_spaces]
        self.observation_shapes = [observation_space.shape for observation_space in env.observation_spaces]

        self.action_highs = [self.numpy_to_tensor(action_space.high) for action_space in env.action_spaces]
        self.action_lows = [self.numpy_to_tensor(action_space.low) for action_space in env.action_spaces]
        
        self.num_agents = env.num_agents

        self.actors, self.critics, self.target_actors, self.target_critics, self.actor_optimizers, self.critic_optimizers  = [], [], [], [], [], []

        for i in range(len(actor_models)):
            self.actors.append(actor_models[i].train().to(self.device))
            self.critics.append(critic_models[i].train().to(self.device))

            self.target_actors.append(copy.deepcopy(actor_models[i]).eval().to(self.device))
            self.target_critics.append(copy.deepcopy(critic_models[i]).eval().to(self.device))

            self.actor_optimizers.append(optim.Adam(self.actors[i].parameters(), lr = actor_learning_rates[i]))
            self.critic_optimizers.append(optim.Adam(self.critics[i].parameters(), lr = critic_learning_rates[i]))

        self.replay_buffer = Multiagent_replay_buffer(self.state_shape,
                                                        self.observation_shapes, 
                                                        self.action_shapes, 
                                                        self.num_agents, 
                                                        self.device, 
                                                        max_size=max_replay_buffer_size, 
                                                        dtype=self.dtype)

        self.loss = torch.nn.MSELoss()


        self.total_iterations = 0

        self.env_done = True
        self.env_observations = None
        self.env_state = None
        self.rewards_histroy = np.zeros((1, self.env.num_agents))
        self.episode_rewards = np.zeros(self.env.num_agents)
        self.total_train_steps = 0

    def train(self, num_train_steps):
        """Train models for num_train_steps, for each train step one or more timesteps in the environment are performed.

        Args:
            num_train_steps ([type]): Number of updates performed on the critic during this training run.

        Returns:
            np.array: Array containing the returns of all finished environments episodes of shape [num_episodes, num_agents]
        """

        if self.env_done:
            self.env_state, self.env_observations = self.env.reset()
            self.env_done = False

        iteration_steps =  max(self.total_iterations, self.burnin_steps) - self.total_iterations + num_train_steps

        for _ in tqdm(range(iteration_steps)):
            self.total_iterations += 1

            if self.env_done:
                self.env_state, self.env_observations = self.env.reset()
                self.rewards_histroy = np.append(self.rewards_histroy, self.episode_rewards.reshape((1,-1)), axis=0)
                self.episode_rewards = np.zeros(self.env.num_agents)

            actions = []
            for actor_id in range(self.env.num_agents):
                with torch.no_grad():
                    actions.append(self.select_action(actor_id, self.env_observations[actor_id]))

            next_state, new_observations, rewards, self.env_done, _ = self.env.step(actions)

            self.replay_buffer.add_transition(self.env_state, next_state, self.env_observations, actions, rewards, new_observations, self.env_done)

            self.episode_rewards = np.sum([self.episode_rewards,rewards], axis=0)

            self.env_observations = new_observations
            self.env_state = next_state

            if(self.burnin_steps < self.total_iterations):

                states_batch, next_states_batch, observations_batch, actions_batch, rewards_batch, next_observations_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
                self.update_critics(states_batch, next_states_batch, actions_batch, rewards_batch, next_observations_batch, done_batch)

                if self.total_train_steps % self.update_target_nets_frequency == 0:
                    self.update_actors(states_batch, observations_batch, actions_batch)
                    self.update_target_nets()

        return self.rewards_histroy


    def update_critics(self, states_batch, next_states_batch, actions_batch, rewards_batch, next_observations_batch, done_batch):
        """Updates the critic according to M3DDPG

        Args:
            states_batch ([type]): Batch of environment states
            next_states_batch ([type]): Batch of next environment states
            actions_batch ([type]): List of batches of actions, one batch per agent. According to order of agents
            rewards_batch ([type]): List of batches of rewards, one batch per agent. According to order of agents
            next_observations_batch ([type]): List of batches of next observations, one batch per agent. According to order of agents 
            done_batch ([type]): Batch of signals whether the episode is done
        """
        with torch.no_grad():
            next_actions_batch_map = _map_function_arg_pairs(self.target_actors, next_observations_batch)
            next_actions_batch = list(map(self.add_noise_to_action, next_actions_batch_map, self.critic_noise_levels, self.noise_clips, self.action_lows, self.action_highs))

        for i, critic in enumerate(self.critics):
            with torch.no_grad():
                next_q_values = self.target_critics[i](next_states_batch, *next_actions_batch)
                q_targets = (rewards_batch[i] + (1-done_batch) * self.discounts[i] * next_q_values).detach()

            q_values = critic(states_batch, *actions_batch)
            
            self.critic_optimizers[i].zero_grad()
            critic_loss = self.loss(q_values, q_targets)
            critic_loss.backward()
            self.critic_optimizers[i].step()

    def update_actors(self, states_batch, observations_batch, actions_batch):
        """Update actors according to M3DDPG

        Args:
            states_batch ([type]): Batch of environment states
            observations_batch ([type]): List of batches of observations, one batch per agent. According to order of agents 
            actions_batch ([type]): List of batches of actions, one batch per agent. According to order of agents
        """

        for i, actor in enumerate(self.actors):
            actions = actor(observations_batch[i])
            joint_actions = copy.deepcopy(actions_batch)
            joint_actions[i] = actions

            self.actor_optimizers[i].zero_grad()
            actor_loss = -self.critics[i](states_batch, *joint_actions).mean()
            actor_loss.backward()
            self.actor_optimizers[i].step()

    def update_target_nets(self):
        """Updates actor and critic target net, uses the tau parameters
        """
        for i in range(self.num_agents):
            self.update_target_net(self.target_actors[i], self.actors[i], self.taus[i])
            self.update_target_net(self.target_critics[i], self.critics[i], self.taus[i])

    def update_target_net(self, target_net, true_net, tau):
        """Updates a specific network

        Args:
            target_net ([type]): Network to update
            true_net ([type]): Network to update to
            tau ([type]): factor to adjust parameters towards true_net
        """
        for target_params, true_params in zip(target_net.parameters(), true_net.parameters()):
            target_params.data.copy_(target_params.data * (1.0 - tau) + true_params.data * tau)

    def select_action(self, actor_id, observation):
        """Selects action of an actor according to a specific observation. Adds explorativ noise and random actions.

        Args:
            actor_id ([type]): Id of actor for which to select an action
            observation ([type]): observation of actor for which to select an action

        Returns:
            np.array: explorativ action
        """
        if(self.burnin_steps >= self.total_iterations or np.random.uniform() <= self.epsilons[actor_id]):
            #take random action
            action = self.env.action_spaces[actor_id].sample()
        else:
            #take greedy action with noise
            torch_observation = self.numpy_to_tensor(observation)
            action = self.actors[actor_id](torch_observation)
            action = self.add_noise_to_action(action, self.noise_levels[actor_id], self.noise_clips[actor_id], self.action_lows[actor_id], self.action_highs[actor_id])
            action = self.tensor_to_numpy(action)

        return action

    def add_noise_to_action(self, action, noise_level, noise_clip, action_low, action_high):
        """Adds noise to an action

        Args:
            action ([type]): action to add noise to
            noise_level ([type]): factor to multiply standard normale noise with
            noise_clip ([type]): clip of noise
            action_low ([type]): lowest possible action values according to environment
            action_high ([type]): highest possible action values according to environment

        Returns:
            torch.tensor: noisy action
        """
        noise = (torch.randn_like(action) * noise_level).clip(-noise_clip, noise_clip)
        return torch.max(torch.min(action + noise, action_high), action_low)

    #Utilility Methods

    def get_policy(self, actor_id):
        """returns a function to easily call the policy of a specific agent

        Args:
            actor_id ([type]): ID of actor from which to get the policy.

        Returns:
            function: Function that takes a observation as numpy array and return an action as numpy array.
        """
        policy = copy.deepcopy(self.actors[actor_id]).eval().cpu()
        def act(state):
            with torch.no_grad():
                tensor_state = torch.tensor(state, dtype=self.dtype, requires_grad=False)
                action = policy(tensor_state)
            return action.numpy()
        return act

    def save_status(self, dir_path, prefix="M3DDPG"):
        """saves status of training, including states of all networks as well as all optimizers

        Args:
            dir_path ([type]): directory path to save files to
            prefix (str, optional): prefix used for all file names. Defaults to "M3DDPG".
        """
        for i in range(self.num_agents):
            actor_file_name = f'{prefix}_actor{i}_{self.total_iterations}its.pt'
            self.save_model(self.actors[i], dir_path, actor_file_name)
            critic_file_name = f'{prefix}_critic{i}_{self.total_iterations}its.pt'
            self.save_model(self.critics[i], dir_path, critic_file_name)
            actor_optimizer_file_name = f'{prefix}_actor{i}_optimizer_{self.total_iterations}its.pt'
            self.save_model(self.actor_optimizers[i], dir_path, actor_optimizer_file_name)
            critic_optimizer_file_name = f'{prefix}_critic{i}_optimizer_{self.total_iterations}its.pt'
            self.save_model(self.critic_optimizers[i], dir_path, critic_optimizer_file_name)

    def load_status(self, dir_path, actor_file_names, critic_file_names, actor_optimizer_file_names, critic_optimizer_file_names):
        """loades status of training, including states of all networks as well as all optimizers

        Args:
            dir_path ([type]): directory path of files to load, all files have to be in the same directory
            actor_file_names ([type]): filename of the actor parameter
            critic_file_names ([type]): filename of the critic paramets
            actor_optimizer_file_names ([type]): filename of the actor optimizer parameters
            critic_optimizer_file_names ([type]): filename of the actor optimizer parameters
        """
        for i in range(self.num_agents): 
            self.load_model(self.actors[i], dir_path, actor_file_names[i])
            self.load_model(self.critics[i], dir_path, critic_file_names[i])
            self.target_actors[i] = copy.deepcopy(self.actors[i]).eval().to(self.device)
            self.target_critics[i] = copy.deepcopy(self.critics[i]).eval().to(self.device)

            self.load_model(self.actor_optimizers[i], dir_path, actor_optimizer_file_names)
            self.load_model(self.critic_optimizers[i], dir_path, critic_optimizer_file_names)

    def save_model(self, model, path, filename):
        """Save a specific model

        Args:
            model ([type]): model to save
            path ([type]): directory path to save to
            filename ([type]): filename used for saving
        """
        save_path = os.path.join(path,filename)
        torch.save(model.state_dict(), save_path)

    def load_model(self, model, path, filename):
        """load a specific model

        Args:
            model ([type]): model to load parameters into
            path ([type]): directory path to load from
            filename ([type]): filename to load from
        """
        load_path = os.path.join(path,filename)
        model.load_state_dict(torch.load(load_path))

    def numpy_to_tensor(self, np_array):
        """converts a numpy array to a torch tensor

        Args:
            np_array (numpy.array): numpy array to convert

        Returns:
            torch.tensor: converted numpy array
        """
        return torch.tensor(np_array, dtype=self.dtype, device=self.device, requires_grad=False)

    def tensor_to_numpy(self, tensor):
        """converts a torch tensor to a numpy array

        Args:
            tensor (torch.tensor): torch tensor to convert

        Returns:
            numpy.array: converted torch tensor
        """
        return tensor.detach().cpu().numpy()

def _map_function_arg_pairs(function_list, arg):
    """Method that maps a list of functions to a list of arguments

    Args:
        function_list ([type]): List of functions
        arg ([type]): List of single arguments

    Returns:
        [type]: map of of functions with arguments put in.
    """
    return map(lambda func, arg: func(arg), function_list, arg)