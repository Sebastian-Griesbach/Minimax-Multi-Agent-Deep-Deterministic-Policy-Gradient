from typing import Dict, Callable, List, Union
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
    ZERO_DIVISION_PREVENTION = 1e-8

    def __init__(self, 
                env: Multiagent_wrapper, 
                actor_models: List[torch.nn.Module],
                critic_models: List[torch.nn.Module],
                actor_learning_rates: List[float],
                critic_learning_rates: List[float],
                device: str,
                discounts: List[float],
                taus: List[float],
                noise_levels: List[float],
                noise_clips: List[float],
                critic_noise_levels: List[float],
                epsilons: List[float],
                alphas: List[float],
                max_replay_buffer_size: int,
                burnin_steps: int = 10000,
                burnin_policies: Union[List[Callable[[np.array], np.array]], None] = None,
                update_target_nets_frequency: int = 2,
                batch_size: int = 64) -> None:
        """initalize algorithm, prepares necessary objects and saves hyperparameters

        Args:
            env (Multiagent_wrapper): gym Multiagent_wrapper of training environment
            actor_models (List[torch.nn.Model]): List of actor models according to order of agents
            critic_models (List[torch.nn.Model]): List of critics models according to order of agents
            actor_learning_rates (List[float]): List of actor learning rates according to order of agents
            critic_learning_rates (List[float]): List of critic learning rates according to order of agents
            device (str): cuda device to use for model training
            discounts (List[float]): List of discount factors for future rewards according to order of agents
            taus (List[float]): List of tau factors used to update the target net, both for critics and actors, according to order of agents
            noise_levels (List[float]): List of factors to multiply standart normal noise with for which is used as exploration noise, according to order of agents
            noise_clips (List[float]): List of values the exploration noise will be clipped to. Only one value per agent, according to order of agents.
            critic_noise_levels (List[float]): List of factors multiplied with standart normal noise. This noise is added to the target_values of the critics during training. According to order of agents
            epsilons (List[float]): List of probability values to take an entirly random action (sampeled from the actionspace) at each time step, for more exploration additionally to exploration noise. According to order of agents
            alphas (List[float]): List of scaling factors for adversarial gradient step on opponent actions during training. According to order of agents.
            max_replay_buffer_size (int): Maximal number of time steps the Replaybuffer will hold. When this number is reached the oldest entries will be overwritten.
            burnin_steps (int, optional): number of timesteps taken befor the training routine starts. During this time random actions sampeled from the actionspace will be performed. Defaults to 10000.. Defaults to 10000.
            burnin_policies Union[List[Callable[[np.array], np.array]], None]: Policies to use during the burinphase. According to order of Agents. Defaults to random sample from actionspaces of agents.
            update_target_nets_frequency (int, optional): Frequency of timesteps at which the targents as well as the original actor net ist updated. Defaults to 2.
            batch_size (int, optional): Number of timesteps that are take into account at each update steps. Defaults to 64.. Defaults to 64.
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
        self.alphas = alphas
        self.batch_size = batch_size
        self.update_target_nets_frequency = update_target_nets_frequency
        self.burnin_steps = burnin_steps

        self.burnin_policies = burnin_policies
        if(self.burnin_policies == None):
            self.burnin_policies = []
            for action_space in self.env.action_spaces:
                self.burnin_policies.append(lambda obs: action_space.sample())

        self.state_shape = env.state_space.shape
        self.action_shapes = [action_space.shape for action_space in env.action_spaces]
        self.observation_shapes = [observation_space.shape for observation_space in env.observation_spaces]

        self.action_highs = [self._numpy_to_tensor(action_space.high) for action_space in env.action_spaces]
        self.action_lows = [self._numpy_to_tensor(action_space.low) for action_space in env.action_spaces]
        
        self.num_agents = env.num_agents

        self.actors, self.critics, self.target_actors, self.target_critics, self.actor_optimizers, self.critic_optimizers  = [], [], [], [], [], []

        for i in range(self.num_agents):
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

    def train(self, num_train_steps: int) -> np.array:
        """Train models for num_train_steps, for each train step one or more timesteps in the environment are performed.

        Args:
            num_train_steps (int): Number of updates performed on the critic during this training run.

        Returns:
            numpy.array: Array containing the returns of all finished environments episodes of shape [num_episodes, num_agents]
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
                    actions.append(self._select_action(actor_id, self.env_observations[actor_id]))

            #self.env.render()
            next_state, new_observations, rewards, self.env_done, _ = self.env.step(actions)

            self.replay_buffer.add_transition(self.env_state, next_state, self.env_observations, actions, rewards, new_observations, self.env_done)

            self.episode_rewards = np.sum([self.episode_rewards,rewards], axis=0)

            self.env_observations = new_observations
            self.env_state = next_state

            if(self.burnin_steps < self.total_iterations):

                states_batch, next_states_batch, observations_batch, actions_batch, rewards_batch, next_observations_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
                self._update_critics(states_batch, next_states_batch, actions_batch, rewards_batch, next_observations_batch, done_batch)

                if self.total_train_steps % self.update_target_nets_frequency == 0:
                    self._update_actors(states_batch, observations_batch, actions_batch)
                    self._update_target_nets()

        return self.rewards_histroy


    def _update_critics(self, states_batch, next_states_batch, actions_batch, rewards_batch, next_observations_batch, done_batch):
        """Updates the critic according to M3DDPG

        Args:
            states_batch: Batch of environment states
            next_states_batch: Batch of next environment states
            actions_batch: List of batches of actions, one batch per agent. According to order of agents
            rewards_batch: List of batches of rewards, one batch per agent. According to order of agents
            next_observations_batch: List of batches of next observations, one batch per agent. According to order of agents 
            done_batch: Batch of signals whether the episode is done
        """
        with torch.no_grad():
            next_actions_batch = list(_map_function_arg_pairs(self.target_actors, next_observations_batch))

        for i, critic in enumerate(self.critics):
            if self.alphas[i] > 0.:
                adversarial_next_actions = self._get_adverserial_actions(next_states_batch, next_actions_batch, self.target_critics[i])
            else:
                adversarial_next_actions = next_actions_batch
            adversarial_next_actions[i] = next_actions_batch[i]
            adversarial_next_actions = list(map(self._add_noise_to_action, adversarial_next_actions, self.critic_noise_levels, self.noise_clips, self.action_lows, self.action_highs))
            with torch.no_grad():
                next_q_values = self.target_critics[i](next_states_batch, adversarial_next_actions)
                q_targets = (rewards_batch[i] + (1.-done_batch) * self.discounts[i] * next_q_values).detach()

            q_values = critic(states_batch, actions_batch)
            
            self.critic_optimizers[i].zero_grad()
            critic_loss = self.loss(q_values, q_targets)
            critic_loss.backward()
            self.critic_optimizers[i].step()

    def _update_actors(self, states_batch, observations_batch, actions_batch):
        """Update actors according to M3DDPG

        Args:
            states_batch: Batch of environment states
            observations_batch: List of batches of observations, one batch per agent. According to order of agents 
            actions_batch: List of batches of actions, one batch per agent. According to order of agents
        """

        for i, actor in enumerate(self.actors):
            actions = actor(observations_batch[i])
            joint_actions = copy.deepcopy(actions_batch)
            if self.alphas[i] > 0.:
                adversarial_joint_actions = self._get_adverserial_actions(states_batch, joint_actions, self.critics[i])
            else:
                adversarial_joint_actions = joint_actions
            adversarial_joint_actions[i] = actions

            self.actor_optimizers[i].zero_grad()
            actor_loss = -self.critics[i](states_batch, adversarial_joint_actions).mean()
            actor_loss.backward()
            self.actor_optimizers[i].step()

    def _get_adverserial_actions(self, states, actions, critic):
        """Creates adverserial perturbed actions based on the gradient of the critic.

        Args:
            states: Batch of states in which the actions took place.
            actions: List of batches of actions taken by each agent in the according states.
            critic: critic after which gradient to adapt the actions.

        Returns:
            List[torch.tensor]: List of batches Adverserially perturbed actions of all agents.
        """

        actions = list(map(lambda action: action.requires_grad_(True), actions))
        actor_gain = critic(states, actions).mean()
        actor_gain.backward()
        adverserial_actions = []
        with torch.no_grad():
            for i, action in enumerate(actions):
                action_gradient = action.grad
                gradient_norm = torch.torch.linalg.norm(action_gradient, dim=1, keepdim=True)
                action_norm = torch.torch.linalg.norm(action, dim=1, keepdim=True)
                perturbation = - self.alphas[i] * action_norm * (action_gradient/(gradient_norm+self.ZERO_DIVISION_PREVENTION))
                adverserial_action = torch.max(torch.min(action+perturbation, self.action_highs[i]), self.action_lows[i]).detach()
                adverserial_actions.append(adverserial_action)

        return adverserial_actions

    def _update_target_nets(self):
        """Updates actor and critic target net, uses the tau parameters
        """
        for i in range(self.num_agents):
            self._update_target_net(self.target_actors[i], self.actors[i], self.taus[i])
            self._update_target_net(self.target_critics[i], self.critics[i], self.taus[i])

    def _update_target_net(self, target_net, true_net, tau):
        """Updates a specific network

        Args:
            target_net: Network to update
            true_net: Network to update to
            tau: factor to adjust parameters towards true_net
        """
        for target_params, true_params in zip(target_net.parameters(), true_net.parameters()):
            target_params.data.copy_(target_params.data * (1.0 - tau) + true_params.data * tau)

    def _select_action(self, actor_id, observation):
        """Selects action of an actor according to a specific observation. Adds explorativ noise and random actions.

        Args:
            actor_id: Id of actor for which to select an action
            observation: observation of actor for which to select an action

        Returns:
            np.array: explorativ action
        """

        if(self.burnin_steps >= self.total_iterations):
            action = self.burnin_policies[actor_id](observation)
        elif(np.random.uniform() <= self.epsilons[actor_id]):
            #take random action
            action = self.env.action_spaces[actor_id].sample()
        else:
            #take greedy action with noise
            torch_observation = self._numpy_to_tensor(observation).unsqueeze(0)
            self.actors[actor_id].eval()
            action = self.actors[actor_id](torch_observation).squeeze(dim=0)
            self.actors[actor_id].train()
            action = self._add_noise_to_action(action, self.noise_levels[actor_id], self.noise_clips[actor_id], self.action_lows[actor_id], self.action_highs[actor_id])
            action = self._tensor_to_numpy(action)

        return action

    def _add_noise_to_action(self, action, noise_level, noise_clip, action_low, action_high):
        """Adds noise to an action

        Args:
            action: action to add noise to
            noise_level: factor to multiply standard normale noise with
            noise_clip: clip of noise
            action_low: lowest possible action values according to environment
            action_high: highest possible action values according to environment

        Returns:
            torch.tensor: noisy action
        """
        noise = (torch.randn_like(action) * noise_level).clip(-noise_clip, noise_clip)
        return torch.max(torch.min(action + noise, action_high), action_low)

    def get_policy(self, actor_id: int) -> Callable[[np.array], np.array]:
        """returns a function to easily call the policy of a specific agent

        Args:
            actor_id (int): ID of actor from which to get the policy.

        Returns:
            Callable[[numpy.array], numpy.array]: Function that takes a observation as numpy array and return an action as numpy array.
        """
        policy = copy.deepcopy(self.actors[actor_id]).eval().cpu()
        def act(obs: np.array) -> np.array:
            """Takes observation and returns actions according to policy

            Args:
                obs (numpy.array): Observation to act on

            Returns:
                numpy.array: action of policy
            """
            with torch.no_grad():
                tensor_obs = torch.tensor(obs, dtype=self.dtype, requires_grad=False).unsqueeze(0)
                action = policy(tensor_obs).squeeze(dim=0)
            return action.numpy()
        return act

    def save_status(self, dir_path: str, prefix="M3DDPG") -> None:
        """saves status of training, including states of all networks as well as all optimizers

        Args:
            dir_path (str): directory path to save files to
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

    def load_status(self, dir_path: str, actor_file_names: List[str], critic_file_names: List[str], actor_optimizer_file_names: List[str], critic_optimizer_file_names: List[str]) -> None:
        """loades status of training, including states of all networks as well as all optimizers

        Args:
            dir_path (str): directory path of files to load, all files have to be in the same directory
            actor_file_names (List[str]): filename of the actor parameter
            critic_file_names (List[str]): filename of the critic paramets
            actor_optimizer_file_names (List[str]): filename of the actor optimizer parameters
            critic_optimizer_file_names (List[str]): filename of the actor optimizer parameters
        """
        for i in range(self.num_agents): 
            self.load_model(self.actors[i], dir_path, actor_file_names[i])
            self.load_model(self.critics[i], dir_path, critic_file_names[i])
            self.target_actors[i] = copy.deepcopy(self.actors[i]).eval().to(self.device)
            self.target_critics[i] = copy.deepcopy(self.critics[i]).eval().to(self.device)

            self.load_model(self.actor_optimizers[i], dir_path, actor_optimizer_file_names[i])
            self.load_model(self.critic_optimizers[i], dir_path, critic_optimizer_file_names[i])

    def save_model(self, model: Union[torch.nn.Module, optim.Optimizer], path: str, filename: str) -> None:
        """Save a specific model or optimizer

        Args:
            model (Union[torch.nn.Model, torch.optim.Optimizer]): model or optimizer to save
            path (str): directory path to save to
            filename (str): filename used for saving
        """
        save_path = os.path.join(path,filename)
        torch.save(model.state_dict(), save_path)

    def load_model(self, model: Union[torch.nn.Module, optim.Optimizer], path: str, filename: str):
        """load parameters of a specific model or optimizer

        Args:
            model (Union[torch.nn.Model, optim.Optimizer]): model or optimizer to load parameters into
            path (str): directory path to load from
            filename (str): filename to load from
        """
        load_path = os.path.join(path,filename)
        model.load_state_dict(torch.load(load_path))

    def _numpy_to_tensor(self, np_array):
        """converts a numpy array to a torch tensor

        Args:
            np_array: numpy array to convert

        Returns:
            torch.tensor: converted numpy array
        """
        return torch.tensor(np_array, dtype=self.dtype, device=self.device, requires_grad=False)

    def _tensor_to_numpy(self, tensor):
        """converts a torch tensor to a numpy array

        Args:
            tensor: torch tensor to convert

        Returns:
            numpy.array: converted torch tensor
        """
        return tensor.detach().cpu().numpy()

def _map_function_arg_pairs(function_list, arg):
    """Method that maps a list of functions to a list of arguments

    Args:
        function_list: List of functions
        arg: List of single arguments

    Returns:
        Map: map of of functions with injected arguments.
    """
    return map(lambda func, arg: func(arg), function_list, arg)