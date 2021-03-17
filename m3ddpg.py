from numpy import random
import torch
import gym
import copy
from torch.serialization import save
from tqdm import tqdm
from env_wrapper import Tensor_wrapper
from model import TwinNetwork
import numpy as np
from noise_generator import Truncated_normal_noise
import torch.optim as optim
import os

class M3DDPG():
    def __init__(self, 
                environment, 
                actor_model,
                critic_model,
                replay_buffer,
                noise_generator,
                device,
                actor_lr,
                critic_lr,
                discount=0.99,
                tau=0.001,
                noise_level=0.2,
                noise_clip=1, 
                noise_decay=1,
                batch_size=64,
                max_episode_length=500,
                update_actor_frequency = 2,
                epsilon = 0.2,
                save_path_models="./models/",
                save_path_data="./data/"):

        self.dtype = torch.float32
        self.env = environment

        self.single_action_size = 4
        self.num_agents = 2
        self.joint_action_size = self.single_action_size * self.num_agents

        self.actors = TwinNetwork(actor_model).train().to(device)
        self.target_actors = copy.deepcopy(self.actors).eval().to(device)
        self.critics = TwinNetwork(critic_model).train().to(device)
        self.target_critics = copy.deepcopy(self.critics).eval().to(device)

        self.actor_1_optimizer = optim.Adam(self.actors.network_1.parameters(), lr = actor_lr)
        self.actor_2_optimizer = optim.Adam(self.actors.network_2.parameters(), lr = actor_lr)

        self.critic_1_optimizer = optim.Adam(self.critics.network_1.parameters(), lr = critic_lr)
        self.critic_2_optimizer = optim.Adam(self.critics.network_2.parameters(), lr = critic_lr)

        self.device = device
        self.replay_buffer = replay_buffer
        self.update_actor_frequency = update_actor_frequency

        self.save_path_models = save_path_models
        self.save_path_data = save_path_data

        self.noise = noise_generator
        self.noise_level = noise_level
        self.noise_clip = noise_clip
        self.noise_decay = noise_decay
        self.epsilon = epsilon

        self.action_high = self.env.action_space.high.max()
        self.action_low = self.env.action_space.low.min()

        self.MSE = torch.nn.MSELoss()
        
        self.batch_size = batch_size
        self.discount = discount
        self.tau = tau

        self.max_episode_length = max_episode_length

        self.total_iterations = 0

    def train(self, num_iterations):
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