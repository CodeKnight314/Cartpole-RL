import gym
import random
from model import CartModel
from replay import ReplayBuffer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import yaml
import os
from gym.wrappers.monitoring import video_recorder

class Environment:
    def __init__(self, config_path: str):
        
        with open(config_path, 'r') as file: 
            self.config = yaml.safe_load(file)
         
        self.env = gym.make('CartPole-v1')
        self.model = CartModel(self.env.observation_space.shape[0], self.env.action_space.n)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(self.config['max_memory'])
        
        self.epsilon = self.config['epsilon']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_decay = self.config['epsilon_decay']
        
    def train_dqn(self, path: str = None):
        for episode in range(self.config["episode"]):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else: 
                    q_values = self.model(torch.tensor(state).float())
                    action = torch.argmax(q_values).item()
                    
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.add((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward
                
                if len(self.replay_buffer) > self.config['batch_size']:
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config['batch_size'])
                    next_q_values = self.model(states).max(dim=1)[0].unsqueeze(1).detach()
                    
                    targets = reward + self.config["gamma"] * next_q_values * (1 - dones)
                    current_q_values = self.model(states).gather(1, actions)
                    
                    loss = self.criterion(targets, current_q_values)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
            epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            print(f"Episode {episode + 1}/{episode}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")
        
        self.model.save(os.path.join(path, 'cartpole.pth'))
        
    def test_dqn(self, path: str):
        self.model.load(os.path.join(path, 'cartpole.pth'))
        
        video = video_recorder.VideoRecorder(self.env, path=os.path.join(path, 'video.mp4'))
        self.env.reset()
        
        done = False
        while not done:
            video.capture_frame()
            q_values = self.model(torch.tensor(state).float())
            action = torch.argmax(q_values).item()
            state, _, done, _ = self.env.step(action)
            
        video.close()
        
    def close(self):
        self.env.close()
