import gym
import random
from model import CartModel
from replay import ReplayBuffer
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import os
from gym.wrappers.monitoring import video_recorder

class Environment:
    def __init__(self, config_path: str):
        
        with open(config_path, 'r') as file: 
            self.config = yaml.safe_load(file)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
         
        self.env = gym.make('CartPole-v1')
        self.model = CartModel(self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
        self.target_model = CartModel(self.env.observation_space.shape[0], self.env.action_space.n).to(self.device)
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(self.config['max_memory'])

        self.target_update_freq = 10
        
        self.epsilon = self.config['epsilon']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_decay = self.config['epsilon_decay']

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def train_dqn(self, path: str = None):
        for episode in range(self.config["episode"]):
            state, _ = self.env.reset()
            total_reward = 0
            done = False
            
            while not done:
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else: 
                    q_values = self.model(torch.tensor(state).float().to(self.device))
                    action = torch.argmax(q_values).item()
                    
                next_state, reward, done, _, _ = self.env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
                if len(self.replay_buffer) > self.config['batch_size']:
                    states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config['batch_size'])
                    next_q_values = self.target_model(next_states.to(self.device)).max(dim=1)[0].unsqueeze(1).detach().cpu()
                    targets = rewards.unsqueeze(1) + self.config["gamma"] * next_q_values * (1 - dones.unsqueeze(1))
                    targets = targets.to(self.device)
                    current_q_values = self.model(states.to(self.device)).gather(1, actions.to(self.device))
                    
                    loss = self.criterion(targets, current_q_values)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            
            if episode % self.target_update_freq == 0:
                self.update_target_model()
                    
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            print(f"Episode {episode + 1}/{self.config['episode']}, Reward: {total_reward}, Epsilon: {self.epsilon:.3f}")
        
        self.model.save(os.path.join(path, 'cartpole.pth'))
        
    def test_dqn(self, path: str):
        self.model.load(os.path.join(path, 'cartpole.pth'))
        
        video = video_recorder.VideoRecorder(self.env, path=os.path.join(path, 'video.mp4'))
        
        state, _ = self.env.reset()
        
        total_reward = 0
        done = False
        
        while not done:
            video.capture_frame()
            
            q_values = self.model(torch.tensor(state).float().to(self.device))
            action = torch.argmax(q_values).item()
            
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            
        video.close()
        print(f"Test completed with total reward: {total_reward}")
        return total_reward
        
    def close(self):
        self.env.close()
