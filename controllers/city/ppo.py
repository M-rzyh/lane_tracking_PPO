import gym
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO

class CustomPolicy(nn.Module):
    def __init__(self, observation_space, action_space):
        super(CustomPolicy, self).__init__()
        # Define the architecture of the neural network here
        self.fc1 = nn.Linear(observation_space, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_space)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Assuming the action space is normalized to [-1, 1]
        return x
    
    
class PPOAgent:
    def __init__(self, observation_space, action_space):
        # Initialize the agent with the custom policy
        self.model = PPO(CustomPolicy, observation_space, action_space, verbose=1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)