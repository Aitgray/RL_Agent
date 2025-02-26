# Note that I refered to the stable baselines3 library for the implementation of the custom agent.
import carla
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, action_dim=3): # 3 actions: left, right, straight
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) # Fully connected layer (input layer -> hidden layer)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim) # Fully connected layer (hidden layer -> hidden layer)
        self.mean = nn.Linear(hidden_dim, action_dim) # mean of action distribution
        self.log_std = nn.Parameter(torch.zeros(action_dim)) # standard dev of action distribution

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.mean(x)), torch.exp(self.log_std) # We use tanh to bound the action between -1 and 1 and log_std to ensure std is positive

class CustomAgent:
    def __init__(self, vehicle, target_speed, gamma=0.99, lr=3e-4): # Gamma and learning rate are random values, can modify later
        self.vehicle = vehicle
        self.target_speed = target_speed
        self.speed_limits = False
        self.destination = None
        self.gamma = gamma

        self.policy = PolicyNetwork(input_dim=3, hidden_dim=64, action_dim=3)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Buffers for storing the data
        self.log_probs = []
        self.rewards = []
        self.obs = []

    def follow_speed_limits(self, follow):
        self.speed_limits = follow

    def set_destination(self, destination):
        self.destination = destination

    def get_observation(self):
        if self.destination is None:
            raise ValueError("No destination set")
        
        transform = self.vehicle.get_transform()
        location = transform.location
        velocity = self.vehicle.get_velocity()
        current_speed = 3.6 * carla.Vector3D(velocity).length()
        dx  = self.destination.x - location.x
        dy = self.destination.y - location.y
        return np.array([dx, dy, current_speed], dtype=np.float32)

    def select_action(self, observation):
        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0) # Convert to tensor and add batch dimension
        mean, std = self.policy(obs_tensor) # Get mean and std from the policy network
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample() # Sample an action from the distribution

        log_prob = dist.log_prob(action).sum(dim=-1) # Sum over the action dimension
        return action.squeeze(0).detach().numpy(), log_prob

    def run_step(self):
        observation = self.get_observation()
        self.obs.append(observation)
        action, log_prob = self.select_action(observation)
        self.log_probs.append(log_prob)

        throttle = (action[0] + 1) / 2
        steer = np.clip(action[1], -1, 1) # Only turn left or right
        brake = (action[2], 0, 1)

        # Speed limit control
        velocity = self.vehicle.get_velocity()
        current_speed = 3.6 * carla.Vector3D(velocity).length()
        if current_speed > self.target_speed and self.speed_limits:
            throttle = 0.0
            brake = np.clip(brake + 0.1, 0, 1) # Brake to reduce speed

        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.vehicle.apply_control(control)

        return control # idk if I need to actually do this
    
    def store_rewards(self, reward):
        self.rewards.append(reward)

    def update_policy(self):
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)

        # Can normalize the returns if needed

        policy_loss = 0
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss += -log_prob * R

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.log_probs = []
        self.rewards = []
        self.obs = []

    def done(self):
        # Checks if we've reached the destination
        if self.destination is None:
            return True
        location = self.vehicle.get_location()
        dx = self.destination.x - location.x
        dy = self.destination.y - location.y
        # How do I calculate the distance without using the sqrt function?
        distance = dx * dx + dy * dy
        return distance < 1.0 # If distance (squared) is less than 1 meter, return True