# RL_Agent
A RL agent built for Carla

## Deliverable:
The RL algorithm isn't particularly good. It crashes frequently, and doesn't drive very well.

Utilizing the Stable Baselines3 library as a reference, I used PyTorch to build the reinforcement learning agent. My core objective was to enable an autonomous vehicle to navigate to a destination whilst adhering to speed constraints.

Within the PolicyNetwork class you can find my neural network, which is constructed as follows:

The input layer accepts a feature vector containing the relative destination coordinates and the current speed.
Two hidden layers (with 64 neurons each) use rectifiers (ReLU) to introduce non-linearity. This is important not only for capturing complex patterns in the data but also for ensuring that gradients flow properly during training.
The output layers produce the mean and standard deviation of a normal distribution. I bounded the mean between -1 and 1 to keep the actions within a sensible range. These outputs don’t choose a single action directly; instead, they parameterize a probability distribution, allowing the agent to sample a variety of actions and encouraging exploration.

The CustomAgent class contains the vehicle’s behavior and the learning loop. Key attributes include the vehicle and speed parameters, destination management, and buffers for reinforcement learning. The buffers store observations, rewards, and log probabilities (i.e., the log likelihood of the selected actions given the distribution from the network). These log probabilities are essential for the policy gradient updates, where they help reinforce actions that lead to higher rewards by influencing the network's learning.

In order to enhance the algorithm I think there are a number of impactful improvements I could make. I could test different configurations for the hidden layers, throwing more (or less) neurons at the problem could lead to improved results. I need to fine tune the hyperparameters like the learning rate and discount factor (ideally I’d like to find a way to automate this). Additionally, adding additional sensory inputs/better state representations could improve the model as well. Increasing the number of inputs and giving the model more data would almost certainly improve its overall performance. Incorporating evaluation metrics is also a must, it would give me a better idea of how the agent performs in different scenarios/environments and it would give me a clearer picture of how to improve the agent overall. Adding in something like experience replay could also provide a large benefit to the algorithm.
