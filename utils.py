import random
from collections import deque

import torch
import torch.nn.functional as F
from gym.core import Env
from torch import nn

class ReplayBuffer():
    def __init__(self, size:int):
        """Replay buffer initialisation

        Args:
            size: maximum numbers of objects stored by replay buffer
        """
        self.size = size
        self.buffer = deque([], size)
    
    def push(self, transition)->list:
        """Push an object to the replay buffer

        Args:
            transition: object to be stored in replay buffer. Can be of any type
        
        Returns:
            The current memory of the buffer (any iterable object e.g. list)
        """  
        self.buffer.append(transition)
        return self.buffer

    def sample(self, batch_size:int)->list:
        """Get a random sample from the replay buffer
        
        Args:
            batch_size: size of sample

        Returns:
            iterable (e.g. list) with objects sampled from buffer without replacement
        """
        return random.sample(self.buffer, batch_size)

# This is the Deep Q-learning Network, which inherits from the Neural Network module
class DQN(nn.Module):
    # For the initialization, the DQN takes in the layer size as a list of ints
    def __init__(self, layer_sizes:list[int]):
        """
        DQN initialisation

        Args:
            layer_sizes: list with size of each layer as elements
        """
        # Super() basically means it inherits ther __init__() from nn.Module
        super(DQN, self).__init__()
        # self.layers will be created from torch NN, where ModuleList() creates lists of Module objects, where Module 
        # is the base class for all NN modules. This list will consist of len(layer_sizes - 1) number of Linear Layers,
        # each each with in_features and out_features
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
    
    # The -> here means that the function returns a torch Tensor, and takes in x which is a torch Tensor as well
    def forward (self, x:torch.Tensor)->torch.Tensor:
        """Forward pass through the DQN

        Args:
            x: input to the DQN
        
        Returns:
            outputted value by the DQN
        """
        # For each layer
        for layer in self.layers:
            # RELU the output of the linear layer on the input x. F is torch.nn.functional, which is just a master class
            # of many different types of functions
            x = F.relu(layer(x))
        return x

# Function returns an int
def greedy_action(dqn:DQN, state:torch.Tensor)->int:
    """Select action according to a given DQN
    
    Args:
        dqn: the DQN that selects the action
        state: state at which the action is chosen

    Returns:
        Greedy action according to DQN
    """
    # This returns the index of the max action as given by the DQN from the passed in state
    return int(torch.argmax(dqn(state)))

# Function returns an int
def epsilon_greedy(epsilon:float, dqn:DQN, state:torch.Tensor)->int:
    """Sample an epsilon-greedy action according to a given DQN
    
    Args:
        epsilon: parameter for epsilon-greedy action selection
        dqn: the DQN that selects the action
        state: state at which the action is chosen
    
    Returns:
        Sampled epsilon-greedy action
    """
    # Get the Q(s,a) values from the DQN on the passed state
    q_values = dqn(state)
    # The number of actions will be the number of rows in the q_values numpy array
    num_actions = q_values.shape[0]
    # The greedy action will be the index of the max value in q_values
    greedy_act = int(torch.argmax(q_values))
    # This returns a tensor filled with random numbers between [0, 1). The shape of the tensor is 1 here
    p = float(torch.rand(1))
    # If this uniform random number is greater than epsilon, where epsilon defines the probability of our exploration, then take
    # the greedy act. Else take a random act as defined from 0 to the number of actions - 1
    if p > epsilon:
        return greedy_act
    else:
        return random.randint(0,num_actions-1)


# Function takes in the target network and policy network, and modifies the target network
def update_target(target_dqn:DQN, policy_dqn:DQN):
    """Update target network parameters using policy network.
    Does not return anything but modifies the target network passed as parameter
    
    Args:
        target_dqn: target network to be modified in-place
        policy_dqn: the DQN that selects the action
    """

    # A state_dict is a Python dictionary object that maps each layer to its parameter tensor, where parameters are weights and biases
    # Here, the target DQN is the one that we load with the main network's parameters once every 1000 epochs
    target_dqn.load_state_dict(policy_dqn.state_dict())

def loss(policy_dqn:DQN, target_dqn:DQN,
         states:torch.Tensor, actions:torch.Tensor,
         rewards:torch.Tensor, next_states:torch.Tensor, dones:torch.Tensor,
         ddqnFlag:bool = False)->torch.Tensor:
    """Calculate Bellman error loss
    
    Args:
        policy_dqn: policy DQN
        target_dqn: target DQN
        states: batched state tensor
        actions: batched action tensor
        rewards: batched rewards tensor
        next_states: batched next states tensor
        dones: batched Boolean tensor, True when episode terminates
        ddqnFlag: Boolean that determines whether we're doing DDQN or DQN, default to DQN
    
    Returns:
        Float scalar tensor with loss value
    """

    # This is the Qmax + rewards. Bellman_targets contains the necessary policy that wants to keep the pole straight, while q_values
    # comes from the main network which trains all the time, and we want to minimize the difference between the main and network
    # Q values
    # In order to change the model to be a DDQN, we change this such that the policy network chooses the Bellman target
    # action, and the target network predicts the q values as the bellman targets from the action chosen by the policy network.
    # We say that the q values are predicted by the target network in that we move the q value observations towards these bellman targets
    # If DDQN
    if ddqnFlag:
        # Choose best actions from the policy network
        best_actions = policy_dqn(next_states).argmax(1).reshape(-1,1)
        # Calculate bellman targets from target network based on actions from policy network
        bellman_targets = (~dones).reshape(-1)*(target_dqn(next_states).gather(1, best_actions)).reshape(-1) + rewards.reshape(-1)
    # If DQN
    else:
        # Find the actions and q value estimation both from the target network
        bellman_targets = (~dones).reshape(-1)*(target_dqn(next_states)).max(1).values + rewards.reshape(-1)

    q_values = policy_dqn(states).gather(1, actions).reshape(-1)

    return ((q_values - bellman_targets)**2).mean()