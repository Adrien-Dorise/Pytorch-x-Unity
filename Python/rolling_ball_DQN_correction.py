import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
import math

class Network(nn.Module):
    
    def __init__(self, input_size, output_size):
        """ Classic neural network implementation using Pytorch

        Args:
            input_size (int): Number of observations perceived by the agent
            output_size (int): Number of actions
        """
        super(Network, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

#       !!!!!!!!!!!!!! Implement the network architecture !!!!!!!!!!!!!!
        self.fc_input = nn.Linear(input_size, 512)
        self.fc1 = nn.Linear(512,512)
        self.fc_output = nn.Linear(512, output_size)
    
    def forward(self, state):
        """ Classic Pytorch forward implementation

        Args:
            state (torch tensor): Current state (observations)

        Returns:
            torch tensor: Q-value
        """

#       !!!!!!!!!!!!!! Implement the input propagation through the network to get the Q-value !!!!!!!!!!!!!! 
        x = F.relu(self.fc_input(state))
        x = F.relu(self.fc1(x))
        q_values = self.fc_output(x)
        return q_values


class ReplayMemory(object):
    
    def __init__(self, capacity):
        """ Experience replay implementation

        Args:
            capacity (int): Max capacity of the memory
        """
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
        self.capacity = capacity
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """ Save a transition """
        self.memory.append(self.Transition(*args))
    
    def sample(self, batch_size):
        """ Sampling among all pre-registered events

        Args:
            batch_size (int): Number of observations to return in the batch

        Returns:
            list of tensor: Observation batch
        """
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        """ Get memory current size

        Returns:
            int: current memory size
        """
        return len(self.memory)
        


class Dqn():
    
    def __init__(self, input_size, output_size, batch_size=128, gamma=0.99, tau=0.005, lr=1e-4, eps_start=0.9, eps_end=0.05, eps_decay=1000):
        """ Implements the deep Q-learning algorithm

        Args:
            input_size (int): Number of observations given to the agent
            output_size (int): Number of possible actions performed by the agent
            batch_size (int, optional): Number of transitions sampled from the replay buffer. Defaults to 128.
            gamma (float, optional): Discount factor. Defaults to 0.95.
            tau (float, optional): Update rate of the target network. Defaults to 0.005.
            lr (_type_, optional): Learning rate of the optimiser. Defaults to 1e-3.
            eps_start (float, optional): Starting value of epsilon. Defaults to 0.9.
            eps_end (float, optional): Final value of epsilon. Defaults to 0.05.
            eps_decay (int, optional): Rate of the exponential decay of epsilon. Higher value means slower decay. Defaults to 1000.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # !!!!!!!!!!!!!! Parameters can be changed here !!!!!!!!!!!!!!
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.update_frequency = 500
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.Q_net = Network(input_size, output_size).to(self.device)
        self.target_net = Network(input_size, output_size).to(self.device)
        self.memory = ReplayMemory(100000)

        self.output_size = output_size
        self.input_size = input_size
        self.target_net.load_state_dict(self.Q_net.state_dict())
        self.optimizer = optim.Adam(self.Q_net.parameters(), lr=self.lr)
        self.state = torch.Tensor(input_size).unsqueeze(0).to(self.device)
        self.last_action = 0
        self.steps_done = 0
    
    def select_actions(self, state, is_train):
        """ Set a possible action based on the given state of observation
        Depending on the epsilon policy, the returned action can be the result of a random generator (exploration),
        or the result of the trained neural network (exploitation).
        
        Args:
            state (torch tensor): Observations
            is_train (bool): If the model is not in training mode, it will only use exploitation

        Returns:
            torch tensor: Selected action 
        """
        if state is None:
            return torch.tensor([[0]])
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold or not is_train:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = self.Q_net(state).max(1).indices.view(1, 1)
        else:
            action =  torch.tensor([[random.randint(0,self.output_size-1)]], device=self.device, dtype=torch.long)
        
        self.last_action = action
        return action
    
    
    def learn(self, batch_size):
        """Train the neural network based on the data stored in the replay memory

        Args:
            batch_size (int): number of information per batch
        """
        if len(self.memory) < batch_size:
            return
        
        # Batch creation
        transitions = self.memory.sample(batch_size)
        batch = self.memory.Transition(*zip(*transitions))
        
        # We create batches containing only non None values.
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Get Q-value for time t+1
        state_action_values = self.Q_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

#       !!!!!!!!!!!!!! Calculate target !!!!!!!!!!!!!!
        target = (next_state_values * self.gamma) + reward_batch
        

        # Training the network by updating its weight
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.Q_net.parameters(), 100)
        self.optimizer.step()

    
    def update(self, reward, observation, is_dead, is_train):
        """Update the model with the new environment state

        Args:
            reward (int): Current state's reward
            observation (torch tensor): Current state's observations
            is_dead (bool): Did the simulation ended with a reset state?
            is_train (bool); Should we update the model weights?

        Returns:
            torch tensor: New state (observations) registered by the model
        """

        # Translate reward variable into a Pytorch tensor type.
        reward = torch.tensor([reward], device=self.device)
        if is_dead:
            new_state = None
        else:
            new_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)


#       !!!!!!!!!!!!!! Add the new state to memory !!!!!!!!!!!!!!
        self.memory.push(self.state, self.last_action, new_state, reward)
        self.state = new_state
        
        # Training
        if len(self.memory) > self.batch_size and is_train:
            self.learn(self.batch_size)


#        !!!!!!!!!!!!!! Update target network at some frequency F !!!!!!!!!!!!!!
        self.target_net_state_dict = self.target_net.state_dict()
        self.Q_net_state_dict = self.Q_net.state_dict()
        for key in self.Q_net_state_dict:
            self.target_net_state_dict[key] = self.Q_net_state_dict[key]*self.tau + self.target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(self.target_net_state_dict)

        return new_state

    def save(self):
        """Save models in root folder
        """
        torch.save({'state_dict': self.Q_net.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        """Load model from root folder
        """
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.Q_net.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")


