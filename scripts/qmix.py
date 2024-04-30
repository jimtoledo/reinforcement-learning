import numpy as np
from itertools import count
from tensordict import tensorclass
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections.abc import Mapping

'''
NOTE: if param sharing (generally if agents have the same observation and action space), include agent ID as part of observation (one-hot encoded)
    https://arxiv.org/pdf/2005.13625.pdf

    if different obs space, zero pad obs to the max?
    if different action space, zero pad input action, clip output action vector for each individual agent
'''

@tensorclass
class Episode:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor 
    is_terminals: torch.Tensor
    agent: torch.Tensor
    pad_masks: torch.Tensor #shape [*, 1]: signal if padded to fill sequence

#Deep Recurrent Q Network
class DRQN(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, 
                 fc1_hidden_dims: tuple[int, ...] = (), fc2_hidden_dims: tuple[int, ...] = (), rnn_dims: tuple[int, int] = (64, 64),
                 activation_fc = nn.ReLU, device=torch.device("cuda")):

        super(DRQN, self).__init__()
        self.activation_fc = activation_fc

        #Build MLP hidden layers
        fc1_hidden_layers = nn.ModuleList()
        for i in range(len(fc1_hidden_dims)-1):
            fc1_hidden_layers.append(nn.Linear(fc1_hidden_dims[i], fc1_hidden_dims[i+1]))
            fc1_hidden_layers.append(activation_fc())

        fc2_hidden_layers = nn.ModuleList()
        for i in range(len(fc2_hidden_dims)-1):
            fc2_hidden_layers.append(nn.Linear(fc2_hidden_dims[i], fc2_hidden_dims[i+1]))
            fc2_hidden_layers.append(activation_fc())
        
        #MLP 1: if no hidden layers, single linear transform from input dim to RNN input dim with activation
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, fc1_hidden_dims[0]),
            activation_fc(),
            *fc1_hidden_layers,
            nn.Linear(fc1_hidden_dims[-1], rnn_dims[0]),
            activation_fc()
        ) if fc1_hidden_dims else nn.Sequential(nn.Linear(input_dim, rnn_dims[0]), activation_fc())

        #MLP 2: if no hidden layers, single linear transform from RNN output dim to (action) output dim
        self.fc2 = nn.Sequential(
            nn.Linear(rnn_dims[-1], fc2_hidden_dims[0]),
            activation_fc(),
            *fc2_hidden_layers,
            nn.Linear(fc2_hidden_dims[-1], output_dim)
        ) if fc2_hidden_dims else nn.Linear(rnn_dims[-1], output_dim)

        #GRU cell - NOTE:batch_first=False by default, input/output shape should be (sequence length, batch size, input size)
        self.rnn = nn.GRUCell(*rnn_dims)

        self.device = torch.device("cpu") if not torch.cuda.is_available() else device
        self.to(self.device)

    def _format(self, input):
        x = input
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, 
                             device=self.device, 
                             dtype=torch.float32)
            #x = x.unsqueeze(0)      
        else:
            x = x.to(self.device)
        return x

    #input shape: (sequence length, input size) | (sequence length, batch size, input size)
    def forward(self, inputs):
        inputs = self._format(inputs)
        outs = []

        batch_size = inputs.shape[1] if inputs.ndim == 3 else 0
        rnn_hidden_dims = self.rnn.weight_hh.shape[1]

        #batch process sequence
        hx = torch.zeros(batch_size, rnn_hidden_dims, device=self.device) if batch_size else torch.zeros(rnn_hidden_dims, device=self.device) #initial RNN hidden layer
        for t in range(inputs.shape[0]): #sequence length
            inputs_t = inputs[t] #shape: (input size) | (batch size, input size)
            x = self.fc1(inputs_t)
            hx = self.rnn(x, hx)
            q = self.fc2(hx) #output shape: (output size) | (batch size, output size)
            outs.append(q) #store batch output at each timestep while stepping through sequence
            #.unsqueeze(0)
        
        qs = torch.stack(outs) #shape: (sequence length, # actions) | (sequence length, batch size, # actions)
        return qs
        #torch.stack(dim=1)
    
    #input shape: (sequence length, input size) | (sequence length, batch size, input size)
    def select_action(self, inputs):
        inputs = self._format(inputs)
        #return current action to be taken (end of sequence)
        #shape: (# actions) | (batch size, # actions)
        batch_size = inputs.shape[1] if inputs.ndim == 3 else 0
        rnn_hidden_dims = self.rnn.weight_hh.shape[1]

        #batch process sequence
        hx = torch.zeros(batch_size, rnn_hidden_dims, device=self.device) if batch_size else torch.zeros(rnn_hidden_dims, device=self.device) #initial RNN hidden layer
        for t in range(inputs.shape[0]): #sequence length
            inputs_t = inputs[t] #shape: (input size) | (batch size, input size)
            x = self.fc1(inputs_t)
            hx = self.rnn(x, hx)
        
        q = self.fc2(hx) #shape: (# actions) | (batch size, # actions)

        #return discrete (one-hot) action
        return q.detach().max(-1, keepdim=True).indices #output shape: (1) | (batch size, 1)


#Deep Q Recurrent Neural Network
class MultiAgentDRQN(nn.Module):

    #agent n_action, n_obs dicts needed
    def __init__(self, agent_obs_dims: Mapping[int, int], agent_action_dims: Mapping[int, int], last_action_input: bool = False,
                 fc1_hidden_dims: tuple[int, ...] = (), fc2_hidden_dims: tuple[int, ...] = (), rnn_dims: tuple[int, int] = (64, 64), 
                 activation_fc = nn.ReLU, device=torch.device("cuda")):
        super(MultiAgentDRQN, self).__init__()

        self.agent_obs_dims = agent_obs_dims #map agent id -> observation space dim
        self.agent_action_dims = agent_action_dims #map agent id -> # available actions
        self.last_action_input = last_action_input

        # num input features = max(observation space dim) + max(num available actions) (if using last action as input) + num agents (one-hot encoded agent ID)
        input_dim = max(agent_obs_dims.values()) + last_action_input*max(agent_action_dims.values()) + len(agent_obs_dims)
        # num output features = max(num available actions)
        output_dim = max(agent_action_dims.values())

        self.rnn = DRQN(input_dim, output_dim, fc1_hidden_dims, fc2_hidden_dims, rnn_dims, activation_fc, device)

    #_build_input(self, batch)
        #last action (t-1)
        #agent id one-hot encoding
        #torch cat [obs, last action, agent id one-hot encoded], dim=-1

    #forward(self, batch)
        #reshape (sequence length, batch size, num agents, input size) -> (sequence length, batch size * num agents, input size)
        #forward self.rnn
        #reshape (sequence length, batch size * num agents, input size) -> (sequence length, batch size, num agents, input size)

    #observation and agent's last action (one-hot encoded) as input
    #all samples from episode batch through MLP -> GRU -> MLP

#QMIX Mixing network
class QMixer(nn.Module):
    pass

class QMIX():
    pass

'''
NOTE: input to multi-agent DQRNN should be episode batch of shape (N, L, X)
    N = batch size
    L = length of longest sequence (shorter sequences should be zero-padded at the beginning?)
        process episode batch from L=t+1=1 to L=t_max+1, where t_max is the max timestep of the longest episode in batch
    X = flattened feature vector (agent obs, agent last action, agent identifier)
'''
