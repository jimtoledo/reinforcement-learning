import numpy as np
from itertools import count
from tensordict import tensorclass
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        #map agent ids to {0, ..., n_agents-1} for one-hot encoding
        self.agent_sorted = sorted(agent_obs_dims.keys())
        self.agent_id_to_idx = {k:self.agent_sorted.index(k) for k in agent_obs_dims}

        self.n_agents = len(agent_obs_dims)

        # num output features = max(num available actions)
        self.output_dim = max(agent_action_dims.values())
        # num input features = max(observation space dim) + max(num available actions) (if using last action as input) + num agents (one-hot encoded agent ID)
        self.input_dim = max(agent_obs_dims.values()) + last_action_input*self.output_dim + self.n_agents

        self.rnn = DRQN(self.input_dim, self.output_dim, fc1_hidden_dims, fc2_hidden_dims, rnn_dims, activation_fc, device)
        self.device = device
    
    def _build_input(self, obs_history: torch.Tensor, agent: torch.Tensor, action_history: torch.Tensor | None = None) -> torch.Tensor:
        #Tensor shapes: (*batch size, seq length, num agents, *) | (seq length, num agents, *)

        #*NOTE*: batch size dimension optional
        #Construct agent one-hot tensor: (*batch size, seq length, num agents, num agents)
        agent_one_hot = torch.empty(agent.shape, dtype=torch.long, device=agent.device)
        for n in range(agent.shape[-2]): #agents dim
            #convert agent id to agent index for all agent ids ([..., n]) in agent dim
            agent_one_hot[..., n, :] = torch.full(agent[..., n, :].shape, self.agent_id_to_idx[agent[0, ..., n, :].max().item()])
        agent_one_hot = F.one_hot(agent_one_hot, self.n_agents).squeeze(-2) #shape: (*batch size, seq length, num agents, 1) -> (*batch size, seq length, num agents, num agents)

        
        if self.last_action_input:
            #Construct last action one-hot tensor: (*batch size, seq length, num agents, max(num available actions))
            last_action_one_hot = torch.empty(action_history.shape, device=action_history.device) #shape: (*batch size, seq length, num agents, 1)
            last_action_one_hot = F.one_hot(action_history, self.output_dim).squeeze(-2) #shape: (*batch size, seq length, num agents, max(num available actions))
            
            #last action at t = action at t-1 for all t>0
            for t in range(last_action_one_hot.shape[-3]-1, 0, -1):
                last_action_one_hot[..., t, :, :] = last_action_one_hot[..., t-1, :, :]
            #zero out last action for initial state
            last_action_one_hot[..., 0, :, :] = torch.zeros(last_action_one_hot[..., 0, :, :].shape)

            #shape: (*batch size, seq length, num agents, obs dim + num agents + max(num available actions))
            nn_input = torch.cat([obs_history, agent_one_hot, last_action_one_hot], dim=-1)
        else:
            nn_input = torch.cat([obs_history, agent_one_hot], dim=-1)
        
        if len(nn_input.shape) == 4: nn_input = nn_input.transpose(0, 1) #swap batch and sequence dims if batch input
        if nn_input.device != self.device: nn_input = nn_input.to(self.device)
        return nn_input #Tensor shape: (seq length, *batch size, num agents, self.input_dim)

    def forward(self, obs_history: torch.Tensor, agent: torch.Tensor, action_history: torch.Tensor | None = None) -> torch.Tensor:
        #*NOTE*: batch size dimension optional
        #NOTE: agent tensor assumes that for all a in [0, ..., num agents-1]: agent[..., a, :] is filled with a single value
        nn_input = self._build_input(obs_history, agent, action_history) #shape: (seq length, *batch size, num agents, input dim)
        input_shape = tuple(nn_input.shape[:-1])
        nn_input = nn_input.flatten(1, -2) #shape: (seq length, num agents, input size) | (seq length, batch size * num agents, input size)
        nn_output = self.rnn(nn_input)
        nn_output = nn_output.reshape(*input_shape, self.output_dim) #shape: (seq length, *batch size, num agents, output dim)
        if len(nn_output.shape) == 4: nn_output = nn_output.transpose(0, 1) #swap batch and sequence dims if batch input
        #nn_output shape: (*batch size, seq length, num agents, output dim)

        #Calculate max values and actions for each agent
        #shape: (seq length, *batch size, num agents)
        max_values = torch.empty(nn_output.shape[:-1], device=self.device)
        max_actions = torch.empty(nn_output.shape[:-1], device=self.device)
        for n in range(agent.shape[-2]): #agents dim
            agent_id = agent[0, ..., n, :].max().item()
            action_dim = self.agent_action_dims[agent_id]
            agent_output = nn_output[..., n, :action_dim] #shape: (*batch size, seq length, agent action dim)
            agent_max = agent_output.max(-1)
            max_values[..., n], max_actions[..., n] = agent_max.values, agent_max.indices

        return nn_output, max_values, max_actions


    def select_action(self, obs_history: torch.Tensor, agent: int, action_history: torch.Tensor | None = None) -> torch.Tensor:
        #Tensor shapes: (*batch size, seq length, *) | (seq length, *)
        obs_history, action_history = obs_history.unsqueeze(-2), action_history.unsqueeze(-2) #add 'num agents' dim of size 1 
        agent_tensor = torch.full(obs_history[..., 0].unsqueeze(-1).shape, agent, dtype=torch.long, device=obs_history.device) #shape: (*batch size, seq length, 1, 1)
        return self(obs_history, agent_tensor, action_history)[2].squeeze(-1)

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

if __name__ == '__main__':
    import time
    agent_obs_dims = {1: 3, 4: 2, 2: 1, 7: 1}
    agent_action_dims = {1: 3, 4: 2, 2: 1, 7: 1}
    rnn = MultiAgentDRQN(agent_obs_dims, agent_action_dims, last_action_input=True)
    rnn1 = MultiAgentDRQN(agent_obs_dims, agent_action_dims, last_action_input=False)

    obs_history = torch.randn(5, 10, 4, 3)

    agent = torch.zeros(5, 10, 4, 1)
    for a in range(agent.shape[-2]):
        agent[..., a, :] = torch.full(agent[..., a, :].shape, list(agent_action_dims.keys())[a])
    
    action_history = torch.randint(0, 3, (5, 10, 4, 1))

    before = time.time()

    a = rnn._build_input(obs_history, agent, action_history)

    b = rnn._build_input(obs_history[0], agent[0], action_history[0])

    c = rnn1._build_input(obs_history, agent, action_history)

    d = rnn1._build_input(obs_history[0], agent[0], action_history[0])

    e = rnn1.select_action(obs_history[..., 1, :], 4, action_history[..., 1, :])

    f = rnn1.select_action(obs_history[0,..., 1, :], 4, action_history[0, ..., 1, :])

    g = rnn(obs_history, agent, action_history)
    h = rnn(obs_history[0], agent[0], action_history[0])
    optimizer = optim.RMSprop(rnn.parameters(), 1e-4)
    for _ in range(20):
        g = rnn(obs_history, agent, action_history)
        loss = nn.MSELoss()(g[1], torch.zeros(g[1].shape, device=g[1].device))
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(time.time() - before)