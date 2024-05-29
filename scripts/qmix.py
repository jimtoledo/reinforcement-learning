import numpy as np
from itertools import count
from tensordict import tensorclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tqdm import tqdm
from collections.abc import Mapping
from pettingzoo import AECEnv, ParallelEnv

'''
NOTE: if param sharing (generally if agents have the same observation and action space), include agent ID as part of observation (one-hot encoded)
    https://arxiv.org/pdf/2005.13625.pdf

    if different obs space, zero pad obs to the max?
    if different action space, zero pad input action, clip output action vector for each individual agent
'''

@tensorclass
class Episode:
    states: torch.Tensor
    obs: torch.Tensor #agents dim needed
    actions: torch.Tensor #agents dim needed
    rewards: torch.Tensor #agents dim needed
    next_states: torch.Tensor
    next_obs: torch.Tensor #agents dim needed
    is_terminals: torch.Tensor #agents dim needed
    agents: torch.Tensor #shape [*, 1]: agent ids, agents dim needed
    pad_masks: torch.Tensor #shape [*, 1]: signal if padded to fill sequence, agents dim needed

def to_tensor(*inputs):
    '''
    Transforms inputs into tensors and return
    '''
    outs = []
    for input in inputs:
        out = torch.tensor(input, dtype=torch.float32)
        if not out.shape: out = out.unsqueeze(0) #ensure tensor has at least 1 dimension
        outs.append(out)
    return tuple(outs) if len(outs) > 1 else outs[0]

def left_pad(length: int, *inputs: torch.Tensor, dim: int = 0, value=0):
    '''
    Left-pads the input tensors to the specified length in the given dimension and return
    '''
    inputs = tuple(input.transpose(0, dim) for input in inputs) #swap with outermost dimension
    #pad outermost dimension, then swap dims back
    outs = tuple(F.pad(input, 2*tuple(0 for _ in range(input.ndim-1)) + (length-input.shape[0], 0), value=value).transpose(0, dim) for input in inputs)
    return outs if len(outs) > 1 else outs[0]

def right_pad(length: int, *inputs: torch.Tensor, dim: int = 0, value=0):
    '''
    Right-pads the input tensors to the specified length in the given dimension and return
    '''
    inputs = tuple(input.transpose(0, dim) for input in inputs) #swap with outermost dimension
    #pad outermost dimension, then swap dims back
    outs = tuple(F.pad(input, 2*tuple(0 for _ in range(input.ndim-1)) + (0, length-input.shape[0]), value=value).transpose(0, dim) for input in inputs)
    return outs if len(outs) > 1 else outs[0]


class ExpDecaySchedule():
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000):
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0

    def reset(self):
        self.t = 0
    
    def get(self):
        return self.epsilon
    
    def step(self):
        self.t += 1
        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        return self.epsilon

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
    def __init__(self, agent_obs_dims: Mapping[str, int], agent_action_dims: Mapping[str, int], last_action_input: bool = False,
                 fc1_hidden_dims: tuple[int, ...] = (), fc2_hidden_dims: tuple[int, ...] = (), rnn_dims: tuple[int, int] = (64, 64), 
                 activation_fc = nn.ReLU, device=torch.device("cuda")):
        super(MultiAgentDRQN, self).__init__()
        #TODO: agent_ids from int to string
        self.agent_obs_dims = agent_obs_dims #map agent id -> observation space dim
        self.agent_action_dims = agent_action_dims #map agent id -> # available actions
        self.last_action_input = last_action_input

        self.obs_dim = max(agent_obs_dims.values())
        
        #map agent ids to {0, ..., n_agents-1} for one-hot encoding
        self.agent_sorted = sorted(agent_obs_dims.keys())
        self.agent_id_to_idx = {k:self.agent_sorted.index(k) for k in agent_obs_dims}

        self.n_agents = len(agent_obs_dims)

        # num output features = max(num available actions)
        self.output_dim = max(agent_action_dims.values())
        # num input features = max(observation space dim) + max(num available actions) (if using last action as input) + num agents (one-hot encoded agent ID)
        self.input_dim = self.obs_dim + last_action_input*self.output_dim + self.n_agents

        self.rnn = DRQN(self.input_dim, self.output_dim, fc1_hidden_dims, fc2_hidden_dims, rnn_dims, activation_fc, device)

        self.device = torch.device("cpu") if not torch.cuda.is_available() else device
        self.to(self.device)
    
    def _build_input(self, obs_history: torch.Tensor, agent_ids: list[str] | tuple[str, ...], action_history: torch.Tensor | None = None) -> torch.Tensor:
        #agent_ids is list/tuple where index->id matches that of the agents dim in the obs_history/action_history tensor
        #Tensor shapes: (*batch size, seq length, num agents, *) | (seq length, num agents, *)

        #*NOTE*: batch size dimension optional
        #Construct agent one-hot tensor: (*batch size, seq length, num agents, num agents)
        agent_one_hot = torch.empty(obs_history[..., 0].unsqueeze(-1).shape, dtype=torch.long, device=obs_history.device) #shape: (*batch size, seq length, num agents, 1)
        for idx, id in enumerate(agent_ids): #order of agent_ids should match that of agents dim
            #convert agent id to agent index for all agent ids ([..., n]) in agent dim
            #agent_one_hot[..., n, :] = torch.full(agent[..., n, :].shape, self.agent_id_to_idx[agent[0, ..., n, :].max().item()])
            agent_one_hot[..., idx, :] = self.agent_id_to_idx[id]
        agent_one_hot = F.one_hot(agent_one_hot, self.n_agents).squeeze(-2) #shape: (*batch size, seq length, num agents, 1) -> (*batch size, seq length, num agents, num agents)

        
        if self.last_action_input:
            if action_history.shape[-3] < obs_history.shape[-3]:
                action_history = right_pad(obs_history.shape[-3], action_history, dim=-3)
            #Construct last action one-hot tensor: (*batch size, seq length, num agents, max(num available actions))
            last_action_one_hot = torch.empty(action_history.shape, device=action_history.device) #shape: (*batch size, seq length, num agents, 1)
            last_action_one_hot = F.one_hot(action_history, self.output_dim).squeeze(-2) #shape: (*batch size, seq length, num agents, max(num available actions))
            
            #last action at t = action at t-1 for all t>0
            for t in range(last_action_one_hot.shape[-3]-1, 0, -1):
                last_action_one_hot[..., t, :, :] = last_action_one_hot[..., t-1, :, :]
            #zero out last action for initial state
            last_action_one_hot[..., 0, :, :] = 0

            #shape: (*batch size, seq length, num agents, obs dim + num agents + max(num available actions))
            nn_input = torch.cat([obs_history, agent_one_hot, last_action_one_hot], dim=-1)
        else:
            nn_input = torch.cat([obs_history, agent_one_hot], dim=-1)
        
        if nn_input.ndim == 4: nn_input = nn_input.transpose(0, 1) #swap batch and sequence dims if batch input
        if nn_input.device != self.device: nn_input = nn_input.to(self.device)
        return nn_input #Tensor shape: (seq length, *batch size, num agents, self.input_dim)

    def forward(self, obs_history: torch.Tensor, agent_ids: list[str] | tuple[str, ...], action_history: torch.Tensor | None = None) -> torch.Tensor:
        #agent_ids is list/tuple where index->id matches that of the agents dim in the obs_history/action_history tensor
        #*NOTE*: batch size dimension optional
        nn_input = self._build_input(obs_history, agent_ids, action_history) #shape: (seq length, *batch size, num agents, input dim)
        input_shape = tuple(nn_input.shape[:-1])
        nn_input = nn_input.flatten(1, -2) #shape: (seq length, num agents, input size) | (seq length, batch size * num agents, input size)
        nn_output = self.rnn(nn_input)
        nn_output = nn_output.view(*input_shape, self.output_dim) #shape: (seq length, *batch size, num agents, output dim)
        if nn_output.ndim == 4: nn_output = nn_output.transpose(0, 1) #swap batch and sequence dims if batch input
        #nn_output shape: (*batch size, seq length, num agents, output dim)

        #Calculate max values and actions for each agent
        #shape: (*batch size, seq length, num agents)
        max_values = torch.empty(nn_output.shape[:-1], device=self.device)
        max_actions = torch.empty(nn_output.shape[:-1], device=self.device, dtype=torch.long)
        for idx, id in enumerate(agent_ids): #agents dim
            action_dim = self.agent_action_dims[id]
            agent_output = nn_output[..., idx, :action_dim] #shape: (*batch size, seq length, agent action dim)
            agent_max = agent_output.max(-1)
            max_values[..., idx], max_actions[..., idx] = agent_max.values, agent_max.indices

        return nn_output, max_values, max_actions


    def select_action(self, obs_history: torch.Tensor, agent: str, action_history: torch.Tensor | None = None) -> torch.Tensor:
        #Tensor shapes: (*batch size, seq length, *) | (seq length, *)
        if obs_history.shape[-1] < self.obs_dim: #pad obs dim if needed
            obs_history = right_pad(self.obs_dim, obs_history, dim=-1)
        if self.last_action_input:
            #0-filled tensor at current t
            if action_history is None or not len(action_history):
                action_history = torch.zeros(obs_history.shape[:-1], dtype=torch.long, device=obs_history.device).unsqueeze(-1)
            action_history = action_history.unsqueeze(-2) #add 'num agents' dim of size 1 
        obs_history = obs_history.unsqueeze(-2) #add 'num agents' dim of size 1 
        #return action to take at current t (shape: (*batch size, 1))
        return self(obs_history, (agent,), action_history)[2].detach().cpu()[..., -1, :]
    
    def select_actions(self, obs_history: torch.Tensor, agent_ids: list[str] | tuple[str, ...], action_history: torch.Tensor | None = None) -> torch.Tensor:
        #Tensor shapes: (*batch size, seq length, num agents, *) | (seq length, num agents, *)
        if obs_history.shape[-1] < self.obs_dim: #pad obs dim if needed
            obs_history = right_pad(self.obs_dim, obs_history, dim=-1)
        if self.last_action_input:
            #0-filled tensor at current t
            if action_history is None or not len(action_history):
                action_history = torch.zeros(obs_history.shape[:-1], dtype=torch.long, device=obs_history.device).unsqueeze(-1)
                
        #return action to take at current t (shape: (*batch size, num agents))
        return self(obs_history, agent_ids, action_history)[2].detach().cpu()[..., -1, :]

#QMIX Mixing network
class QMixer(nn.Module):
    #n mixing layers, n agents (hypernetwork output), state dim (for hypernetwork), hypernetwork hidden layers, 
    #mixing network: n_agents -> hidden layers -> 1
    #hypernet: state dim -> hidden layers -> mixing network weights/biases
    def __init__(self, agent_ids: list[str] | tuple[str, ...], state_shape: tuple[int, ...] | torch.Size | int, 
                 mixer_hidden_dims: tuple[int, ...] = (32,), hypernet_hidden_dims: tuple[int, ...] | None = (64,), device=torch.device("cuda")):
        super(QMixer, self).__init__()
        
        self.n_agents = len(agent_ids)
        self.agent_sorted = sorted(agent_ids)
        self.agent_id_to_idx = {k:self.agent_sorted.index(k) for k in agent_ids}
        self.state_dim = state_shape if type(state_shape) == int else int(np.prod(state_shape))

        hypernet_init = None
        if hypernet_hidden_dims:
            #Build extra hypernet hidden layers if len(hypernet_hidden_dims) > 2
            hypernet_hidden = nn.ModuleList()
            for i in range(len(hypernet_hidden_dims)-1):
                hypernet_hidden.append(nn.Linear(hypernet_hidden_dims[i], hypernet_hidden_dims[i+1]))
                hypernet_hidden.append(nn.ReLU())
        
            #If len(hypernet_hidden_dims) > 1, initial + hidden layers
            hypernet_init = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_hidden_dims[0]),
                nn.ReLU(),
                *hypernet_hidden
            )
        
        #Build weights and biases for mixer layers with hypernetwork
        self.mixer_layers = nn.ModuleList()
        for i, dim in enumerate(mixer_hidden_dims):
            if i == 0: #Weights for Q_agent inputs -> first hidden layer
                hypernet_out_dim = self.n_agents * dim
            else: #Weights for hidden layer i-1 -> hidden layer i
                hypernet_out_dim = dim * mixer_hidden_dims[i-1]
            
            #Hypernet layers -> mixer weights
            weights = nn.Sequential(
                hypernet_init,
                nn.Linear(hypernet_hidden_dims[-1], hypernet_out_dim)
            ) if hypernet_init else nn.Linear(self.state_dim, hypernet_out_dim)

            #Bias from single linear layer hypernetwork
            bias = nn.Linear(self.state_dim, dim)

            self.mixer_layers.append(nn.ModuleDict({
                'weights' : weights,
                'bias' : bias
            }))
        
        #Weights for final mixer layer->Q_total
        mixer_final_weights = nn.Sequential(
            hypernet_init,
            nn.Linear(hypernet_hidden_dims[-1], mixer_hidden_dims[-1])
        ) if hypernet_init else nn.Linear(self.state_dim, mixer_hidden_dims[-1])
        
        #V(S) instead of bias for last layers
        V_s = nn.Sequential(
            hypernet_init,
            nn.Linear(hypernet_hidden_dims[-1], 1)
        ) if hypernet_init else nn.Sequential( #if no hypernet hidden dims, construct 2 layer hypernetwork using first mixer hidden dim
            nn.Linear(self.state_dim, mixer_hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(mixer_hidden_dims[0], 1)
        )

        self.mixer_layers.append(nn.ModuleDict({
            'weights': mixer_final_weights,
            'bias': V_s
        }))

        self.device = torch.device("cpu") if not torch.cuda.is_available() else device
        self.to(self.device)
    
    def _format_agent_qs(self, agent_qs: torch.Tensor, agent_ids: list[str] | tuple[str, ...]):
        if agent_qs.ndim == 1: agent_qs = agent_qs.unsqueeze(0)
        agent_qs = agent_qs.view(-1, agent_qs.shape[-1]) #reshape to 2D tensor
        
        #expand agent dim if necessary
        if(len(agent_ids) < self.n_agents):
            cat = torch.zeros(agent_qs.shape[0], self.n_agents - agent_qs.shape[1], device=agent_qs.device)
            agent_qs = torch.cat((agent_qs, cat), dim=1)
            
            #add missing ids to agent_ids
            missing_ids = set(self.agent_sorted) - set(agent_ids)
            agent_ids += list(missing_ids)
        
        #reindex agent dim to match mixer network
        if tuple(agent_ids) != tuple(self.agent_sorted):
            indices = torch.tensor([agent_ids.index(id) for id in self.agent_sorted], dtype=torch.long, device=agent_qs.device)
            agent_qs = agent_qs.index_select(1, indices)
        
        #return agent_qs as batch (1 x n_agents) matrix for mixer network
        return agent_qs.view(-1, 1, self.n_agents)

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor, agent_ids: list[str] | tuple[str, ...]):
        #input shape: (*batch size, *seq length, num agents|state dim)
        in_shape = agent_qs.shape[:-1]
        if agent_qs.device != self.device: agent_qs = agent_qs.to(self.device)
        if states.device != self.device: states = states.to(self.device)
        if states.ndim == 1: states = states.unsqueeze(0)

        #NOTE: len(agent_ids) must match agent_qs.shape[-1]
        x = self._format_agent_qs(agent_qs, agent_ids)
        states = states.view(-1, self.state_dim) #2D tensor (seq length rolled into batch size)
        batch_size = x.shape[0]

        #pass through mixer network layers
        for layer in self.mixer_layers:
            w = torch.abs(layer['weights'](states)) #force non-negative weights for monotonic mixing
            w = w.view(batch_size, x.shape[-1], -1) #batch weight matrix
            b = layer['bias'](states).view(batch_size, 1, -1) #bias
            x = torch.bmm(x, w) + b #apply linear layer
            if layer != self.mixer_layers[-1]: x = F.elu(x) #ELU activation for hidden layers
        
        #Q_tot return shape: (*batch size, *seq length, 1)
        return x.view(*in_shape, 1)

class QMIXLearner():
    def __init__(self, 
            MultiAgentDQN_fn = lambda agent_obs_dims, agent_action_dims: MultiAgentDRQN(agent_obs_dims, agent_action_dims),
            Qmixer_fn = lambda agent_ids, state_shape: QMixer(agent_ids, state_shape),
            optimizer_fn = lambda params, lr : optim.RMSprop(params, lr), #model params, lr -> optimizer
            optimizer_lr = 1e-4, #optimizer learning rate
            loss_fn = nn.MSELoss(), #input, target -> loss
            epsilon_schedule = ExpDecaySchedule(), #module with step, get, and reset
            replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=500, device=torch.device("cuda"))),
            max_gradient_norm = None):
        self.MultiAgentDQN_fn = MultiAgentDQN_fn
        self.Qmixer_fn = Qmixer_fn
        self.optimizer_fn = optimizer_fn
        self.optimizer_lr = optimizer_lr
        self.loss_fn = loss_fn
        self.epsilon_schedule = epsilon_schedule
        self.memory = replay_buffer
        self.max_gradient_norm = max_gradient_norm

    def _init_model(self, env: AECEnv | ParallelEnv):
        self.agents = env.possible_agents
        self.agent_idxs = {agent: idx for idx, agent in enumerate(self.agents)}
        self.agent_obs_dims = {agent: int(np.prod(env.observation_space(agent).shape)) for agent in self.agents}
        self.agent_action_dims = {agent: env.action_space(agent).n for agent in self.agents}
        self.agent_state_idxs = {}
        idx = 0
        for agent in self.agents:
            obs_length = env.observation_space(agent).shape[0]
            self.agent_state_idxs[agent] = slice(idx, idx+obs_length)
            idx += obs_length
            
        #initialize online and target models/mixers
        self.online_model = self.MultiAgentDQN_fn(self.agent_obs_dims, self.agent_action_dims)
        self.target_model = self.MultiAgentDQN_fn(self.agent_obs_dims, self.agent_action_dims)
        self.target_model.load_state_dict(self.online_model.state_dict())

        self.online_mixer = self.Qmixer_fn(self.agents, sum(self.agent_obs_dims.values()))
        self.target_mixer = self.Qmixer_fn(self.agents, sum(self.agent_obs_dims.values()))
        self.target_mixer.load_state_dict(self.online_mixer.state_dict())

        #initialize optimizer
        self.optimizer = self.optimizer_fn(list(self.online_model.parameters()) + list(self.online_mixer.parameters()), lr=self.optimizer_lr)
    
    def _copy_model(self):
        copy = self.MultiAgentDQN_fn(self.agent_obs_dims, self.agent_action_dims)
        copy.load_state_dict(self.online_model.state_dict())
        return copy

    def _update_target(self, tau):
        for target, online in zip(self.target_model.parameters(), self.online_model.parameters()):
            target_weights = tau*online.data + (1-tau)*target.data
            target.data.copy_(target_weights)
        
        for target, online in zip(self.target_mixer.parameters(), self.online_mixer.parameters()):
            target_weights = tau*online.data + (1-tau)*target.data
            target.data.copy_(target_weights)

    def _optimize_model(self):
        batch = self.memory.sample(self.batch_size)
        #remove agent dims
        rewards = batch.rewards[:, :, 0]
        is_terminals = batch.is_terminals.any(dim=-2).long()
        pad = batch.pad_masks.any(dim=-2).long()

        #get predicted Q value with online network/mixer
        agent_qs = self.online_model(batch.obs, self.agents, batch.actions)[1]
        qs = self.online_mixer(agent_qs, batch.states, self.agents)

        #get target Q values
        with torch.no_grad():
            #select best action of next state according to online model (double q learning)
            next_actions = self.online_model(batch.next_obs, self.agents, batch.actions[:, 1:])[2].unsqueeze(-1) #shift action history left for next_actions input
            #get values of next states using target network/mixer
            next_agent_qs = self.target_model(batch.next_obs, self.agents, batch.actions[:, 1:])[0].gather(dim=-1, index=next_actions).squeeze()
            next_qs = self.target_mixer(next_agent_qs, batch.next_states, self.agents).detach()
        target_qs = rewards + (self.gamma * next_qs * (1 - is_terminals))

        #don't include pad entries in loss calculation
        loss = self.loss_fn((1 - pad) * qs, (1 - pad) * target_qs) #calculate loss between prediction and target

        #optimize step (gradient descent)
        self.optimizer.zero_grad()
        loss.backward()
        if self.max_gradient_norm:
            torch.nn.utils.clip_grad_norm_(self.online_model.parameters(), self.max_gradient_norm)
        self.optimizer.step()

    def _select_action(self, env: AECEnv | ParallelEnv, obs_hist: torch.Tensor, agent: str, act_hist: torch.Tensor | None, explore: bool = True) -> torch.Tensor:
        if explore and self.epsilon_schedule.get() > np.random.rand():
            action = torch.tensor(env.action_space(agent).sample()).unsqueeze(0)
        else:
            action = self.online_model.select_action(obs_hist, agent, act_hist)
        return action

    def evaluate(self, env: AECEnv | ParallelEnv, gamma: float = 1.0, max_length: int | None = None, seed: int | None = None) -> dict[str, float]:
        ep_return = {agent: 0 for agent in env.possible_agents}
        rnn = self.online_model
        if max_length:
            obs_history = {agent: torch.zeros(max_length, rnn.obs_dim, dtype=torch.float32) for agent in env.possible_agents}
            act_history = {agent: torch.zeros(max_length, 1, dtype=torch.long) for agent in env.possible_agents}
        else:
            obs_history = {agent: [] for agent in env.possible_agents}
            act_history = {agent: [] for agent in env.possible_agents}
        
        if issubclass(type(env), AECEnv):
            env.reset(seed=seed)
            agent_t = {agent: 0 for agent in env.possible_agents}
            for agent in env.agent_iter():
                t = agent_t[agent]
                obs, reward, terminated, truncated, _ = env.last()

                ep_return[agent] += reward*gamma**t
                if max_length and t >= max_length: break

                obs = right_pad(rnn.obs_dim, torch.tensor(obs))
                if max_length:
                    obs_history[agent][t] = obs
                    curr_obs_hist = obs_history[agent][:t+1]
                    curr_act_hist = act_history[agent][:t+1]
                else:
                    obs_history[agent].append(obs)
                    act_history[agent].append(torch.zeros(1, dtype=torch.long))
                    curr_obs_hist = torch.stack(obs_history[agent])
                    curr_act_hist = torch.stack(act_history[agent])
                
                if terminated or truncated:
                    env.step(None)
                else:
                    action = rnn.select_action(curr_obs_hist, agent, curr_act_hist)
                    env.step(action.item())
                    if max_length:
                        act_history[agent][t] = action
                    else:
                        act_history[agent][-1] = action.cpu()
                
                agent_t[agent] += 1
        else:
            state = env.reset(seed=seed)[0]
            for t in count():
                if not env.agents or (max_length and t >= max_length): break
                actions = {} #agent: action dict to pass to env.step function
                for agent in env.agents:
                    #store agent obs at time t
                    obs = right_pad(rnn.obs_dim, torch.tensor(state[agent]))
                    if max_length:
                        obs_history[agent][t] = obs
                        curr_obs_hist = obs_history[agent][:t+1]
                        curr_act_hist = act_history[agent][:t+1]
                    else:
                        obs_history[agent].append(obs)
                        act_history[agent].append(torch.zeros(1, dtype=torch.long))
                        curr_obs_hist = torch.stack(obs_history[agent])
                        curr_act_hist = torch.stack(act_history[agent])
                    
                    action = rnn.select_action(curr_obs_hist, agent, curr_act_hist)

                    if max_length:
                        act_history[agent][t] = action
                    else:
                        act_history[agent][-1] = action.cpu()
                    
                    actions[agent] = action.item()
                state, reward, _, _, _ = env.step(actions)

                for agent, r in reward.items():
                    ep_return[agent] += r*gamma**t

        return ep_return
    
    def train(self, env: AECEnv | ParallelEnv, gamma: float = 1.0, num_episodes: int = 5000, max_episode_length: int = 500, 
              batch_size: int = 32, n_warmup_batches: int = 1, online_update_steps: int = 1, tau: float = 0.005, target_update_steps: int = 1, 
              save_models: list[int] | None = None, seeds: list[int] = [], evaluate: bool = True):
        
        if save_models: #list of episodes to save models
                save_models.sort()
        self.gamma = gamma
        self.batch_size = batch_size
        self._init_model(env)

        saved_models = {}
        best_model = None
        episode_returns = {agent: [] for agent in self.agents}

        rnn = self.online_model
        state_history = torch.zeros(max_episode_length, sum(self.agent_obs_dims.values()))
        obs_history = torch.zeros(max_episode_length, rnn.n_agents, rnn.obs_dim, dtype=torch.float32) #L = max(obs space length) 
        act_history = torch.zeros(max_episode_length, rnn.n_agents, 1, dtype=torch.long) #L = 1
        reward_history = torch.zeros(max_episode_length, rnn.n_agents, 1, dtype=torch.float32) #L = 1
        next_state_history = torch.zeros(max_episode_length, sum(self.agent_obs_dims.values()))
        next_obs_history = torch.zeros(max_episode_length, rnn.n_agents, rnn.obs_dim, dtype=torch.float32) # L = max(obs space length) 
        is_term_history = torch.zeros(max_episode_length, rnn.n_agents, 1, dtype=torch.bool) #L = 1

        pad_mask = torch.ones(max_episode_length, rnn.n_agents, 1, dtype=torch.bool) #L = 1, pad=1 by default, set to 0 within episode loop
        agent_id = torch.zeros_like(act_history) #L = 1
        for idx in self.agent_idxs.values():
            agent_id[:, idx, 0] = idx
        
        
        for episode in tqdm(range(num_episodes)):
            seed = np.random.choice(seeds).item() if len(seeds) else None
            state = env.reset(seed=seed)
            for agent in self.agents: episode_returns[agent].append(0)

            if issubclass(type(env), AECEnv):
                obs = None
                agent_t = {agent: 0 for agent in self.agents}
                for agent in env.agent_iter():
                    t = agent_t[agent]

                    obs, reward, terminated, truncated, _ = env.last()
                    episode_returns[agent][-1] += reward*gamma**t
                    obs = torch.tensor(obs)
                    idx = self.agent_idxs[agent]
                    
                    #store results of action taken from previous timestep
                    if t > 0 and t <= max_episode_length:
                        next_state_history[t-1, self.agent_state_idxs[agent]] = obs
                        next_obs_history[t-1, idx] = right_pad(rnn.obs_dim, obs)
                        reward_history[t-1, idx, 0] = reward
                        is_term_history[t-1, idx, 0] = terminated
                        pad_mask[t-1, idx, 0] = False

                    if terminated or truncated:
                        env.step(None)
                    elif t >= max_episode_length:
                        env.step(env.action_space(agent).sample())
                    else:
                        #store agent obs at time t
                        state_history[t, self.agent_state_idxs[agent]] = obs #shared state history
                        obs_history[t, idx] = right_pad(rnn.obs_dim, obs) #individual obs history
                        #get and store action taken at time t
                        action = self._select_action(env, obs_history[:t+1, idx], agent, act_history[:t+1, idx])
                        act_history[t, idx] = action
                        env.step(action.item())

                    if min(agent_t.values()) > max_episode_length: break
                    if max(agent_t.values()) < t+1: self.epsilon_schedule.step()
                    agent_t[agent] += 1

            else:
                state = state[0]
                for t in count():
                    if not env.agents or t>=max_episode_length: break
                    actions = {} #agent: action dict to pass to env.step function
                    for agent in env.agents:
                        #store agent obs at time t
                        obs = torch.tensor(state[agent])
                        state_history[t, self.agent_state_idxs[agent]] = obs #shared state history
                        idx = self.agent_idxs[agent]
                        obs_history[t, idx] = right_pad(rnn.obs_dim, obs) #individual obs history
                        
                        action = self._select_action(env, obs_history[:t+1, idx], agent, act_history[:t+1, idx])
                        act_history[t, idx] = action #store agent action at time t
                        actions[agent] = action.item()
                        pad_mask[t, idx, 0] = False

                    state, reward, terminated, truncated, _ = env.step(actions)

                    for agent in env.agents:
                        #store agent next obs at time t
                        obs = torch.tensor(state[agent])
                        next_state_history[t, self.agent_state_idxs[agent]] = obs
                        idx = rnn.agent_id_to_idx[agent]
                        next_obs_history[t, idx] = right_pad(rnn.obs_dim, obs) #individual obs history

                        reward_history[t, idx, 0] = reward[agent]
                        is_term_history[t, idx, 0] = terminated[agent]
                        episode_returns[agent][-1] += reward[agent]*gamma**t
                    
                    self.epsilon_schedule.step()

            #store episode data to memory
            self.memory.add(Episode(state_history, obs_history, act_history, reward_history, next_state_history, next_obs_history, is_term_history, agent_id, pad_mask))

            if evaluate: 
                ep_return = self.evaluate(env, gamma, seed)
                for agent, r in ep_return.items():
                    episode_returns[agent][-1] = r
            if episode_returns[agent][-1] >= np.max(episode_returns[agent]): #save best model
                best_model = self._copy_model()
            #copy and save models
            if save_models and len(saved_models) < len(save_models) and episode+1 == save_models[len(saved_models)]:
                saved_models[episode+1] = self._copy_model()

            if (episode+1) % online_update_steps == 0 and len(self.memory) >= batch_size*n_warmup_batches: #optimize online model
                self._optimize_model()

            #update targets with tau
            if (episode+1) % target_update_steps == 0 and len(self.memory) >= batch_size*n_warmup_batches:
                for target, online in zip(self.target_model.parameters(), self.online_model.parameters()):
                    target_weights = tau*online.data + (1-tau)*target.data
                    target.data.copy_(target_weights)
                for target, online in zip(self.target_mixer.parameters(), self.online_mixer.parameters()):
                    target_weights = tau*online.data + (1-tau)*target.data
                    target.data.copy_(target_weights)
        
        return episode_returns, self.online_model, best_model, saved_models


if __name__ == '__main__':
    # import time
    # agent_obs_dims = {'1': 3, '4': 2, '2': 1, '7': 1}
    # agent_action_dims = {'1': 3, '4': 2, '2': 1, '7': 1}
    # rnn = MultiAgentDRQN(agent_obs_dims, agent_action_dims, last_action_input=True)
    # rnn1 = MultiAgentDRQN(agent_obs_dims, agent_action_dims, last_action_input=False)

    # obs_history = torch.randn(5, 10, 4, 3)

    # agent = list(agent_obs_dims.keys())

    # action_history = torch.randint(0, 3, (5, 10, 4, 1))

    # before = time.time()

    # a = rnn(obs_history, agent, action_history)

    # b = rnn(obs_history[0], agent, action_history[0])

    # c = rnn1._build_input(obs_history, agent, action_history)

    # d = rnn1._build_input(obs_history[0], agent, action_history[0])

    # e = rnn1.select_action(obs_history[..., 1, :], '7', action_history[..., 1, :])

    # f = rnn1.select_action(obs_history[0,..., 1, :], '7', action_history[0, ..., 1, :])

    # g = rnn(obs_history, agent, action_history)
    # h = rnn(obs_history[0], (agent[0], ), action_history[0])
    # optimizer = optim.RMSprop(rnn.parameters(), 1e-4)
    # for _ in range(20):
    #     g = rnn(obs_history, agent, action_history)
    #     loss = nn.MSELoss()(g[1], torch.zeros(g[1].shape, device=g[1].device))
    #     print(loss)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    # print(time.time() - before)

    from pettingzoo.mpe import simple_speaker_listener_v4, simple_spread_v3

    SEQ_LENGTH = 25
    env = simple_speaker_listener_v4.parallel_env(max_cycles=SEQ_LENGTH)
    aec_env = simple_speaker_listener_v4.env(max_cycles=SEQ_LENGTH)

    agent_idxs = {agent: idx for idx, agent in enumerate(env.possible_agents)}
    agent_ids_list = list(agent_idxs.keys())
    agent_obs_dims = {agent: env.observation_space(agent).shape[0] for agent in env.possible_agents}
    agent_action_dims = {agent: env.action_space(agent).n for agent in env.possible_agents}
    rnn = MultiAgentDRQN(agent_obs_dims, agent_action_dims, last_action_input=True)
    agent_state_idxs = {}
    
    idx = 0
    for agent in env.possible_agents:
        obs_length = env.observation_space(agent).shape[0]
        agent_state_idxs[agent] = slice(idx, idx+obs_length)
        idx += obs_length

    #NOTE: assume env observation space is flattened (1 dimension) array
    #NOTE: assume discrete action space

    #dims = (seq length, num agents, L)
    state_history = torch.zeros(SEQ_LENGTH, sum(agent_obs_dims.values()))
    obs_history = torch.zeros(SEQ_LENGTH, rnn.n_agents, rnn.obs_dim, dtype=torch.float32) #L = max(obs space length) 
    act_history = torch.zeros(SEQ_LENGTH, rnn.n_agents, 1, dtype=torch.long) #L = 1
    reward_history = torch.zeros(SEQ_LENGTH, rnn.n_agents, 1, dtype=torch.float32) #L = 1
    next_state_history = torch.zeros(SEQ_LENGTH, sum(agent_obs_dims.values()))
    next_obs_history = torch.zeros(SEQ_LENGTH, rnn.n_agents, rnn.obs_dim, dtype=torch.float32) # L = max(obs space length) 
    is_term_history = torch.zeros(SEQ_LENGTH, rnn.n_agents, 1, dtype=torch.bool) #L = 1

    pad_mask = torch.ones(SEQ_LENGTH, rnn.n_agents, 1, dtype=torch.bool) #L = 1, pad=1 by default, set to 0 within episode loop
    agent_id = torch.zeros_like(act_history) #L = 1
    for idx in agent_idxs.values():
        agent_id[:, idx, 0] = idx

    state = env.reset()[0]
    for t in count():
        if not env.agents or t==20: break
        actions = {} #agent: action dict to pass to env.step function
        for agent in env.agents:
            #store agent obs at time t
            obs = torch.tensor(state[agent])
            state_history[t, agent_state_idxs[agent]] = obs #shared state history
            idx = agent_idxs[agent]
            obs_history[t, idx] = right_pad(rnn.obs_dim, obs) #individual obs history
            
            action = rnn.select_action(obs_history[:t+1, idx], agent, act_history[:t+1, idx])
            act_history[t, idx] = action #store agent action at time t
            actions[agent] = action.item()

        state, reward, terminated, truncated, _ = env.step(actions)

        for agent in env.agents:
            #store agent next obs at time t
            obs = torch.tensor(state[agent])
            next_state_history[t, agent_state_idxs[agent]] = obs
            idx = rnn.agent_id_to_idx[agent]
            next_obs_history[t, idx] = right_pad(rnn.obs_dim, obs) #individual obs history

            reward_history[t, idx, 0] = reward[agent]
            is_term_history[t, idx, 0] = terminated[agent]
            pad_mask[t, idx, 0] = False

        #store stuff
    episode = Episode(state_history, obs_history, act_history, reward_history, next_state_history, next_obs_history, is_term_history, agent_id, pad_mask)
    print('parallel env done')
    res = rnn(episode.obs, agent_ids_list, episode.actions)
    
    x = QMixer(agent_ids_list, sum(agent_obs_dims.values()), (4, 8, 12), (4, 8))
    y = QMixer(agent_ids_list, sum(agent_obs_dims.values()))
    z = QMixer(agent_ids_list, sum(agent_obs_dims.values()), hypernet_hidden_dims=None)
    x(res[1][..., 1:4], episode.states, agent_ids_list[1:4])
    y(torch.cat((res[1].unsqueeze(0),res[1].unsqueeze(0))), torch.cat((episode.states.unsqueeze(0),episode.states.unsqueeze(0))), agent_ids_list)
    z(res[1], episode.states, agent_ids_list)
    x(res[1][0], episode.states[0], agent_ids_list)

    optimizer = optim.RMSprop(x.parameters(), 1e-4)
    for _ in range(20):
        res = rnn(episode.obs, agent_ids_list, episode.actions)
        g = x(torch.cat((res[1].unsqueeze(0),res[1].unsqueeze(0))), torch.cat((episode.states.unsqueeze(0),episode.states.unsqueeze(0))), agent_ids_list)
        loss = nn.MSELoss()(g, torch.zeros(g.shape, device=g.device))
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #TODO: parameter (epsilon/alpha, etc.) schedule instead of 'exploration_strategy'


    print('test')

    learner = QMIXLearner(lambda agent_obs_dims, agent_action_dims: MultiAgentDRQN(agent_obs_dims, agent_action_dims, last_action_input=True))
    learner._init_model(env)
    #learner.train(env, num_episodes=50, max_episode_length=20, evaluate=True)
    x = learner.train(env, num_episodes=50, max_episode_length=SEQ_LENGTH, evaluate=False)
    print('done')