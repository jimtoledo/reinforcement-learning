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
    outs = tuple(F.pad(input, 2*tuple(0 for _ in range(len(input.shape)-1)) + (length-input.shape[0], 0), value=value).transpose(0, dim) for input in inputs)
    return outs if len(outs) > 1 else outs[0]

def right_pad(length: int, *inputs: torch.Tensor, dim: int = 0, value=0):
    '''
    Right-pads the input tensors to the specified length in the given dimension and return
    '''
    inputs = tuple(input.transpose(0, dim) for input in inputs) #swap with outermost dimension
    #pad outermost dimension, then swap dims back
    outs = tuple(F.pad(input, 2*tuple(0 for _ in range(len(input.shape)-1)) + (0, length-input.shape[0]), value=value).transpose(0, dim) for input in inputs)
    return outs if len(outs) > 1 else outs[0]

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
        self.device = device
    
    def _build_input(self, obs_history: torch.Tensor, agent_ids: list[int] | tuple[int, ...], action_history: torch.Tensor | None = None) -> torch.Tensor:
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
            #Construct last action one-hot tensor: (*batch size, seq length, num agents, max(num available actions))
            last_action_one_hot = torch.empty(action_history.shape, device=action_history.device) #shape: (*batch size, seq length, num agents, 1)
            last_action_one_hot = F.one_hot(action_history, self.output_dim).squeeze(-2) #shape: (*batch size, seq length, num agents, max(num available actions))
            
            #last action at t = action at t-1 for all t>0
            for t in range(last_action_one_hot.shape[-3]-1, 0, -1):
                last_action_one_hot[..., t, :, :] = last_action_one_hot[..., t-1, :, :]
            #zero out last action for initial state
            #last_action_one_hot[..., 0, :, :] = torch.zeros(last_action_one_hot[..., 0, :, :].shape)
            last_action_one_hot[..., 0, :, :] = 0

            #shape: (*batch size, seq length, num agents, obs dim + num agents + max(num available actions))
            nn_input = torch.cat([obs_history, agent_one_hot, last_action_one_hot], dim=-1)
        else:
            nn_input = torch.cat([obs_history, agent_one_hot], dim=-1)
        
        if len(nn_input.shape) == 4: nn_input = nn_input.transpose(0, 1) #swap batch and sequence dims if batch input
        if nn_input.device != self.device: nn_input = nn_input.to(self.device)
        return nn_input #Tensor shape: (seq length, *batch size, num agents, self.input_dim)

    def forward(self, obs_history: torch.Tensor, agent_ids: list[int] | tuple[int, ...], action_history: torch.Tensor | None = None) -> torch.Tensor:
        #agent_ids is list/tuple where index->id matches that of the agents dim in the obs_history/action_history tensor
        #*NOTE*: batch size dimension optional
        #NOTE: agent tensor assumes that for all a in [0, ..., num agents-1]: agent[..., a, :] is filled with a single value
        nn_input = self._build_input(obs_history, agent_ids, action_history) #shape: (seq length, *batch size, num agents, input dim)
        input_shape = tuple(nn_input.shape[:-1])
        nn_input = nn_input.flatten(1, -2) #shape: (seq length, num agents, input size) | (seq length, batch size * num agents, input size)
        nn_output = self.rnn(nn_input)
        nn_output = nn_output.view(*input_shape, self.output_dim) #shape: (seq length, *batch size, num agents, output dim)
        if len(nn_output.shape) == 4: nn_output = nn_output.transpose(0, 1) #swap batch and sequence dims if batch input
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


    def select_action(self, obs_history: torch.Tensor, agent: int, action_history: torch.Tensor | None = None) -> torch.Tensor:
        #Tensor shapes: (*batch size, seq length, *) | (seq length, *)
        if obs_history.shape[-1] < self.obs_dim: #pad obs dim if needed
            #obs_history = F.pad(obs_history, (0, self.obs_dim-obs_history.shape[-1]) + 2*tuple(0 for _ in range(len(obs_history.shape)-1)), value=0)
            obs_history = right_pad(self.obs_dim, obs_history, dim=-1)
        if self.last_action_input:
            #0-filled tensor at current t
            if action_history is None or not len(action_history):
                action_history = torch.zeros(obs_history.shape[:-1], dtype=torch.long, device=obs_history.device).unsqueeze(-1)
            elif action_history.shape[-2] < obs_history.shape[-2]:
                action_history = right_pad(obs_history.shape[-2], action_history, dim=-2)
            
            action_history = action_history.unsqueeze(-2) #add 'num agents' dim of size 1 
        obs_history = obs_history.unsqueeze(-2) #add 'num agents' dim of size 1 
        #return action to take at current t (shape: (*batch size, 1))
        return self(obs_history, (agent,), action_history)[2].detach().cpu()[..., -1, :]
    
    def select_actions(self, obs_history: torch.Tensor, agent_ids: list[int] | tuple[int, ...], action_history: torch.Tensor | None = None) -> torch.Tensor:
        #Tensor shapes: (*batch size, seq length, num agents, *) | (seq length, num agents, *)
        if obs_history.shape[-1] < self.obs_dim: #pad obs dim if needed
            #obs_history = F.pad(obs_history, (0, self.obs_dim-obs_history.shape[-1]) + 2*tuple(0 for _ in range(len(obs_history.shape)-1)), value=0)
            obs_history = right_pad(self.obs_dim, obs_history, dim=-1)
        if self.last_action_input:
            #0-filled tensor at current t
            if action_history is None or not len(action_history):
                action_history = torch.zeros(obs_history.shape[:-1], dtype=torch.long, device=obs_history.device).unsqueeze(-1)
            elif action_history.shape[-3] < obs_history.shape[-3]:
                action_history = right_pad(obs_history.shape[-3], action_history, dim=-3)
        #return action to take at current t (shape: (*batch size, num agents))
        return self(obs_history, agent_ids, action_history)[2].detach().cpu()[..., -1, :]

#QMIX Mixing network
class QMixer(nn.Module):
    #n mixing layers, n agents (hypernetwork output), state dim (for hypernetwork), hypernetwork hidden layers, 
    #mixing network: n_agents -> hidden layers -> 1
    #hypernet: state dim -> hidden layers -> mixing network weights/biases
    def __init__(self, n_agents: int, state_shape: tuple[int, ...] | torch.Size, mixer_hidden_dims: tuple[int, ...] = (32,), hypernet_hidden_dims: tuple[int, ...] | None = (64,)):
        super(QMixer, self).__init__()
        
        self.n_agents = n_agents
        self.state_dim = int(np.prod(state_shape))

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
        ) if hypernet_hidden_dims else None
        
        #Build weights for mixer layers with hypernetwork
        self.mixer_hidden_weights = nn.ModuleList()
        for i, dim in enumerate(mixer_hidden_dims):
            if i == 0: #Q_agent inputs -> first hidden layer
                hypernet_out_dim = self.n_agents * dim
            else:
                hypernet_out_dim = dim * mixer_hidden_dims[i-1]
            
            #Hypernet layers -> mixer weights
            weights = nn.Sequential(
                hypernet_init,
                nn.Linear(hypernet_hidden_dims[-1], hypernet_out_dim)
            ) if hypernet_init else nn.Linear(self.state_dim, hypernet_out_dim)

            self.mixer_weights.append(weights)
        
        #Weights for final mixer layer->Q_total
        self.mixer_final_weights = nn.Sequential(
            hypernet_init,
            nn.Linear(hypernet_hidden_dims[-1], mixer_hidden_dims[-1])
        ) if hypernet_init else nn.Linear(self.state_dim, mixer_hidden_dims[-1])
        
        #Weights use ReLU between hidden layers -> abs activation function
        #Make biases single linear layer, and final bias use hypernet hidden dims
        #The first bias is produced by a hypernetwork with a single linear layer, and the final bias is produced by a two-layer hypernetwork with a ReLU nonlinearity

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

    agent = list(agent_obs_dims.keys())

    action_history = torch.randint(0, 3, (5, 10, 4, 1))

    before = time.time()

    a = rnn(obs_history, agent, action_history)

    b = rnn(obs_history[0], agent, action_history[0])

    # c = rnn1._build_input(obs_history, agent, action_history)

    # d = rnn1._build_input(obs_history[0], agent, action_history[0])

    e = rnn1.select_action(obs_history[..., 1, :], 7, action_history[..., 1, :])

    f = rnn1.select_action(obs_history[0,..., 1, :], 7, action_history[0, ..., 1, :])

    g = rnn(obs_history, agent, action_history)
    h = rnn(obs_history[0], (agent[0], ), action_history[0])
    optimizer = optim.RMSprop(rnn.parameters(), 1e-4)
    for _ in range(20):
        g = rnn(obs_history, agent, action_history)
        loss = nn.MSELoss()(g[1], torch.zeros(g[1].shape, device=g[1].device))
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(time.time() - before)

    from pettingzoo.mpe import simple_speaker_listener_v4
    # env = simple_speaker_listener_v4.env()

    # agent_ids = {agent: id for id, agent in enumerate(env.possible_agents)}
    # agent_obs_dims = {id: env.observation_space(agent).shape[0] for id, agent in enumerate(env.possible_agents)}
    # agent_action_dims = {id: env.action_space(agent).n for id, agent in enumerate(env.possible_agents)}
    # rnn = MultiAgentDRQN(agent_obs_dims, agent_action_dims, last_action_input=True)
    # rnn = MultiAgentDRQN(agent_obs_dims, agent_action_dims, last_action_input=False)

    # env.reset()
    # obs_history = {}
    # act_history = {}
    # for agent in env.agent_iter():
    #     state, reward, terminated, truncated, _ = env.last()
    #     if agent in obs_history:
    #         obs_history[agent] = torch.cat((obs_history[agent], torch.tensor(state, dtype=torch.float32).unsqueeze(0)))
    #     else:
    #         obs_history[agent] = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        
    #     if terminated or truncated:
    #         env.step(None)
    #         if agent in act_history:
    #             act_history[agent] = torch.cat((act_history[agent], torch.zeros(1, 1, dtype=torch.long)))
    #         else:
    #             act_history[agent] = torch.zeros(1, 1, dtype=torch.long)
            
    #     else:
    #         action = rnn.select_action(obs_history[agent], agent_ids[agent], act_history[agent] if agent in act_history else None)
    #         env.step(action.item())

    #         if agent in act_history:
    #             act_history[agent] = torch.cat((act_history[agent], action.unsqueeze(0)))
    #         else:
    #             act_history[agent] = action.unsqueeze(0)
    #             pass
    # print('serial done')

    SEQ_LENGTH = 25
    env = simple_speaker_listener_v4.parallel_env(max_cycles=SEQ_LENGTH)

    agent_ids = {agent: id for id, agent in enumerate(env.possible_agents)}
    agent_obs_dims = {id: env.observation_space(agent).shape[0] for id, agent in enumerate(env.possible_agents)}
    agent_action_dims = {id: env.action_space(agent).n for id, agent in enumerate(env.possible_agents)}
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
    for id in agent_ids.values():
        idx = rnn.agent_id_to_idx[id]
        agent_id[:, idx, 0] = id

    state = env.reset()[0]
    for t in count():
        if not env.agents or t==20: break
        actions = {} #agent: action dict to pass to env.step function
        for agent in env.agents:
            #store agent obs at time t
            obs = torch.tensor(state[agent])
            state_history[t, agent_state_idxs[agent]] = obs #shared state history
            idx = rnn.agent_id_to_idx[agent_ids[agent]]
            obs_history[t, idx] = right_pad(rnn.obs_dim, obs) #individual obs history
            
            action = rnn.select_action(obs_history[:t+1, idx], agent_ids[agent], act_history[:t+1, idx])
            act_history[t, idx] = action #store agent action at time t
            actions[agent] = action.item()

        state, reward, terminated, truncated, _ = env.step(actions)

        for agent in env.agents:
            #store agent next obs at time t
            obs = torch.tensor(state[agent])
            next_state_history[t, agent_state_idxs[agent]] = obs
            idx = rnn.agent_id_to_idx[agent_ids[agent]]
            next_obs_history[t, idx] = right_pad(rnn.obs_dim, obs) #individual obs history

            reward_history[t, idx, 0] = reward[agent]
            is_term_history[t, idx, 0] = terminated[agent]
            pad_mask[t, idx, 0] = False

        #store stuff
    episode = Episode(state_history, obs_history, act_history, reward_history, next_state_history, next_obs_history, is_term_history, agent_id, pad_mask)
    #TODO: replace agent tensor param to agent_idx_to_id mapping
    print('parallel env done')