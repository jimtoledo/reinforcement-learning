import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from tensordict import tensorclass

@tensorclass
class Experience:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor 
    is_terminals: torch.Tensor
    pad_masks: torch.Tensor #shape [*, 1]: signal if padded to fill sequence

def to_tensor(*inputs):
    '''
    Transforms inputs into tensors and return
    '''
    outs = []
    for input in inputs:
        out = torch.tensor(input, dtype=torch.float32)
        if not out.shape: out = out.unsqueeze(0) #ensure tensor has at least 1 dimension
        outs.append(out)
    return tuple(outs)

def left_pad(length: int, *inputs: torch.Tensor, value: float=0):
    '''
    Left-pads the input tensors to the specified length in the outermost (0) dimension and return
    '''
    return tuple(F.pad(input, 2*tuple(0 for _ in range(len(input.shape)-1)) + (length-input.shape[0], 0), value=value) for input in inputs)

#Deep Recurrent Q Network
class DRQN(nn.Module):

    def __init__(self, input_dim, output_dim, 
                 fc1_hidden_dims = (), fc2_hidden_dims = (), rnn_dims = (64, 64), activation_fc = nn.ReLU, device=torch.device("cuda")):

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
        return x

    #input shape: (sequence length, input size) | (sequence length, batch size, input size)
    def forward(self, inputs):
        self._format(inputs)
        outs = []

        batch_size = inputs.shape[1] if inputs.ndim == 3 else 0
        rnn_hidden_dims = self.rnn.weight_hh.shape[1]

        #batch process sequence
        hx = torch.zeros(batch_size, rnn_hidden_dims) if batch_size else torch.zeros(rnn_hidden_dims) #initial RNN hidden layer
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
        self._format(inputs)
        #return current action to be taken (end of sequence)
        #shape: (# actions) | (batch size, # actions)
        batch_size = inputs.shape[1] if inputs.ndim == 3 else 0
        rnn_hidden_dims = self.rnn.weight_hh.shape[1]

        #batch process sequence
        hx = torch.zeros(batch_size, rnn_hidden_dims) if batch_size else torch.zeros(rnn_hidden_dims) #initial RNN hidden layer
        for t in range(inputs.shape[0]): #sequence length
            inputs_t = inputs[t] #shape: (input size) | (batch size, input size)
            x = self.fc1(inputs_t)
            hx = self.rnn(x, hx)
        
        q = self.fc2(hx) #shape: (# actions) | (batch size, # actions)

        #return discrete (one-hot) action
        #TODO: maybe convert tensor output to numpy?
        return q.detach().max(-1, keepdim=True).indices #output shape: (1) | (batch size, 1)

class EGreedyExpStrategy():
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000):
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0

    def select_action(self, model, state, n_actions):
        with torch.no_grad():
            if np.random.rand() > self.epsilon:
                action = model.select_action(state).item() #choose action with highest estimated value
            else:
                action = np.random.randint(n_actions) #random action

        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        
        return action

class DRQNLearner():
    def __init__(self, 
                DRQN_fn = lambda num_obs, nA: DRQN(num_obs, nA), #state vars, nA -> model
                optimizer_fn = lambda params, lr : optim.RMSprop(params, lr), #model params, lr -> optimizer
                optimizer_lr = 1e-4, #optimizer learning rate
                loss_fn = nn.MSELoss(), #input, target -> loss
                exploration_strategy = EGreedyExpStrategy(), #module with select_action function (model, state) -> action
                replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=500, device=torch.device("cuda"))),
                max_gradient_norm = None):
        self.DRQN_fn = DRQN_fn
        self.optimizer_fn = optimizer_fn
        self.optimizer_lr = optimizer_lr
        self.loss_fn = loss_fn
        self.exploration_strategy = exploration_strategy
        self.memory = replay_buffer
        self.max_gradient_norm = max_gradient_norm

    def _init_model(self, env):
        #initialize online and target models
        self.online_model = self.DRQN_fn(len(env.observation_space.sample()), env.action_space.n)
        self.target_model = self.DRQN_fn(len(env.observation_space.sample()), env.action_space.n)
        self.target_model.load_state_dict(self.online_model.state_dict()) #copy online model parameters to target model
        #initialize optimizer
        self.optimizer = self.optimizer_fn(self.online_model.parameters(), lr=self.optimizer_lr)

    def train(self, env, gamma=1.0, num_episodes=5000, max_episode_length=500, batch_size=32, n_warmup_batches = 1, tau=0.005, target_update_steps=1, save_models=None):
        if save_models: #list of episodes to save models
                save_models.sort()
        self.gamma = gamma
        self.batch_size = batch_size
        self._init_model(env)

        saved_models = {}
        best_model = None

        i = 0
        episode_returns = np.zeros(num_episodes)
        for episode in tqdm(range(num_episodes)):
            state = env.reset()[0]
            ep_return = 0
            sequence = []
            #Experience() in episode loop
            #torch.stack() at end of episode
            #F.pad(stack., pad=(0, 0, max_episode_length - shape[0], ))
            for t in range(max_episode_length):
                i += 1
                action = self.exploration_strategy.select_action(self.online_model, state, env.action_space.n) #use online model to select action
                next_state, reward, terminated, truncated, _ = env.step(action)

                sequence.append(Experience(*to_tensor(state, action, reward, next_state, terminated), torch.tensor([0]))) #add experience to episode sequence

                state = next_state
                ep_return += reward * gamma**t #add discounted reward to return
                if terminated or truncated: break

            #TODO: evaluate and save
            #save best model
            if ep_return >= episode_returns.max():
                copy = self.value_model_fn(len(env.observation_space.sample()), env.action_space.n)
                copy.load_state_dict(self.online_model.state_dict())
                best_model = copy
            #copy and save model
            if save_models and len(saved_models) < len(save_models) and episode+1 == save_models[len(saved_models)]:
                copy = self.value_model_fn(len(env.observation_space.sample()), env.action_space.n)
                copy.load_state_dict(self.online_model.state_dict())
                saved_models[episode+1] = copy

            episode_returns[episode] = ep_return

            history = torch.stack(sequence)
            #pad episode experience history to max_episode_length
            states, actions, rewards, next_states, is_terminals = left_pad(max_episode_length, history.states, history.actions, history.rewards, history.next_states, history.is_terminals)
            pad_masks = left_pad(max_episode_length, history.pad_masks, value=1)
            self.memory.add(Experience(states, actions, rewards, next_states, is_terminals, pad_masks))

            if len(self.memory) >= batch_size*n_warmup_batches: #optimize online model
                self._optimize_model()

            #update target network with tau
            if i % target_update_steps == 0:
                #self.target_model.load_state_dict(self.online_model.state_dict())
                for target, online in zip(self.target_model.parameters(), self.online_model.parameters()):
                    target_weights = tau*online.data + (1-tau)*target.data
                    target.data.copy_(target_weights)

        
        return episode_returns, self.online_model, best_model, saved_models

if __name__ == '__main__':
    from torchrl.data import ReplayBuffer, LazyTensorStorage
    replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=1000, device=torch.device("cuda")))