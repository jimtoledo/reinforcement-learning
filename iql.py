import gymnasium as gym
import numpy as np
from itertools import count
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from tqdm import tqdm

#DQN with double Q-learning, dueling network and prioritized experience replay improvements
#TODO: PER

#Prioritized Replay Buffer module code from https://github.com/mimoralea/gdrl, comments added for more clarity
class PrioritizedReplayBuffer():
    def __init__(self, 
                 max_samples=10000, 
                 batch_size=64, 
                 rank_based=False,
                 alpha=0.6, 
                 beta0=0.1, 
                 beta_rate=0.99992,
                 epsilon=1e-6):
        self.max_samples = max_samples
        self.memory = np.empty(shape=(self.max_samples, 2), dtype=np.ndarray) #2-d array of sample experiences and td errors
        self.batch_size = batch_size
        self.n_entries = 0 #current size of buffer
        self.next_index = 0 #index to add/overwrite new samples
        self.td_error_index = 0 #self.memory[idx, 0] = sample td error
        self.sample_index = 1 #self.memory[idx, 1] = sampled experience
        self.rank_based = rank_based # if not rank_based, then proportional
        self.alpha = alpha # how much prioritization to use 0 is uniform (no priority), 1 is full priority
        self.beta = beta0 # bias correction 0 is no correction 1 is full correction
        self.beta0 = beta0 # beta0 is just beta's initial value
        self.beta_rate = beta_rate
        self.epsilon = epsilon

    #update absolute td errors on sampled experiences
    def update(self, idxs, td_errors):
        self.memory[idxs, self.td_error_index] = np.abs(td_errors)
        if self.rank_based: #if using rank-based PER, resort samples
            sorted_arg = self.memory[:self.n_entries, self.td_error_index].argsort()[::-1]
            self.memory[:self.n_entries] = self.memory[sorted_arg]

    def store(self, sample):
        priority = 1.0
        #set initial priority to max to prioritize unsampled experiences
        if self.n_entries > 0:
            priority = self.memory[
                :self.n_entries, 
                self.td_error_index].max()
        self.memory[self.next_index, 
                    self.td_error_index] = priority #store sample priority
        self.memory[self.next_index, 
                    self.sample_index] = np.array(sample) #store sample experience
        
        self.n_entries = min(self.n_entries + 1, self.max_samples)
        self.next_index += 1
        self.next_index = self.next_index % self.max_samples

    def _update_beta(self):
        self.beta = min(1.0, self.beta * self.beta_rate**-1)
        return self.beta

    def sample(self, batch_size=None):
        batch_size = self.batch_size if batch_size == None else batch_size
        self._update_beta()
        entries = self.memory[:self.n_entries]

        if self.rank_based: #rank based priority: P(i) = 1/rank(i) (highest absolute TD error = 1)
            priorities = 1/(np.arange(self.n_entries) + 1)
        else: # proportional
            priorities = entries[:, self.td_error_index] + self.epsilon #add small epsilon for zero TD error samples
        scaled_priorities = priorities**self.alpha #scale by prioritization hyperparameter (0=uniform/no priority, 1=full priority)
        probs = np.array(scaled_priorities/np.sum(scaled_priorities), dtype=np.float64) #rescale to probabilities

        #introduce importance-sampling weights in loss function to offset prioritization bias
        weights = (self.n_entries * probs)**-self.beta
        normalized_weights = weights/weights.max()

        #sample replay buffer
        idxs = np.random.choice(self.n_entries, batch_size, replace=False, p=probs)
        samples = np.array([entries[idx] for idx in idxs])
        
        #return samples (experiences), indexes (to update absolute TD errors during optimization), and importance-sampling weights (for loss function to offset bias)
        samples_stacks = [np.vstack(batch_type) for batch_type in np.vstack(samples[:, self.sample_index]).T]
        idxs_stack = np.vstack(idxs)
        weights_stack = np.vstack(normalized_weights[idxs])
        return idxs_stack, weights_stack, samples_stacks

    def __len__(self):
        return self.n_entries
    
    def __repr__(self):
        return str(self.memory[:self.n_entries])
    
    def __str__(self):
        return str(self.memory[:self.n_entries])

class FCDuelingQ(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dims=(32,32),
                 v_hidden_dims = (32,),
                 a_hidden_dims = (32,),
                 activation_fc=nn.ReLU,
                 device = torch.device("cpu")):
        super(FCDuelingQ, self).__init__()
        
        #build hidden layers for features, value stream, and advantage stream
        feature_hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            feature_hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            feature_hidden_layers.append(activation_fc())

        v_hidden_layers = nn.ModuleList()
        for i in range(len(v_hidden_dims)-1):
            v_hidden_layers.append(nn.Linear(v_hidden_dims[i], v_hidden_dims[i+1]))
            v_hidden_layers.append(activation_fc())

        a_hidden_layers = nn.ModuleList()
        for i in range(len(a_hidden_dims)-1):
            a_hidden_layers.append(nn.Linear(a_hidden_dims[i], a_hidden_dims[i+1]))
            a_hidden_layers.append(activation_fc())

        #build features, value stream, and advantage stream
        self.features = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            activation_fc(),
            *feature_hidden_layers
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], v_hidden_dims[0]),
            activation_fc(),
            *v_hidden_layers,
            nn.Linear(v_hidden_dims[-1], 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[-1], a_hidden_dims[0]),
            activation_fc(),
            *a_hidden_layers,
            nn.Linear(a_hidden_dims[-1], output_dim)
        )

        if not torch.cuda.is_available():
            device = torch.device("cpu")

        self.device = device
        self.to(self.device)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, 
                             device=self.device, 
                             dtype=torch.float32)
            x = x.unsqueeze(0)      
        return x

    def forward(self, state):
        x = self._format(state)
        features = self.features(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qs = values + (advantages - advantages.mean())
        return qs
    
    def numpy_float_to_device(self, variable):
        variable = torch.from_numpy(variable).float().to(self.device)
        return variable

    def load(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals

class EGreedyExpStrategy():
    def __init__(self, init_epsilon=1.0, min_epsilon=0.1, decay_steps=20000):
        self.epsilon = init_epsilon
        self.init_epsilon = init_epsilon
        self.decay_steps = decay_steps
        self.min_epsilon = min_epsilon
        self.epsilons = 0.01 / np.logspace(-2, 0, decay_steps, endpoint=False) - 0.01
        self.epsilons = self.epsilons * (init_epsilon - min_epsilon) + min_epsilon
        self.t = 0

    def select_action(self, model, state):
        self.exploratory_action_taken = False
        with torch.no_grad():
            if np.random.rand() > self.epsilon:
                action = model(state).detach().max(1).indices.view(1, 1).item() #choose action with highest estimated value
            else:
                action = np.random.randint(model(state).shape[1]) #random action

        self.epsilon = self.min_epsilon if self.t >= self.decay_steps else self.epsilons[self.t]
        self.t += 1
        
        return action

#DDQN with PER
class DDQN():
    def __init__(self, 
                value_model_fn = lambda num_obs, nA: FCDuelingQ(num_obs, nA), #state vars, nA -> model
                value_optimizer_fn = lambda params, lr : optim.RMSprop(params, lr), #model params, lr -> optimizer
                value_optimizer_lr = 1e-4, #optimizer learning rate
                loss_fn = nn.MSELoss(), #input, target -> loss
                exploration_strategy = EGreedyExpStrategy(), #module with select_action function (model, state) -> action
                memory_size = 10000 #replay memory capacity
                ):
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.loss_fn = loss_fn
        self.exploration_strategy = exploration_strategy
        self.memory = PrioritizedReplayBuffer(memory_size)

    def _init_model(self, env):
        #initialize online and target models
        self.online_model = self.value_model_fn(len(env.observation_space.sample()), env.action_space.n)
        self.target_model = self.value_model_fn(len(env.observation_space.sample()), env.action_space.n)
        self.target_model.load_state_dict(self.online_model.state_dict()) #copy online model parameters to target model
        #initialize optimizer
        self.optimizer = self.value_optimizer_fn(self.online_model.parameters(), lr=self.value_optimizer_lr)
    
    def _optimize_model(self):
        experiences = self.memory.sample(self.batch_size)
        experiences = self.online_model.load(experiences)
        states, actions, rewards, next_states, is_terminals = experiences
    
        max_a_q_sp = self.target_model(next_states).detach().max(1)[0].unsqueeze(1) #values for next states
        target_q_sa = rewards + (self.gamma * max_a_q_sp * (1 - is_terminals)) #calculate q target

        q_sa = self.online_model(states).gather(1, actions) #get predicted q from model for each state, action pair

        #loss(weights*q_sa, weights*target_q_sa)
        loss = self.loss_fn(q_sa, target_q_sa) #calculate loss between prediction and target

        #optimize step (gradient descent)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train(self, env, gamma=1.0, num_episodes=100, batch_size=64, n_warmup_batches = 5, tau=0.005, target_update_steps=1, save_models=None):
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
            for t in count():
                i += 1
                action = self.exploration_strategy.select_action(self.online_model, state) #use online model to select action
                next_state, reward, terminated, truncated, _ = env.step(action)
                self.memory.store((state, action, reward, next_state, terminated)) #store experience in replay memory
                
                state = next_state

                if len(self.memory) >= batch_size*n_warmup_batches: #optimize policy model
                    self._optimize_model()

                #update target network with tau
                if i % target_update_steps == 0:
                    #self.target_model.load_state_dict(self.online_model.state_dict())
                    for target, online in zip(self.target_model.parameters(), self.online_model.parameters()):
                        target_weights = tau*online.data + (1-tau)*target.data
                        target.data.copy_(target_weights)

                ep_return += reward * gamma**t #add discounted reward to return
                if terminated or truncated:
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
                    break
        
        return episode_returns, best_model, saved_models

