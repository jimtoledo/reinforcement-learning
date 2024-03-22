import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

#Fully connected deterministic policy network
class FCDP(nn.Module):
    def __init__(self, 
                 input_dim, 
                 action_bounds, #[action_mins], [action_maxs]
                 hidden_dims=(32,32), 
                 activation_fc=nn.ReLU,
                 out_activation_fc=nn.Tanh,
                 device = torch.device("cpu")):
        super(FCDP, self).__init__()
        self.activation_fc = activation_fc
        self.out_activation_fc = out_activation_fc
        self.action_min, self.action_max = action_bounds

        hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            hidden_layers.append(activation_fc())
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            activation_fc(),
            *hidden_layers,
            nn.Linear(hidden_dims[-1], len(self.action_max)),
            out_activation_fc()
        )

        self.action_min = torch.tensor(self.action_min, device=device, dtype=torch.float32)
        self.action_max = torch.tensor(self.action_max, device=device, dtype=torch.float32)
        
        #get min/max output of last activation function for rescaling to action values
        self.nn_min = out_activation_fc()(torch.Tensor([float('-inf')])).to(device)
        self.nn_max = out_activation_fc()(torch.Tensor([float('inf')])).to(device)

        self.device = device
        self.to(self.device)

    #rescale nn outputs to fit within action bounds
    def _rescale_fn(self, x):
        return (x - self.nn_min) * (self.action_max - self.action_min) / (self.nn_max - self.nn_min) + self.action_min

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
        x = self.layers(x)
        return self._rescale_fn(x)
    
    def select_action(self, state):
        return self.forward(state).cpu().detach().numpy().squeeze(axis=0)

#Fully-connected twin value network (state observation, action -> value_1, value_2)
#'Shared' critic network as described in MADDPG paper https://arxiv.org/pdf/1706.02275.pdf
#TODO: implement shared critic architecture
class Shared_FCTQV(nn.Module):
    def __init__(self, 
                 state_dim,
                 action_dim,
                 hidden_dims=(32,32), #define hidden layers as tuple where each element is an int representing # of neurons at a layer
                 activation_fc=nn.ReLU,
                 device = torch.device("cpu")):
        super(Shared_FCTQV, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        hidden_layers = (nn.ModuleList(), nn.ModuleList())  #layers tuple for twin value networks
        for i in range(len(hidden_dims)-1):
            [l.append(nn.Linear(hidden_dims[i], hidden_dims[i+1])) for l in hidden_layers]
            [l.append(activation_fc()) for l in hidden_layers]
        
        layers = [nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dims[0]),
            activation_fc(),
            *l,
            nn.Linear(hidden_dims[-1], 1)
        ) for l in hidden_layers] #layers for twin value networks

        self.critic1 = layers[0]
        self.critic2 = layers[1]

        self.device = device
        self.to(device)
        
    def _format(self, state, action):
        x, u = state, action
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, 
                             device=self.device, 
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        if not isinstance(u, torch.Tensor):
            u = torch.tensor(u, 
                             device=self.device, 
                             dtype=torch.float32)
            u = u.unsqueeze(0)
        return torch.cat((x,u), dim=1)

    def forward(self, state, action):
        x = self._format(state, action)
        return self.critic1(x), self.critic2(x)
    
    def Q1(self, state, action):
        x = self._format(state, action)
        return self.critic1(x)

    def Q2(self, state, action):
        x = self._format(state, action)
        return self.critic2(x)
    
    def load(self, experiences):
        states, actions, new_states, rewards, is_terminals = experiences
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        new_states = torch.from_numpy(new_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        is_terminals = torch.from_numpy(is_terminals).float().to(self.device)
        return states, actions, new_states, rewards, is_terminals

    def numpy_float_to_device(self, variable):
        variable = torch.from_numpy(variable).float().to(self.device)
        return variable

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
                    self.sample_index] = np.array(sample, dtype=object) #store sample experience
        
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

#Normal noise process (from https://github.com/mimoralea/gdrl)
class NormalNoiseProcess():
    def __init__(self, exploration_noise_ratio=0.1):
        self.noise_ratio = exploration_noise_ratio

    def get_noise(self, size, max_exploration=False):
        return np.random.normal(loc=0, scale=1 if max_exploration else self.noise_ratio, size=size)

#Decaying noise process for exploration (from https://github.com/mimoralea/gdrl)
class NormalNoiseDecayProcess():
    def __init__(self, init_noise_ratio=0.5, min_noise_ratio=0.1, decay_steps=10000):
        self.t = 0
        self.noise_ratio = init_noise_ratio
        self.init_noise_ratio = init_noise_ratio
        self.min_noise_ratio = min_noise_ratio
        self.decay_steps = decay_steps

    def _noise_ratio_update(self):
        noise_ratio = 1 - self.t / self.decay_steps
        noise_ratio = (self.init_noise_ratio - self.min_noise_ratio) * noise_ratio + self.min_noise_ratio
        noise_ratio = np.clip(noise_ratio, self.min_noise_ratio, self.init_noise_ratio)
        self.t += 1
        return noise_ratio

    def get_noise(self, size, max_exploration=False):
        noise = np.random.normal(loc=0, scale=1 if max_exploration else self.noise_ratio, size=size)
        self.noise_ratio = self._noise_ratio_update()
        return noise

#DDPG with PER
class TD3():
    def __init__(self, 
                policy_model_fn = lambda num_obs, bounds: FCDP(num_obs, bounds), #state vars, action bounds -> model
                policy_optimizer_fn = lambda params, lr : optim.Adam(params, lr), #model params, lr -> optimizer
                policy_optimizer_lr = 1e-4, #optimizer learning rate
                policy_max_gradient_norm = None,
                value_model_fn = lambda nS, nA: FCTQV(nS, nA), #state vars, action vars -> model
                value_optimizer_fn = lambda params, lr : optim.Adam(params, lr), #model params, lr -> optimizer
                value_optimizer_lr = 1e-4, #optimizer learning rate
                value_max_gradient_norm = None,
                value_loss_fn = nn.MSELoss(), #input, target -> loss
                exploration_noise_process_fn = lambda: NormalNoiseDecayProcess(), #module with get_noise function size -> noise array (noise in [-1,1])
                target_policy_noise_process_fn = lambda: NormalNoiseProcess(),
                target_policy_noise_clip_ratio = 0.3, 
                replay_buffer_fn = lambda : PrioritizedReplayBuffer(10000),
                tau=0.005, 
                target_update_steps=1
                ):
        self.policy_model_fn = policy_model_fn
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        self.policy_max_gradient_norm = policy_max_gradient_norm
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.value_max_gradient_norm = value_max_gradient_norm
        self.value_loss_fn = value_loss_fn
        self.exploration_noise_process_fn = exploration_noise_process_fn
        self.target_policy_noise_process_fn = target_policy_noise_process_fn
        self.target_policy_noise_clip_ratio = target_policy_noise_clip_ratio
        self.memory_fn = replay_buffer_fn
        self.tau = tau
        self.target_update_steps = target_update_steps

    def _init_model(self, env):
        nS, nA = env.observation_space.shape[0], env.action_space.shape[0]

        #initialize online and target models
        self.online_policy_model = self.policy_model_fn(nS, self.action_bounds)
        self.target_policy_model = self.policy_model_fn(nS, self.action_bounds)
        self.target_policy_model.load_state_dict(self.online_policy_model.state_dict()) #copy online model parameters to target model

        self.online_value_model = self.value_model_fn(nS, nA)
        self.target_value_model = self.value_model_fn(nS, nA)
        self.target_value_model.load_state_dict(self.online_value_model.state_dict()) #copy online model parameters to target model

        #initialize optimizer
        self.policy_optimizer = self.policy_optimizer_fn(self.online_policy_model.parameters(), lr=self.policy_optimizer_lr)
        self.value_optimizer = self.value_optimizer_fn(self.online_value_model.parameters(), lr=self.value_optimizer_lr)

    def _copy_policy_model(self, env):
        copy = self.policy_model_fn(env.observation_space.shape[0], (env.action_space.low, env.action_space.high))
        copy.load_state_dict(self.online_policy_model.state_dict())
        return copy
    
    def _copy_value_model(self, env):
        copy = self.value_model_fn(env.observation_space.shape[0], env.action_space.shape[0])
        copy.load_state_dict(self.online_value_model.state_dict())
        return copy


    def _update_target_networks(self, tau=None):
        tau = tau if tau else self.tau
        for target, online in zip(self.target_policy_model.parameters(), self.online_policy_model.parameters()):
            target_weights = tau*online.data + (1-tau)*target.data
            target.data.copy_(target_weights)

        for target, online in zip(self.target_value_model.parameters(), self.online_value_model.parameters()):
            target_weights = tau*online.data + (1-tau)*target.data
            target.data.copy_(target_weights)

    def _optimize_model(self, batch_size=None, update_policy=True):
        idxs, weights, experiences = self.memory.sample(batch_size)
        weights = self.online_value_model.numpy_float_to_device(weights)
        experiences = self.online_value_model.load(experiences) #numpy to tensor; move to device
        states, actions, rewards, next_states, is_terminals = experiences
    
        #get target action noise
        with torch.no_grad():
            action_max, action_min = self.target_policy_model.action_max, self.target_policy_model.action_min
            a_range = action_max - action_min
            #get noise in [-1,1] and scale to action range
            a_noise = torch.tensor(self.target_policy_noise.get_noise(actions.shape), device=self.target_policy_model.device, dtype=torch.float32) * a_range
            n_min = action_min * self.target_policy_noise_clip_ratio
            n_max = action_max * self.target_policy_noise_clip_ratio
            a_noise = torch.clip(a_noise, n_min, n_max) #clip noise according to clip ratio

            argmax_a_q_sp = self.target_policy_model(next_states) #select best action of next state according to target policy network
            noisy_argmax_a_q_sp = argmax_a_q_sp + a_noise #add noise
            noisy_argmax_a_q_sp = torch.clip(noisy_argmax_a_q_sp, action_min, action_max) #clip noisy action to fit action range
            max_a_q_sp_1, max_a_q_sp_2 = self.target_value_model(next_states, argmax_a_q_sp) #get values of next states using target value network
            target_q_sa = rewards + (self.gamma * torch.min(max_a_q_sp_1, max_a_q_sp_2) * (1 - is_terminals)) #calculate q target using minimum of the two value estimates

        q_sa_1, q_sa_2 = self.online_value_model(states, actions) #get predicted q from model for each state, action pair

        #get value loss for each critic - weigh sample losses by importance sampling for bias correction
        #calculate loss between prediction and target
        value_loss = self.value_loss_fn(torch.cat([weights*q_sa_1, weights*q_sa_2]), torch.cat([weights*target_q_sa, weights*target_q_sa]))

        #optimize critic networks
        self.value_optimizer.zero_grad()
        value_loss.backward()
        if self.value_max_gradient_norm:
            torch.nn.utils.clip_grad_norm_(self.online_value_model.parameters(), self.value_max_gradient_norm)
        self.value_optimizer.step()

        #update TD errors
        td_errors = (q_sa_1 - target_q_sa).detach().cpu().numpy()
        self.memory.update(idxs, td_errors)

        #get policy gradient/loss
        if update_policy:
            argmax_a_q_s = self.online_policy_model(states) #select best action of state using online policy network
            max_a_q_s = self.online_value_model.Q1(states, argmax_a_q_s) #get value using online value network
            policy_loss = -max_a_q_s.mean() #policy gradient calculated using "backward" on the *negative* mean of the values

            #optimize actor network
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            if self.policy_max_gradient_norm:
                torch.nn.utils.clip_grad_norm_(self.online_policy_model.parameters(), self.policy_max_gradient_norm)
            self.policy_optimizer.step()

    def get_action(self, state, explore=False):
        with torch.no_grad():
            action = self.online_policy_model.select_action(state)
        if explore:
            noise = self.exploration_noise.get_noise(len(self.action_bounds[0]))
            action = np.clip(action + noise, *self.action_bounds)
        
        return action

    def evaluate(self, env, gamma, seed=None):
        state = env.reset(seed=seed)[0]
        ep_return = 0
        for t in count():
            action = self.get_action(state, explore=False)
            state, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward * gamma**t
            if terminated or truncated:
                return ep_return


    def train(self, env, gamma=1.0, num_episodes=100, batch_size=None, n_warmup_batches = 5, tau=0.005, target_update_steps=2, policy_update_steps=2, save_models=None, seed=None, evaluate=True):
        self.memory = self.memory_fn()
        self.exploration_noise = self.exploration_noise_process_fn()
        self.target_policy_noise = self.target_policy_noise_process_fn()
        if save_models: #list of episodes to save models
                save_models.sort()
        self.gamma = gamma
        self.action_bounds = env.action_space.low, env.action_space.high
        self._init_model(env)

        saved_models = {}
        best_model = None

        i = 0
        episode_returns = np.full(num_episodes, np.NINF)
        for episode in tqdm(range(num_episodes)):
            state = env.reset(seed=seed)[0]
            ep_return = 0
            for t in count():
                i += 1
                action = self.get_action(state, explore=True) #use online model to select action
                next_state, reward, terminated, truncated, _ = env.step(action)
                self.memory.store((state, action, reward, next_state, terminated)) #store experience in replay memory
                
                state = next_state

                if len(self.memory) >= batch_size*n_warmup_batches: #optimize policy model
                    self._optimize_model(batch_size, i % policy_update_steps == 0) #only update policy every d update

                #update target network with tau
                if i % target_update_steps == 0:
                    self._update_target_networks(tau)

                ep_return += reward * gamma**t #add discounted reward to return
                if terminated or truncated:
                    #copy and save model
                    if save_models and len(saved_models) < len(save_models) and episode+1 == save_models[len(saved_models)]:
                        saved_models[episode+1] = self._copy_policy_model(env)

                    #save best model
                    if evaluate:
                        ep_return = self.evaluate(env, gamma, seed) #evaluate current policy with no noise
                    if ep_return >= episode_returns.max():
                        best_model = self._copy_policy_model(env)
                    episode_returns[episode] = ep_return
                    break

        
        return episode_returns, best_model, saved_models
    
if __name__ == '__main__':
    import gymnasium as gym
    from gymnasium.wrappers.time_limit import TimeLimit
    #Pendulum
    # td3 = TD3(policy_model_fn = lambda num_obs, bounds: FCDP(num_obs, bounds, hidden_dims=(512, 128), device=torch.device("cuda")), 
    #           policy_optimizer_lr = 0.0005,
    #           value_model_fn = lambda num_obs, nA: FCTQV(num_obs, nA, hidden_dims=(512, 128), device=torch.device("cuda")),
    #           value_optimizer_lr = 0.0005,
    #           replay_buffer_fn = lambda : PrioritizedReplayBuffer(alpha=0.0, beta0=0.0, beta_rate=1.0)) #no PER: alpha=0.0, beta0=0.0, beta_rate=1.0

    # env = gym.make("Pendulum-v1")
    # episode_returns, best_model, saved_models = td3.train(env, gamma=0.99, num_episodes=500, tau=0.005, batch_size=512, save_models=[1, 10, 50, 100, 250, 500])
    # results = {'episode_returns': episode_returns, 'best_model': best_model, 'saved_models': saved_models}
    # import pickle
    # with open('testfiles/td3_pendulum.results', 'wb') as file:
    #     pickle.dump(results, file)

    #Lunar Lander
    ddpg = TD3(policy_model_fn = lambda num_obs, bounds: FCDP(num_obs, bounds, hidden_dims=(512, 128), device=torch.device("cuda")), 
               policy_optimizer_lr = 0.0001,
               value_model_fn = lambda num_obs, nA: FCTQV(num_obs, nA, hidden_dims=(512, 128), device=torch.device("cuda")),
               value_optimizer_lr = 0.0002,
               replay_buffer_fn = lambda : PrioritizedReplayBuffer(alpha=0.0, beta0=0.0, beta_rate=1.0),
               exploration_noise_process_fn = lambda: NormalNoiseDecayProcess(init_noise_ratio=0.99, decay_steps=25000, min_noise_ratio=0.1)) #no PER: alpha=0.0, beta0=0.0, beta_rate=1.0
    env = TimeLimit(gym.make('LunarLander-v2', continuous=True), max_episode_steps=1000)
    #env = gym.make('LunarLander-v2', continuous=True)
    episode_returns, best_model, saved_models = ddpg.train(env, gamma=0.95, num_episodes=500, tau=0.005, batch_size=128, save_models=[1, 10, 50, 100, 250, 500])
    print(episode_returns.max())
    print(episode_returns[-50:])
    results = {'episode_returns': episode_returns, 'best_model': best_model, 'saved_models': saved_models}
    import pickle
    with open('testfiles/td3_lunarlander_limit.results', 'wb') as file:
       pickle.dump(results, file)