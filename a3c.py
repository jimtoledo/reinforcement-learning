import gymnasium as gym
from gymnasium.wrappers.time_limit import TimeLimit
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

#Code implementations heavily influenced by https://github.com/mimoralea/gdrl
#A3C algorithm described in this paper: https://arxiv.org/pdf/1602.01783.pdf with additional max_episodes parameter for balanced training statistics

class FCDAP(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hidden_dims=(32,32), #define hidden layers as tuple where each element is an int representing # of neurons at a layer
                 activation_fc=nn.ReLU):
        super(FCDAP, self).__init__()
        self.activation_fc = activation_fc

        hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            hidden_layers.append(activation_fc())
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            activation_fc(),
            *hidden_layers,
            nn.Linear(hidden_dims[-1], output_dim)
        )

        device = "cpu"
        #if torch.cuda.is_available():
        #    device = "cuda"
        self.device = torch.device(device)
        self.to(self.device)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, 
                             dtype=torch.float32)
            x = x.unsqueeze(0)
        return x
        
    def forward(self, state):
        x = self._format(state)
        return self.layers(x)

    #select and return action, corresponding log prob of the action, and entropy of the distribution
    def select_action(self, state):
        logits = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).unsqueeze(-1), dist.entropy().unsqueeze(-1)

#Fully-connected value network (state observation -> state value)
class FCV(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dims=(32,32), #define hidden layers as tuple where each element is an int representing # of neurons at a layer
                 activation_fc=nn.ReLU):
        super(FCV, self).__init__()

        hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            hidden_layers.append(activation_fc())
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            activation_fc(),
            *hidden_layers,
            nn.Linear(hidden_dims[-1], 1)
        )

        device = "cpu"
        #if torch.cuda.is_available():
        #    device = "cuda"
        self.device = torch.device(device)
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
        return self.layers(x)

#Shared optimizer code directly taken from https://github.com/mimoralea/gdrl
class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super(SharedAdam, self).__init__(
            params, lr=lr, betas=betas, eps=eps, 
            weight_decay=weight_decay, amsgrad=amsgrad)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['shared_step'] = torch.zeros(1).share_memory_()
                state['exp_avg'] = torch.zeros_like(p.data).share_memory_()
                state['exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()
                if weight_decay:
                    state['weight_decay'] = torch.zeros_like(p.data).share_memory_()
                if amsgrad:
                    state['max_exp_avg_sq'] = torch.zeros_like(p.data).share_memory_()

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                self.state[p]['steps'] = self.state[p]['shared_step'].item()
                self.state[p]['shared_step'] += 1
        super().step(closure)

class A3CWorker(mp.Process):
    def __init__(self, rank, make_env_fn, policy_model_fn, value_model_fn, shared_policy_model, shared_value_model, shared_policy_optimizer, shared_value_optimizer, 
                 max_episodes, shared_T, max_T, goal, stats, policy_model_max_grad_norm, value_model_max_grad_norm, max_td_steps=5, gamma=1.0, entropy_weight=1e-4, save_models=None):
        super(A3CWorker, self).__init__()
        self.rank = rank
        self.env = make_env_fn()
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.shared_policy_model = shared_policy_model
        self.shared_value_model = shared_value_model
        self.shared_policy_optimizer = shared_policy_optimizer
        self.shared_value_optimizer = shared_value_optimizer
        self.policy_model_max_grad_norm = policy_model_max_grad_norm
        self.value_model_max_grad_norm = value_model_max_grad_norm
        self.max_td_steps = max_td_steps
        self.max_episodes = max_episodes
        self.T = shared_T
        self.max_T = max_T
        self.goal = goal
        self.stats = stats
        #initialize local models
        self.local_policy_model = policy_model_fn(len(self.env.observation_space.sample()), self.env.action_space.n)
        self.local_value_model = value_model_fn(len(self.env.observation_space.sample()))
        print(f'Worker {self.rank} created...')
    
    def _optimize_model(self, rewards, values, log_probs, entropies):
        T = len(rewards)
        #Calculate n_step returns
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([np.sum(discounts[:T-t] * rewards[t:]) for t in range(T)])
        #drop return of final td step next_state
        discounts = torch.FloatTensor(discounts[:-1]).unsqueeze(1)
        returns = torch.FloatTensor(returns[:-1]).unsqueeze(1)
        
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)
        values = torch.cat(values)

        #calculate "policy loss" as the negative policy gradient with weighted entropy
        advantages = returns - values #use advantage estimates (A_t = G_t-V(S_t)) instead of returns for policy gradient
        policy_grad = (advantages.detach() * log_probs).mean()
        policy_loss = -(policy_grad + self.entropy_weight*entropies.mean())

        #get new policy gradient from loss
        self.shared_policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_policy_model.parameters(), 
                                       self.policy_model_max_grad_norm)
        
        #transfer gradients from local model to shared model and step
        for param, shared_param in zip(self.local_policy_model.parameters(), 
                                       self.shared_policy_model.parameters()):
            shared_param.grad = param.grad
        self.shared_policy_optimizer.step()
        #load updated shared model back into local model
        self.local_policy_model.load_state_dict(self.shared_policy_model.state_dict())

        #get new value gradient from loss
        value_loss = advantages.pow(2).mul(0.5).mean() #mean square error
        self.shared_value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_value_model.parameters(), 
                                       self.value_model_max_grad_norm)
        
        #transfer gradients from local model to shared model and step
        for param, shared_param in zip(self.local_value_model.parameters(), 
                                       self.shared_value_model.parameters()):
            shared_param.grad = param.grad
        self.shared_value_optimizer.step()
        #load updated shared model back into local model
        self.local_value_model.load_state_dict(self.shared_value_model.state_dict())


    def run(self):
        #copy current shared model parameters
        self.local_policy_model.load_state_dict(self.shared_policy_model.state_dict())
        self.local_value_model.load_state_dict(self.shared_value_model.state_dict())
        print(f'Worker {self.rank} started')
        torch.seed() ; np.random.seed() ; random.seed() #reset seeds
        state = self.env.reset()[0]
        ep_return = 0
        ep_number = 0
        '''
        Stop conditions:
        - Max iterations across all workers exceeded (bounded by T)
        - Max episodes exceeded in current worker
        - Best moving average of last `goal[1]` episodes (across all workers) matches/exceeds `goal[0]`
        '''
        while self.T < self.max_T and ep_number < self.max_episodes and self.stats['best_moving_avg_return'][0] < self.goal[0]:
            log_probs, rewards, values, entropies = [], [], [], []
            #gather data for n_step td
            t = 0
            for _ in range(self.max_td_steps):
                action, log_prob, entropy = self.local_policy_model.select_action(state) #select action and get corresponding log prob and dist entropy
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                #gather log probs, rewards, and entropies to calculate policy gradient
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)
                #get estimated value of current state
                values.append(self.local_value_model(state))

                ep_return += reward * self.gamma**t
                if terminated or truncated:
                    self.stats['episode_returns'][self.rank, ep_number] = ep_return #store episode return for stats
                    #get avg return of last 100 episodes
                    moving_avg_return = self.stats['episode_returns'][self.rank, max(0, ep_number-self.goal[1]):ep_number+1].mean()
                    #compare and save best model across all workers
                    if moving_avg_return > self.stats['best_moving_avg_return'][0]:
                        self.stats['best_model'].load_state_dict(self.local_policy_model.state_dict())
                        self.stats['best_moving_avg_return'][0] = moving_avg_return

                    #reset environment/return, increment episode counter
                    state = self.env.reset()[0]
                    ep_return = 0
                    ep_number += 1

                    #save models at specific episodes for worker 0 only
                    saved_models = self.stats['saved_models']
                    if self.rank == 0 and saved_models.get(ep_number):
                        saved_models[ep_number].load_state_dict(self.local_policy_model.state_dict())

                    
                    break
                else:
                    state = next_state
                
                t += 1 #current td-step counter
                self.T += 1 #global step counter

            #bootstrap with predicted value of next state for n-step td estimate ("critic")
            R = 0 if terminated else self.local_value_model(next_state).detach().item()
            rewards.append(R)
            self._optimize_model(rewards, values, log_probs, entropies)
        
        self.stats['episode_returns'][self.rank, ep_number:] = np.nan
        if ep_number > self.stats['max_episode'][0]:
            self.stats['max_episode'][0] = ep_number
        print(f'Worker {self.rank} done...')
        
class A3C():
    def __init__(self, 
                policy_model_fn = lambda num_obs, nA: FCDAP(num_obs, nA), #state vars, nA -> model
                policy_optimizer_fn = lambda params, lr : SharedAdam(params, lr), #model params, lr -> optimizer
                policy_optimizer_lr = 1e-4, #optimizer learning rate
                value_model_fn = lambda num_obs: FCV(num_obs), #state vars  -> model
                value_optimizer_fn = lambda params, lr : SharedAdam(params, lr), #model params, lr -> optimizer
                value_optimizer_lr = 1e-4, #optimizer learning rate
                entropy_weight = 1e-4
                ):
        self.policy_model_fn = policy_model_fn
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        self.value_model_fn = value_model_fn
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.entropy_weight = entropy_weight

    def _init_model(self, env, policy_lr=None, value_lr=None):
        if not policy_lr:
            policy_lr = self.policy_optimizer_lr
        if not value_lr:
            value_lr = self.value_optimizer_lr

        self.policy_model = self.policy_model_fn(len(env.observation_space.sample()), env.action_space.n)
        self.policy_optimizer = self.policy_optimizer_fn(self.policy_model.parameters(), lr=policy_lr)

        self.value_model = self.value_model_fn(len(env.observation_space.sample()))
        self.value_optimizer = self.value_optimizer_fn(self.value_model.parameters(), lr=value_lr)

    def train(self, make_env_fn, num_workers=4, max_episodes=500, max_T=50000, goal=(float('inf'), 100), max_td_steps=5, gamma=1.0, entropy_weight=None, 
              policy_lr=None, value_lr=None, policy_model_max_grad_norm=1, value_model_max_grad_norm=float('inf'), save_models=None):
        self.gamma = gamma
        if not entropy_weight:
            entropy_weight = self.entropy_weight
        env = make_env_fn()

        self._init_model(env, policy_lr, value_lr)
        self.shared_T = torch.IntTensor([0])
        self.stats = {'episode_returns' : torch.zeros([num_workers, max_episodes], dtype=torch.float),
                      'best_moving_avg_return' : torch.tensor([0.0]),
                      'max_episode' : torch.tensor([0]),
                      'best_model' : self.policy_model_fn(len(env.observation_space.sample()), env.action_space.n),
                      'saved_models' : {ep : self.policy_model_fn(len(env.observation_space.sample()), env.action_space.n) for ep in save_models}}
        #shared models/stats and counter
        self.policy_model.share_memory()
        self.value_model.share_memory()
        self.shared_T.share_memory_()
        self.stats['episode_returns'].share_memory_()
        self.stats['best_moving_avg_return'].share_memory_()
        self.stats['max_episode'].share_memory_()
        self.stats['best_model'].share_memory()
        [model.share_memory() for model in self.stats['saved_models'].values()]

        self.workers = [A3CWorker(rank, make_env_fn, self.policy_model_fn, self.value_model_fn, self.policy_model, self.value_model, self.policy_optimizer, self.value_optimizer, max_episodes, self.shared_T, max_T, 
                                  goal, self.stats, policy_model_max_grad_norm, value_model_max_grad_norm, max_td_steps, gamma, entropy_weight, save_models) for rank in range(num_workers)]
        [w.start() for w in self.workers]
        for w in self.workers:
            w.join()
            print(f'{w} Joined')
        
        self.stats['episode_returns'] = self.stats['episode_returns'][:, :self.stats['max_episode']+1]
        return self.stats
        
if __name__ == '__main__':
    import time
    print('starting')
    #make_env_fn = lambda : gym.make('CartPole-v1')
    make_env_fn = lambda : TimeLimit(gym.make('LunarLander-v2'), max_episode_steps=5000)

    a3c = A3C(policy_model_fn= lambda num_obs, nA: FCDAP(num_obs, nA, hidden_dims=(512, 128)),
          value_model_fn=lambda num_obs: FCV(num_obs, hidden_dims=(512, 128)))
    start_time = time.time()
    #results = a3c.train(make_env_fn, num_workers=6, max_episodes=50000, max_T=float("inf"), goal=(500, 50), max_td_steps=50, policy_lr=1e-4, value_lr=2e-4, entropy_weight=1e-3, save_models=[1,100,250,500,750,1000,5000])
    results = a3c.train(make_env_fn, num_workers=6, max_episodes=50000, max_T=float("inf"), goal=(200, 50), max_td_steps=100, policy_lr=1e-4, value_lr=2e-4, entropy_weight=1e-3, save_models=[1,100,250,500,750,1000,5000])

    elapsed = time.time() - start_time
    print(f'Elapsed time: {int(elapsed/60)} min {elapsed % 60} sec')
    print('Saving results...')
    import pickle
    #with open('a3c.results', 'wb') as file:
    with open('a3c_lunarlander.results', 'wb') as file:
        pickle.dump(results, file)
    print(f"Best moving avg return: {results['best_moving_avg_return'][0]}")
