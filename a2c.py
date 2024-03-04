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

#Code implementations heavily influenced by https://github.com/mimoralea/gdrl

#fully connected actor critic neural network
class FCAC(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dim,
                 hidden_dims=(32,32), #define hidden layers as tuple where each element is an int representing # of neurons at a layer
                 activation_fc=nn.ReLU,
                 device = torch.device("cpu")
                 ):
        super(FCAC, self).__init__()
        self.activation_fc = activation_fc

        policy_hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            policy_hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            policy_hidden_layers.append(activation_fc())

        value_hidden_layers = nn.ModuleList()
        for i in range(len(hidden_dims)-1):
            value_hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            value_hidden_layers.append(activation_fc())
        
        self.policy_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            activation_fc(),
            *policy_hidden_layers,
            nn.Linear(hidden_dims[-1], output_dim)
        )

        self.value_model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            activation_fc(),
            *value_hidden_layers,
            nn.Linear(hidden_dims[-1], 1)
        )

        self.device = device

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, 
                             dtype=torch.float32).to(self.device)
           #x = x.unsqueeze(0)
        return x
        
    def forward(self, state):
        x = self._format(state)
        return self.policy_model(x), self.value_model(x)
    
    #select and return action, corresponding log prob of the action, entropy of the distribution, and value of state
    def select_action(self, state):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def evaluate_state(self, state):
        x = self._format(state)
        return self.value_model(x)
        
class A2C():
    def __init__(self, 
                actor_critic_model_fn = lambda num_obs, nA: FCAC(num_obs, nA),
                policy_optimizer_fn = lambda params, lr : optim.Adam(params, lr), #model params, lr -> optimizer
                policy_optimizer_lr = 1e-4, #optimizer learning rate
                value_optimizer_fn = lambda params, lr : optim.Adam(params, lr), #model params, lr -> optimizer
                value_optimizer_lr = 1e-4, #optimizer learning rate
                entropy_weight = 1e-4
                ):
        self.actor_critic_model_fn = actor_critic_model_fn
        self.policy_optimizer_fn = policy_optimizer_fn
        self.policy_optimizer_lr = policy_optimizer_lr
        self.value_optimizer_fn = value_optimizer_fn
        self.value_optimizer_lr = value_optimizer_lr
        self.entropy_weight = entropy_weight

    def _init_model(self, envs, policy_lr=None, value_lr=None, device=torch.device("cpu")):
        if not policy_lr:
            policy_lr = self.policy_optimizer_lr
        if not value_lr:
            value_lr = self.value_optimizer_lr

        self.actor_critic_model = self.actor_critic_model_fn(len(envs.observation_space.sample()[0]), envs.action_space[0].n)
        self.actor_critic_model.device = device
        self.actor_critic_model.to(device)
        self.policy_optimizer = self.policy_optimizer_fn(self.actor_critic_model.policy_model.parameters(), lr=policy_lr)
        self.value_optimizer = self.value_optimizer_fn(self.actor_critic_model.value_model.parameters(), lr=value_lr)

    def _optimize_model(self, rewards, values, log_probs, entropies, device):
        T = len(rewards) #num of n-steps
        n_envs = len(rewards[0])

        #reformat tensors/arrays
        log_probs = torch.stack(log_probs).squeeze() #tensor of shape [n_steps, n_envs]
        entropies = torch.stack(entropies).squeeze()
        values = torch.stack(values).squeeze()
        rewards = np.array(rewards).squeeze() #np array of shape [n_steps, n_envs]
        
        #Calculate n_step returns
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = np.array([[np.sum(discounts[:T-t] * rewards[t:, w]) for t in range(T)] 
                             for w in range(n_envs)])
        
        values_array = values.data.cpu().numpy()
        #gae discount = (gamma*lambda)^t from t=0 to t=T
        discounts = np.logspace(0, T-1, num=T-1, base=self.gamma*self.lmbda, endpoint=False)
        #calculate A^k advantage estimators (deltas) for all steps
        advantages = rewards[:-1] + self.gamma*values_array[1:] - values_array[:-1]
        #calculate GAE for all steps
        gaes = np.array([[np.sum(discounts[:T-1-t] * advantages[t:, w]) for t in range(T-1)] 
                             for w in range(n_envs)])

        log_probs = log_probs.view(-1).unsqueeze(1).to(device)
        entropies = entropies.view(-1).unsqueeze(1).to(device)
        entropies = entropies.view(-1).unsqueeze(1).to(device)
        returns = torch.FloatTensor(returns.T[:-1]).reshape(-1).unsqueeze(1).to(device)
        gaes = torch.FloatTensor(gaes.T).reshape(-1).unsqueeze(1).to(device)

        #calculate "policy loss" as the negative policy gradient with weighted entropy
        policy_grad = (gaes.detach() * log_probs).mean()
        policy_loss = -(policy_grad + self.entropy_weight*entropies.mean())

        #get new policy gradient from loss and step
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic_model.policy_model.parameters(), 
                                       self.policy_model_max_grad_norm)
        self.policy_optimizer.step()

        #get new value gradient from loss and step
        values = values[:-1,:].view(-1).unsqueeze(1).to(device)

        value_loss = (returns.detach()-values).pow(2).mul(0.5).mean() #mean square error
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic_model.value_model.parameters(), 
                                       self.value_model_max_grad_norm)
        self.value_optimizer.step()

    def train(self, envs, max_episodes=500, goal=(float('inf'), 100), max_n_steps=5, gamma=1.0, lmbda=1.0, entropy_weight=None, 
              policy_lr=None, value_lr=None, policy_model_max_grad_norm=1, value_model_max_grad_norm=float('inf'), save_models=None, cuda=False):
        self.gamma = gamma
        self.lmbda = lmbda
        if not entropy_weight:
            entropy_weight = self.entropy_weight
        self.policy_model_max_grad_norm = policy_model_max_grad_norm
        self.value_model_max_grad_norm = value_model_max_grad_norm

        device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
        
        self._init_model(envs, policy_lr, value_lr, device)
        self.stats = {'episode_returns' : [[] for _ in range(envs.num_envs)],
                      'best_moving_avg_return' : 0.0,
                      'best_model' : self.actor_critic_model_fn(len(envs.observation_space.sample()[0]), envs.action_space[0].n),
                      'saved_models' : {ep : self.actor_critic_model_fn(len(envs.observation_space.sample()[0]), envs.action_space[0].n) for ep in save_models}}
        
        #training loop
        state = envs.reset()[0]
        ep_return = np.array([0.0 for _ in range(envs.num_envs)])
        ep_number = np.array([0 for _ in range(envs.num_envs)])
        while np.min(ep_number) < max_episodes and self.stats['best_moving_avg_return'] < goal[0]:
            log_probs, rewards, values, entropies, is_terminated = [], [], [], [], []
            #gather data for n_step td
            t = 0
            while t < max_n_steps:
                #select action and get corresponding log prob and dist entropy/get value for current state
                action, log_prob, entropy, value = self.actor_critic_model.select_action(torch.Tensor(state).to(device))
                next_state, reward, terminated, truncated, _ = envs.step(action.cpu().numpy())
                #gather log probs, rewards, and entropies to calculate policy gradient/values for value gradient
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)
                values.append(value)
                #store is_terminated for td/gae calculations
                is_terminated.append(terminated)
                
                state = next_state

                ep_return += reward * self.gamma**t
                done = terminated | truncated
                if done.any():
                    for idx, d in enumerate(done):
                        if d:
                            self.stats['episode_returns'][idx].append(ep_return[idx]) #store episode return for stats
                            #get avg return of last x episodes
                            moving_avg_return = np.mean(self.stats['episode_returns'][idx][max(0, ep_number[idx]-goal[1]):])
                            #compare and save best model across all workers
                            if moving_avg_return > self.stats['best_moving_avg_return']:
                                self.stats['best_model'].load_state_dict(self.actor_critic_model.state_dict())
                                self.stats['best_moving_avg_return'] = moving_avg_return

                            #reset environment/return, increment episode counter
                            ep_return[idx] = 0
                            ep_number[idx] += 1

                            #save models at specific episodes (using first vectorized environment)
                            saved_models = self.stats['saved_models']
                            if idx==0 and saved_models.get(ep_number[idx]):
                                saved_models[ep_number[idx]].load_state_dict(self.actor_critic_model.state_dict())
                    break
                
                t += 1 #current td-step counter

            #bootstrap with predicted value of next state for n-step td estimate ("critic")
            R = self.actor_critic_model.evaluate_state(torch.Tensor(next_state).to(device)).detach().cpu().numpy().ravel() * (1-terminated)
            rewards.append(R)
            values.append(torch.FloatTensor(R).unsqueeze(-1).to(device))
            self._optimize_model(rewards, values, log_probs, entropies, device)

        #self.stats['episode_returns'] = self.stats['episode_returns'][:, :self.stats['max_episode']+1]
        return self.stats
        
if __name__ == '__main__':
    import time
    envs = gym.vector.make("CartPole-v1", num_envs=8)

    a2c = A2C(actor_critic_model_fn = lambda num_obs, nA: FCAC(num_obs, nA, hidden_dims=(512, 128)))
    start_time = time.time()
    #note: for "true" GAE, set max_n_steps to inf; for regular n-step td, set max_n_steps to an int and lmbda to 1
    results = a2c.train(envs, max_episodes=1000, goal=(500, 50), max_n_steps=50, lmbda=0.97, policy_lr=1e-4, value_lr=5e-4, 
                        entropy_weight=1e-3, save_models=[1,100,250,500,750,1000], cuda=True)
    elapsed = time.time() - start_time
    envs.close()
    print(f'Elapsed time: {int(elapsed/60)} min {elapsed % 60} sec')
    print('Saving results...')
    import pickle
    with open('a2c.results', 'wb') as file:
        pickle.dump(results, file)
    print(f"Best moving avg return: {results['best_moving_avg_return']}")
