import numpy as np
from itertools import count
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
#Code implementations derived from https://github.com/mimoralea/gdrl

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
    
class VPG():
    def __init__(self, 
                policy_model_fn = lambda num_obs, nA: FCDAP(num_obs, nA), #state vars, nA -> model
                policy_optimizer_fn = lambda params, lr : optim.Adam(params, lr), #model params, lr -> optimizer
                policy_optimizer_lr = 1e-4, #optimizer learning rate
                value_model_fn = lambda num_obs: FCV(num_obs), #state vars  -> model
                value_optimizer_fn = lambda params, lr : optim.Adam(params, lr), #model params, lr -> optimizer
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
    
    def _optimize_model(self, rewards, values, log_probs, entropies):
        T = len(rewards)
        #calculate returns G_t(tau)
        discounts = np.logspace(0, T, num=T, base=self.gamma, endpoint=False)
        returns = torch.FloatTensor([np.sum(discounts[:T-t] * rewards[t:]) for t in range(T)]).unsqueeze(1)
        log_probs = torch.cat(log_probs)
        entropies = torch.cat(entropies)
        values = torch.cat(values)

        #calculate "policy loss" as the negative policy gradient with weighted entropy
        advantages = returns - values #use advantage estimates (A_t = G_t-V(S_t)) instead of returns for policy gradient
        policy_grad = (advantages.detach() * log_probs).mean()
        policy_loss = -(policy_grad + self.entropy_weight*entropies.mean())

        #optimize policy network (gradient descent)
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        #optimize value network
        value_loss = advantages.pow(2).mul(0.5).mean() #mean square error
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def train(self, env, gamma=1.0, num_episodes=100, policy_lr=None, value_lr=None, save_models=None, seed=None):
        if save_models: #list of episodes to save models
                save_models.sort()
        self.gamma = gamma
        self._init_model(env, policy_lr, value_lr)

        saved_models = {}
        best_model = None

        i = 0
        episode_returns = np.zeros(num_episodes)
        for episode in tqdm(range(num_episodes)):
            state = env.reset(seed=seed)[0]
            ep_return = 0
            log_probs, rewards, values, entropies = [], [], [], []
            for t in count():
                i += 1
                action, log_prob, entropy = self.policy_model.select_action(state) #select action and get corresponding log prob and dist entropy
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                #gather log probs, rewards, and entropies to calculate policy gradient
                log_probs.append(log_prob)
                rewards.append(reward)
                entropies.append(entropy)
                #get estimated value of current state
                values.append(self.value_model(state))

                state = next_state

                ep_return += reward * gamma**t #add discounted reward to return
                if terminated or truncated:
                    #save best model
                    if ep_return >= episode_returns.max():
                        copy = self.policy_model_fn(len(env.observation_space.sample()), env.action_space.n)
                        copy.load_state_dict(self.policy_model.state_dict())
                        best_model = copy
                    #copy and save model
                    if save_models and len(saved_models) < len(save_models) and episode+1 == save_models[len(saved_models)]:
                        copy = self.policy_model_fn(len(env.observation_space.sample()), env.action_space.n)
                        copy.load_state_dict(self.policy_model.state_dict())
                        saved_models[episode+1] = copy

                    episode_returns[episode] = ep_return
                    break
            #apply gradient optimization at end of each episode
            self._optimize_model(rewards, values, log_probs, entropies)
        
        return episode_returns, best_model, saved_models

if __name__ == '__main__':
    import time
    import gymnasium as gym
    from gymnasium.wrappers.time_limit import TimeLimit
    print('starting')
    env = TimeLimit(gym.make('LunarLander-v2'), max_episode_steps=5000)

    vpg = VPG(policy_model_fn= lambda num_obs, nA: FCDAP(num_obs, nA, hidden_dims=(128, 128)),
          value_model_fn=lambda num_obs: FCV(num_obs, hidden_dims=(128, 128)),
          entropy_weight=0.0003)

    episode_returns, best_model, saved_models = vpg.train(env, gamma=0.95, num_episodes=8000, policy_lr=2e-4, value_lr=5e-4)
    results = {'episode_returns': episode_returns, 'best_model': best_model, 'saved_models': saved_models}
    print('Saving results...')
    import pickle
    with open('testfiles/vpg_lunarlander.results', 'wb') as file:
        pickle.dump(results, file)
