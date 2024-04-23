import numpy as np
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

'''
NOTE: if param sharing (generally if agents have the same observation and action space), include agent ID as part of observation (one-hot encoded)
    https://arxiv.org/pdf/2005.13625.pdf

    if different obs space, zero pad obs to the max?
    if different action space, zero pad input action, clip output action vector for each individual agent
'''
    
#Deep Q Recurrent Neural Network
class MultiAgentDQRNN(nn.Module):

    def __init__(self, obs_dim, ):
        pass
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
