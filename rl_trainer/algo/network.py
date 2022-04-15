import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal
from rl_trainer.algo.cnn import CNNLayer
import numpy as np 
from gym.spaces import Box, Discrete


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class CNNGaussianActor(nn.Module):
    
    def __init__(self, input_shape, act_dim, activation):
        super().__init__()
        self.input_shape = input_shape
        self.act_dim = act_dim 
        self.cnn_layer = CNNLayer(input_shape)
        self.linear_layer = mlp([64]+[256]+[act_dim], activation, output_activation=nn.Tanh)
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = nn.Sequential(self.cnn_layer, self.linear_layer)
        
    def distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self.distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self.log_prob_from_distribution(pi, act)
        return pi, logp_a

    def save_model(self, pth):
        torch.save(self.state_dict(), pth)

    def load_model(self, pth):
        self.load_state_dict(torch.load(pth))

    def eval(self, obs):
        """
        return the best action
        """
        mu = self.mu_net(obs).view(-1)
        return mu.detach().cpu().numpy()

class CNNCategoricalActor(nn.Module):

    def __init__(self, input_shape, act_dim, activation):
        super().__init__()
        self.input_shape = input_shape
        self.act_dim = act_dim 
        # self.cnn_layer = CNNLayer(input_shape)
        # self.extra_layer = mlp([info_dim]+[64], activation) 
        self.linear_layer = mlp([input_shape]+[256]+[256]+[act_dim], activation)
        # self.linear_layer = mlp([256]+[256]+[act_dim], activation)

        
    def distribution(self, obs):
        # cnn_out = self.cnn_layer(obs)
        # extra_out = self.extra_layer(info)
        # full_out = torch.cat([cnn_out, extra_out], dim=1)
        logits = self.linear_layer(obs)

        return Categorical(logits=logits)

    def log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self.distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self.log_prob_from_distribution(pi, act.view(-1))
        return pi, logp_a

    def save_model(self, pth):
        torch.save(self.state_dict(), pth)

    def load_model(self, pth):
        self.load_state_dict(torch.load(pth))

    def eval(self, obs):
        """
        return the best action
        """
        # cnn_out = self.cnn_layer(obs)
        # extra_out = self.extra_layer(info)
        # full_out = torch.cat([cnn_out, extra_out], dim=1)
        logits = self.linear_layer(obs)

        return torch.argmax(logits).item()
        

class CNNCritic(nn.Module):

    def __init__(self, input_shape, activation):
        super().__init__()
        self.input_shape = input_shape
        # self.cnn_layer = CNNLayer(input_shape)
        # self.extra_layer = mlp([info_dim]+[64], activation) 
        self.linear_layer = mlp([input_shape]+[256+64]+[256]+[1], activation)
        # self.linear_layer = mlp([256]+[256]+[1], activation)

    def forward(self, obs):
        
        # cnn_out = self.cnn_layer(obs)
        # extra_out = self.extra_layer(info)
        # full_out = torch.cat([cnn_out, extra_out], dim=1)
        v = self.linear_layer(obs)

        return torch.squeeze(v, -1)

    def save_model(self, pth):
        torch.save(self.state_dict(), pth)

    def load_model(self, pth):
        self.load_state_dict(torch.load(pth))

class CNNActorCritic(nn.Module):
    
    def __init__(self, state_shape, action_space, activation=nn.ReLU):
        super().__init__()

        if isinstance(action_space, Box):
            self.pi = CNNGaussianActor(state_shape, action_space.shape[0], activation)
        elif isinstance(action_space, Discrete):
            self.pi = CNNCategoricalActor(state_shape, action_space.n, activation)
        self.v = CNNCritic(state_shape, activation)
    
    def step(self, obs):
        with torch.no_grad():
            pi = self.pi.distribution(obs)
            a = pi.sample()
            logp_a = self.pi.log_prob_from_distribution(pi, a)
            v = self.v(obs)

        return a.detach().cpu().numpy(), v.detach().cpu().numpy(), logp_a.detach().cpu().numpy()

    def act(self, obs, phase='train'):
        if phase == 'test':
            return self.pi.eval(obs)
        elif phase == 'train':
            return self.step(obs)[0]
        else:
            raise NotImplementedError

class Discrimator(nn.Module):
    """
    This module is for GAIL
    """
    def __init__(self, obs_shape, info_dim, action_dim, hidden_size=(128, 128), activation='tanh'):
        super().__init__()

        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.obs_shape = obs_shape
        self.cnn_layer = CNNLayer(self.obs_shape)
        self.extra_layer = mlp([info_dim]+[64], self.activation, output_activation=torch.relu) 
        self.linear_layer = mlp([256+64+action_dim]+hidden_size+[1], self.activation)

    def forward(self, obs, info, action):
        
        cnn_out = self.cnn_layer(obs)
        extra_out = self.extra_layer(info)
        full_out = torch.cat([cnn_out, extra_out, action], dim=1)
        d = torch.sigmoid(self.linear_layer(full_out))

        return torch.squeeze(d, -1)