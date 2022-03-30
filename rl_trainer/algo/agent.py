import random
from gym.spaces import Box, Discrete
from torch.distributions import Categorical, Normal
from rl_trainer.algo.cnn import CNNLayer
import torch.nn as nn
import numpy as np
import torch  

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

class random_agent:
    def __init__(self, action_space, seed=0):

        self.action_space = action_space
        self.seed(seed)

    def seed(self, seed = None):
        random.seed(seed)

    def act(self, obs, deterministic=False):

        if isinstance(self.action_space, Discrete):
            action = random.randint(0, self.action_space.n-1)
        if isinstance(self.action_space, Box):
            action =  np.random.uniform([-1, -1], [1, 1])

        return action


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
        self.cnn_layer = CNNLayer(input_shape)
        self.linear_layer = mlp([256]+[256]+[act_dim], activation)
        self.logits_net = nn.Sequential(self.cnn_layer, self.linear_layer)
        
    def distribution(self, obs):
        logits = self.logits_net(obs)
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
        logits = self.logits_net(obs).view(-1)
        return torch.argmax(logits).item()

class rl_agent:
    
    def __init__(self, state_shape, action_space, device):
        
        if isinstance(action_space, Box):
            self.actor = CNNGaussianActor(state_shape, action_space.shape[0], nn.ReLU).to(device)
        elif isinstance(action_space, Discrete):
            self.actor = CNNCategoricalActor(state_shape, action_space.n, nn.ReLU).to(device)

    def act(self, obs, deterministic=False):

        if deterministic:
            logits = self.actor.logits_net(obs)
            a_raw = torch.argmax(logits)
        else:
            pi, _ = self.actor(obs)
            a_raw = pi.sample()
        
        return a_raw.detach().cpu().numpy().item()

    def load_model(self, pth):

        self.actor.load_model(pth)
