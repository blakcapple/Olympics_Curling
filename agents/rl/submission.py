
import torch.nn as nn
import torch 
import os
from torch.distributions import Categorical, Normal
import numpy as np
from gym.spaces import Box, Discrete

###
def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNNLayer(nn.Module):
    
    out_channel = 32
    hidden_size = 256
    kernel_size = 3
    stride = 1
    use_Relu = True
    use_orthogonal = True
    
    def __init__(self, state_shape):
        
        super().__init__()
        
        active_func = [nn.Tanh(), nn.ReLU()][self.use_Relu]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self.use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][self.use_Relu])
        input_channel = state_shape[0]
        input_width = state_shape[1]
        input_height = state_shape[2]
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        cnn1_out_shape = [input_width - self.kernel_size + self.stride, input_height - self.kernel_size + self.stride]
        pool_out_shape = [int((cnn1_out_shape[0] - 2)/2) + 1, int((cnn1_out_shape[0] - 2)/2) + 1 ]
        cnn2_out_shape = [pool_out_shape[0] - self.kernel_size + self.stride, pool_out_shape[1] - self.kernel_size + self.stride]
        cnn_out_size = cnn2_out_shape[0] * cnn2_out_shape[1] * self.out_channel
        pool = nn.AvgPool2d(kernel_size=2)
        self.cnn = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channel,
                            out_channels=self.out_channel,
                            kernel_size=self.kernel_size,
                            stride=self.stride)
                  ),
            pool,
            init_(nn.Conv2d(in_channels=self.out_channel,
                            out_channels=self.out_channel,
                            kernel_size=self.kernel_size,
                            stride=self.stride)),
            active_func,
            Flatten(),
                            )
        with torch.no_grad():
            dummy_ob = torch.ones(1, input_channel, input_width, input_height).float()
            n_flatten = self.cnn(dummy_ob).shape[1] # 6400
        self.linear = nn.Sequential(init_(nn.Linear(n_flatten, self.hidden_size)), nn.ReLU())
    def forward(self, input):
        cnn_output = self.cnn(input)
        output = self.linear(cnn_output)
        return output
    
###
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
        self.linear_layer = mlp([256]+[256]+[act_dim], activation, output_activation=nn.Tanh)
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
        torch.save(self.state_dict(), pth, _use_new_zipfile_serialization=False)

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
        torch.save(self.state_dict(), pth, _use_new_zipfile_serialization=False)

    def load_model(self, pth):
        self.load_state_dict(torch.load(pth))

    def eval(self, obs):
        """
        return the best action
        """
        logits = self.logits_net(obs).view(-1)
        return torch.argmax(logits).item()
####
class RLAgent:

    def __init__(self, state_shape, action_space):
        
        if isinstance(action_space, Box):
            self.is_act_continuous = True
            self.actor = CNNGaussianActor(state_shape, action_space.shape[0], nn.ReLU)
            self.action_space = action_space
        elif isinstance(action_space, Discrete):
            self.is_act_continuous = False
            self.actor = CNNCategoricalActor(state_shape, action_space.n, nn.ReLU)
            if action_space.n == 36:
                self.actor = CNNCategoricalActor(state_shape, 35, nn.ReLU)
            num = action_space.n
            #dicretise action space
            forces = np.linspace(-100, 200, num=int(np.sqrt(num)), endpoint=True)
            thetas = np.linspace(-30, 30, num=int(np.sqrt(num)), endpoint=True)
            actions = [[force, theta] for force in forces for theta in thetas]
            actions_map = {i:actions[i] for i in range(num)}
            self.actions_map = actions_map
        self.obs_sequence = None

    def choose_action(self, obs, deterministic=False):
        state = torch.from_numpy(obs).float() # [1, 25, 25]
        if self.obs_sequence is not None:
            self.obs_sequence = torch.cat((self.obs_sequence[1:, :, :], state), dim=0) #[4, 25, 25]
        else:
            self.obs_sequence = state.repeat(4, 1, 1) # [4, 25, 25]
        obs_sequence  = self.obs_sequence.unsqueeze(0)
        if self.is_act_continuous:
            if deterministic:
                a_raw = self.actor.mu_net(obs_sequence)
            else:
                pi, _ = self.actor(obs_sequence)
                a_raw = pi.sample()
        else:
            if deterministic:
                logits = self.actor.logits_net(obs_sequence)
                a_raw = torch.argmax(logits)
            else:
                pi, _ = self.actor(obs_sequence)
                a_raw = pi.sample()

        return a_raw

    def load_model(self, pth):

        self.actor.load_model(pth)
    
    def save_model(self, pth):

        self.actor.save_model(pth)

state_shape = [4, 30, 30]
action_num = 49
continue_space = Box(low=np.array([-100, -10]), high=np.array([200, 10]))   
discrete_space = Discrete(action_num)
load_pth = os.path.dirname(os.path.abspath(__file__)) + "/actor_400.pth"
agent = RLAgent(state_shape, discrete_space)
agent.load_model(load_pth)
# agent.save_model(load_pth)


def my_controller(observation_list, action_space_list, is_act_continuous):
    obs = observation_list['obs'].copy()
    obs = np.array(obs)
    actions_raw = agent.choose_action(obs, True)
    if agent.is_act_continuous:
        actions_raw = actions_raw.detach().cpu().numpy().reshape(-1)
        action = np.clip(actions_raw, -1, 1)
        high = agent.action_space.high
        low = agent.action_space.low
        actions = low + 0.5*(action + 1.0)*(high - low)
    else:
        actions = agent.actions_map[actions_raw.item()]
    wrapped_actions = [[actions[0]], [actions[1]]]
    return wrapped_actions

