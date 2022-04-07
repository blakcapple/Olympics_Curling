
import torch.nn as nn
import torch 
import os
from torch.distributions import Categorical, Normal
import numpy as np
from gym.spaces import Box, Discrete
import math 
from copy import deepcopy
import pdb

##----------------------------------------------
##---This block is for rule helper----

def calculate_pos(obs, colour, ego_pos=[300, 186]):
    scalar_y = 10.5
    scalar_x = 10.5
    """calculate pos given color"""
    color_pos = [] # relative pos in the obs
    color_group = [[] for _ in range (4)]
    match_number = 5 if colour == 'purple' else 1
    color_point = np.argwhere(obs == match_number)

    if color_point.shape[0] > 0:
        # gruop point
        for x, y in color_point:
            for i, point_group in enumerate(color_group):
                if len(point_group) == 0:
                    color_group[i].append([x,y])
                    break
                else:
                    x_mean = np.mean([point_group[i][0] for i in range(len(point_group))])
                    y_mean = np.mean([point_group[i][1] for i in range(len(point_group))])
                    if np.abs(x-x_mean) + np.abs(y-y_mean) < 4:
                        color_group[i].append([x,y])
                        break 

    for i, point_group in enumerate(color_group):
        if len(point_group) > 0:
            x_mean = np.mean([point_group[i][0] for i in range(len(point_group))])
            y_mean = np.mean([point_group[i][1] for i in range(len(point_group))])
            color_pos.append([x_mean, y_mean])
    relative_pos = color_pos
    real_pos = []
    for pos in relative_pos:
        pos_x = 450 - pos[1]*scalar_x
        pos_y = ego_pos[1]+30*scalar_y - pos[0]*scalar_y
        real_pos.append([pos_x, pos_y])
    return real_pos

##------------------------------------------------------------


##
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
    # constant 
    gamma = 0.98
    delta_t = 0.1
    mass = 1 
    fix_forward_count = 21
    fix_forward_force = 50
    fix_backward_force = -100
    fix_backward_count = 24

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
            thetas = np.linspace(-10, 10, num=int(np.sqrt(num)), endpoint=True)
            actions = [[force, theta] for force in forces for theta in thetas]
            actions_map = {i:actions[i] for i in range(num)}
            self.actions_map = actions_map
        self.obs_sequence = None
        self.throw_left = 4
        self.ep_count = 0

        # physical information 
        self.theta = 90 
        self.pose = [300, 150]
        self.v =  [0, 0]
        self.acc = [0, 0]


        # 对手的位置信息
        self.oppo_pos = []
        # 规则阶段 1表示已经智能体停止，2表示智能体完成转向计算，3表示完成转向
        self.stage = 0
        # 表征是否是最后一个投掷回合（考虑对手的）
        self.last_throw = False
        # 表征是否是第一个投掷回合（考虑对手的）
        self.first_throw = False
        # 智能体的模型集合
        self.model_dict = dict()
        # 要撞击的目标位置
        self.crush_pos = None

    def set_model_dict(self, pth, name='normal'):

        self.model_dict[name] = pth

    def set_game_information(self, score, game_round):

        self.score = score
        self.game_round = game_round

    def reset(self):
    
        self.theta = 90 
        self.pose = [300, 150]
        self.v =  [0, 0]
        self.acc = [0, 0]

        self.ep_count = 0
        self.obs_sequence = None
        self.stage = 0
        self.last_throw = False
        self.first_throw = False
    
    def set_agent_idx(self, idx):

        self.agent_idx = idx 

    def _action_to_accel(self, action):
        """
        Convert action(force) to acceleration
        """
        self.theta = self.theta + action[1]
        force = action[0] / self.mass
        accel_x = force * math.cos(self.theta / 180 * math.pi)
        accel_y = force * math.sin(self.theta / 180 * math.pi)
        accel = [accel_x, accel_y]
        self.acc = accel

    def _update_physics(self):
        x,y = self.pose
        vx, vy = self.v
        accel_x, accel_y = self.acc
        x_new = x + vx * self.delta_t  # update position with t
        y_new = y + vy * self.delta_t
        vx_new = self.gamma * vx + accel_x * self.delta_t  # update v with acceleration
        vy_new = self.gamma * vy + accel_y * self.delta_t
        self.v = [vx_new, vy_new]
        self.pose = [x_new, y_new]

    def step(self, action):
        # 更新加速度，更新角度
        self._action_to_accel(action)
        # 更新智能体速度和位置
        self._update_physics()

    def get_model_action(self, deterministic):
        """
        产生强化学习模型的动作
        """
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
        actions = self.actions_map[a_raw.item()]
        return actions

    def get_rule_action(self, crush_pos):
        """
        得到基于规则的动作
        """
        # 让智能体完成对目标位置的转向
        if self.stage == 1:
            delta = np.array(crush_pos) - np.array(self.pose)
            delta = delta.reshape(-1)
            radius = math.atan2(delta[0], delta[1])
            delta_theta = math.degrees(radius)
            self.goal_theta  = self.theta - delta_theta
            self.stage = 2
        if self.stage == 2:
            if self.theta != self.goal_theta:
                theta = self.goal_theta - self.theta
                theta = np.clip(theta, -30, 30)
                actions  = [0, theta]
            else: self.stage =3
        # 冲向目标位置
        if self.stage == 3:
            actions = [200, 0]
        return actions
    
    def _store_oppo_pos(self, color, obs):
        """
        存储有价值的冰壶位置信息
        """
        crush_pos = None
        opponent_color = 'green' if color=='purple' else 'purple'
        self.oppo_pos = calculate_pos(obs.reshape(30, 30), opponent_color, self.pose)
        self.own_pos = calculate_pos(obs.reshape(30, 30), color, self.pose)
        if len(self.oppo_pos) > 0:
            oppo_temp = deepcopy(self.oppo_pos)
            for pos in self.oppo_pos:
                # 距离场地大于一定的距离则无视
                if np.linalg.norm([pos[0] - 300, pos[1] - 500]) > 105 and pos[1] > 500 or pos[0]<200 or pos[0]> 400:
                    oppo_temp.remove(pos)
                    continue 
                # 继续判断与目标冰球连线上是否存在己方冰球
                if len(self.own_pos) > 0 and len(oppo_temp) > 0:
                    delta = np.array(pos) - np.array(self.pose)
                    delta = delta.reshape(-1)
                    radius_oppo = math.atan2(delta[0], delta[1])
                    for pos2 in self.own_pos:
                        delta = np.array(pos2) - np.array(self.pose)
                        delta = delta.reshape(-1)
                        radius_own = math.atan2(delta[0], delta[1])
                        # 角度在3度之内且自己的冰壶距离中心点更近则移除这个点
                        if np.abs(radius_oppo-radius_own) <= 5/180*np.pi:
                            if np.linalg.norm([pos[0]-300, pos[1]-500]) > np.linalg.norm([pos2[0]-300, pos2[1]-500]):
                                oppo_temp.remove(pos)
            # 选择距离中心点最近的对手作为撞击对象
            if len(oppo_temp) > 1:
                min_dist = np.inf 
                for i, pos in enumerate(oppo_temp):
                    dist = np.linalg.norm([pos[0] - 300, pos[1] - 500])
                    if dist < min_dist:
                        min_dist = dist
                        crush_pos = pos
            elif len(oppo_temp) > 0:crush_pos = oppo_temp[0]
            else: crush_pos=None
        return crush_pos

    def choose_action(self, obs, throw_left, color, deterministic=False):
        state = torch.from_numpy(obs).float() # [1, 25, 25]
        # 如果出现throw次数的变化，说明到了新的投掷回合
        if throw_left[self.agent_idx] != self.throw_left:
            self.throw_left = throw_left[self.agent_idx]
            self.reset()
        # 更新智能体的观测序列(放在reset之后)
        if self.obs_sequence is not None:
            self.obs_sequence = torch.cat((self.obs_sequence[1:, :, :], state), dim=0) #[4, 25, 25]
        else:
            self.obs_sequence = state.repeat(4, 1, 1) # [4, 25, 25]
        if self.throw_left == 0 and throw_left[1-self.agent_idx] == 0:
            self.last_throw = True
        if self.throw_left == 3 and throw_left[1-self.agent_idx] == 4:
            self.first_throw = True
        if self.first_throw:
            # 加载新策略,仅用于第一个回合
            if self.pose[1] < 310:
                actions = [15, 0]
            elif np.abs(self.v[1] - 16) >= 0.1:
                k_gain = 15
                force = -k_gain*(self.v[1] - 16)
                force = np.clip(force, -100, 200)
                actions = [force, 0]
            else:
                actions = [0,0]
            self.ep_count += 1
            return actions
        if self.ep_count <= self.fix_forward_count+self.fix_backward_count:
            if self.ep_count <=self.fix_forward_count:
                actions = [self.fix_forward_force,0]
            else:
                actions = [self.fix_backward_force, 0]
                # 判断场上对方的位置信息（只做一次）
                if self.ep_count == self.fix_forward_count+1:
                    self.crush_pos = self._store_oppo_pos(color, obs)
        # 让智能体先停止下来(比例控制)
        elif self.stage == 0:
            if np.abs(self.v[1] - 0) >= 0.1:
                k_gain = 15
                force = -k_gain*(self.v[1] - 0)
                force = np.clip(force, -100, 200)
                actions = [force, 0]
            else: 
                self.stage = 1
        if self.stage != 0:
            if self.last_throw == True:
                win = self._current_winner()
                if self.game_round == 1:
                    point = self._current_point()
                    # 如果判断已经稳赢的情况下，则不做任何动作，避免风险
                    if self.score[self.agent_idx]+point > self.score[1-self.agent_idx]:
                        self.ep_count += 1
                        return [0,0]
                if win:
                    self.crush_pos = None            
            if self.crush_pos is not None:
                actions = self.get_rule_action(self.crush_pos)
            else:
                load_name = self.decide_model()
                self.load_model(self.model_dict[load_name])
                actions = self.get_model_action(deterministic)
        self.ep_count += 1
        return actions 

    def load_model(self, pth):

        self.actor.load_model(pth)
    
    def save_model(self, pth):

        self.actor.save_model(pth)

    def decide_model(self):
        """
        根据场上的信息来选择用哪个模型
        """
        win = self._current_winner()
        record = []
        load_name = None
        if win:
            if self.throw_left == 0:
                load_name = 'center'
                return load_name
            if len(self.oppo_pos) > 0:
                for pos in self.oppo_pos:
                    if np.abs(pos[0] - 300) < 10:
                        load_name = 'right'
                        break
            if len(self.own_pos) > 0 and load_name !='right':
                for point in self.own_pos:
                    if point[0] > 300:
                        if point[1] > 500:
                            record.append('right_down')
                        if point[0] < 500:
                            record.append('right_up')
                    elif point[0] < 300:
                        if point[1] > 500:
                            record.append('left_down')
                        if point[0] < 500:
                            record.append('left_up')
                if 'right_down' not in record:
                    load_name = 'down'
                elif 'right_up' not in record:
                    load_name = 'right'
                elif 'left_down' not in record:
                    load_name = 'left'
                elif 'left_up' not in record:
                    load_name = 'up'
                else: load_name = 'center'
        else:
            load_name = 'center'
        return load_name

    def _current_winner(self):
        """
        得到目前场上的胜方
        """
        if len(self.oppo_pos) > 0:
            oppo_dist = self._get_mindist(self.oppo_pos)
        else:
            oppo_dist = 1e4 ## 
        if len(self.own_pos) > 0:
            team_dist = self._get_mindist(self.own_pos)
        else: 
            team_dist = 1e4 ## 
        win = True if oppo_dist > team_dist else False

        return win 

    def _get_mindist(self, point_list):
        mini_dist = np.inf
        for point in point_list:
            distance = np.linalg.norm(np.array(point) - [300,500])
            if distance < mini_dist:
                mini_dist = distance
        return mini_dist

    def _current_point(self):
        """
        得到目前场上的分数信息
        """
        point = 0
        if len(self.oppo_pos) > 0:
            mindist = self._get_mindist(self.oppo_pos)
        else:
            mindist = 67
        if len(self.own_pos):
            for pos in self.own_pos:
                distance = np.linalg.norm((np.array(pos) - np.array([300, 500])))
                if mindist > distance:
                    point += 1
        if point == 0:
            if len(self.oppo_pos) > 0:
                mindist = self._get_mindist(self.own_pos)
            else:
                mindist = 67
            if len(self.oppo_pos) > 0:
                for pos in self.oppo_pos:
                    distance = np.linalg.norm((np.array(pos) - np.array([300, 500])))
                    if mindist > distance:
                        point -= 1
        return point 

state_shape = [4, 30, 30]
action_num = 49
continue_space = Box(low=np.array([-100, -10]), high=np.array([200, 10]))   
discrete_space = Discrete(action_num)
load_right = os.path.dirname(os.path.abspath(__file__)) + "/actor_right.pth"
load_left = os.path.dirname(os.path.abspath(__file__)) + "/actor_left.pth"
load_center = os.path.dirname(os.path.abspath(__file__)) + "/actor_center.pth"
load_top = os.path.dirname(os.path.abspath(__file__)) + "/actor_top.pth"
load_down = os.path.dirname(os.path.abspath(__file__)) + "/actor_down.pth"


agent = RLAgent(state_shape, discrete_space)
agent_base = RLAgent(state_shape, discrete_space)
agent.set_model_dict(load_center, 'center')
agent.set_model_dict(load_right, 'right')
agent.set_model_dict(load_left, 'left')
agent.set_model_dict(load_top, 'up')
agent.set_model_dict(load_down, 'down')
agent_base.set_model_dict(load_center, 'center')
agent_base.set_model_dict(load_right, 'right')
agent_base.set_model_dict(load_left, 'left')
agent_base.set_model_dict(load_top, 'up')
agent_base.set_model_dict(load_down, 'down')


def my_controller(observation_list, action_space_list, is_act_continuous):

    ## observation infomation
    obs = observation_list['obs'].copy()
    control_index = observation_list['controlled_player_index']
    agent.set_agent_idx(control_index)
    throws_left = observation_list['throws left']
    color = observation_list['team color']
    score_point = observation_list['score']
    game_round = observation_list['game round']
    agent.set_game_information(score_point, game_round)
    # for RL agent decide action
    obs = np.array(obs)
    actions = agent.choose_action(obs, throws_left, color, True)
    wrapped_actions = [[actions[0]], [actions[1]]]
    # update physical information 
    agent.step([actions[0], actions[1]])
    return wrapped_actions

