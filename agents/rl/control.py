from tkinter.messagebox import NO
from numpy import newaxis
import torch.nn as nn
import torch 
import os
from torch.distributions import Categorical, Normal
import numpy as np
from gym.spaces import Box, Discrete
import math 
from copy import deepcopy
import pdb

class PredictNet(nn.Module):
    """
    网络：根据观察到的点，预测冰壶位置
    """
    def __init__(self):
        super().__init__()
        self.linear_layer = mlp([2]+[64]+[64]+[2], nn.LeakyReLU)

    def forward(self, input):
        out = self.linear_layer(input)
        return out
    
    def save_model(self, pth):

        torch.save(self.state_dict(), pth)
    
    def load_model(self, pth):

        self.load_state_dict(torch.load(pth))

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
            x_mean = np.median([point_group[i][0] for i in range(len(point_group))])
            y_mean = np.median([point_group[i][1] for i in range(len(point_group))])
            color_pos.append([x_mean, y_mean])
    relative_pos = color_pos
    real_pos = []
    for pos in relative_pos:
        pos_x = 450 - pos[1]*scalar_x
        pos_y = ego_pos[1]+30*scalar_y - pos[0]*scalar_y
        real_pos.append([pos_x, pos_y])
    return relative_pos, real_pos

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
        # self.cnn_layer = CNNLayer(input_shape)
        # self.extra_layer = nn.Linear(7, 64)
        self.linear_layer = mlp([7]+[256]+[256]+[act_dim], activation)
        # self.logits_net = nn.Sequential(self.cnn_layer, self.linear_layer)
        
    def distribution(self, obs, info):
        # cnn_out = self.cnn_layer(obs)
        # info_out = F.relu(self.extra_layer(info))
        # full = torch.cat([cnn_out, info_out], dim=1)
        logits = self.linear_layer(info)
        # logits = self.logits_net(obs)
        return Categorical(logits=logits)
    
    def select_action(self, obs, info):

        # cnn_out = self.cnn_layer(obs)
        # info_out = F.relu(self.extra_layer(info))
        # full = torch.cat([cnn_out, info_out], dim=1)
        logits = self.linear_layer(info)
        acition = torch.argmax(logits)
        return acition

    def log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

    def forward(self, obs, info, act=None):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self.distribution(obs, info)
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
class LowController:
    # constant 
    gamma = 0.98
    delta_t = 0.1
    mass = 1 
    fix_forward_count = 12
    fix_forward_force = 100
    fix_backward_force = -100
    fix_backward_count = 25
    fix_observe_count = 24
    fix_stop_length = 25 # 减速到零的滑行距离

    def __init__(self, state_shape, action_space, policy='rule'):
        
        if isinstance(action_space, Box):
            self.is_act_continuous = True
            self.actor = CNNGaussianActor(state_shape, action_space.shape[0], nn.ReLU)
            self.action_space = action_space
        elif isinstance(action_space, Discrete):
            self.is_act_continuous = False
            self.actor = CNNCategoricalActor(state_shape, action_space.n, nn.ReLU)
            num = action_space.n
            #dicretise action space
            forces = np.linspace(-100, 200, num=int(np.sqrt(num)), endpoint=True)
            thetas = np.linspace(-10, 10, num=int(np.sqrt(num)), endpoint=True)
            actions = [[force, theta] for force in forces for theta in thetas]
            actions_map = {i:actions[i] for i in range(num)}
            self.actions_map = actions_map
        self.predict_model = PredictNet()
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
        self.crash_pos = None
        # 目标点
        self.goal = [300, 500]
        # 打击方式
        self.crash_way = 0
        # 策略：规则 0 RL 1
        self.policy = 0 if policy == 'rule' else 1

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

        self.goal = [300, 500]
    
    def set_agent_idx(self, idx):

        self.agent_idx = idx 

    def _wrap(self, action):
        
        theta = self.theta + action
        if theta > 180:
            theta -= 360
        elif theta < -180:
            theta += 360
        return theta

    def _action_to_accel(self, action):
        """
        Convert action(force) to acceleration
        """
        self.theta = self._wrap(action[1])
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
        extra_info = self.get_extra_info()
        extra_info = torch.tensor(extra_info[newaxis], dtype=torch.float32)
        if self.is_act_continuous:
            if deterministic:
                a_raw = self.actor.mu_net(obs_sequence, extra_info)
            else:
                pi, _ = self.actor(obs_sequence, extra_info)
                a_raw = pi.sample()
        else:
            if deterministic:
                a_raw = self.actor.select_action(obs_sequence, extra_info)
            else:
                pi, _ = self.actor(obs_sequence)
                a_raw = pi.sample()
        actions = self.actions_map[a_raw.item()]
        return actions

    def _get_crash_way(self, crash_pos):
        """
        判断击打方式
        """
        crash_way = None
        # 计算路线角度
        delta = np.array(crash_pos) - np.array(self.pose)
        delta = delta.reshape(-1)
        crush_radius = math.atan2(delta[0], delta[1])
        self.other_pos = deepcopy(self.own_pos + self.oppo_pos)
        self.other_pos.remove(crash_pos)
        if len(self.other_pos) == 0:
            crash_way = 0
        if crash_pos[0] < 200 or crash_pos[0] > 400:
            crash_way = 2
        if crash_way == None :
            for pos in self.other_pos:
                delta = np.array(pos) - np.array(self.pose)
                delta = delta.reshape(-1)
                other_radius = math.atan2(delta[0], delta[1])
                other_distance = np.linalg.norm(delta)
                # 弧度差*距离 为撞击预留出一定的半径空间，防止撞击到别的冰壶
                if np.abs(crush_radius-other_radius)*other_distance <= 38: # 这里的判断可以更加精准一点
                    crash_way = 1
                    break
                else: crash_way = 0
        return crash_way

    def get_rule_action(self, crash_way):
        """
        得到基于规则的动作
        """
        # # # 对撞击点做修正
        # if np.abs(self.crash_pos[0] - 300) < 1:
        #     crash_pos =  [self.crash_pos[0], self.crash_pos[1]]
        # elif self.crash_pos[0] > 300:
        #     crash_pos = [self.crash_pos[0] - 3, self.crash_pos[1]]
        # else:
        #     crash_pos = [self.crash_pos[0] + 3, self.crash_pos[1]]
        crash_pos =  [self.crash_pos[0], self.crash_pos[1]]
        if crash_way == 0:
            # 往上移动一段距离
            if self.stage == 1:
                if self.pose[1] > 130:
                    actions = [-50,0]
                elif self.pose[1] < 130:
                    # 比例控制减速到零
                    k_gain = 15
                    force = k_gain*(0-self.v[1])
                    force = np.clip(force, -100, 200)
                    actions = [force,0]
                    if np.abs(self.v[1])<0.1: 
                        self.stage = 2
            # 让智能体完成对目标位置的转向
            if self.stage == 2:
                delta = np.array(crash_pos) - np.array(self.pose)
                delta = delta.reshape(-1)
                radius = math.atan2(delta[0], delta[1])
                delta_theta = math.degrees(radius)
                self.goal_theta  = self.theta - delta_theta
                self.stage = 3
            if self.stage == 3:
                if self.theta != self.goal_theta:
                    theta = self.goal_theta - self.theta
                    theta = np.clip(theta, -30, 30)
                    actions  = [0, theta]
                else: self.stage = 4
            if self.stage == 4:
                actions = [200, 0]
        elif crash_way == 1:
            if self.stage == 1:
                other_oppo_pos = deepcopy(self.oppo_pos)
                other_oppo_pos.remove(self.crash_pos)
                # 选择打击移动方向
                if len(self.own_pos) > 0:
                    min_distance = 1e4
                    for pos in self.own_pos:
                        delta = np.array(pos) - np.array([300, 500])
                        other_distance = np.linalg.norm(delta)
                        if other_distance < min_distance:
                            min_distance = other_distance
                            min_team_pos = pos
                    delta_team = np.array(min_team_pos) - np.array(crash_pos) 
                    if np.prod(delta_team) < 0:
                        self.goal_theta = 180
                        self.goal_v = -70
                    else:
                        self.goal_theta = 0
                        self.goal_v = 70
                elif len(other_oppo_pos) > 0:
                    min_distance = 1e4
                    for pos in other_oppo_pos:
                        delta = np.array(pos) - np.array(crash_pos)
                        other_distance = np.linalg.norm(delta)
                        if other_distance < min_distance:
                            min_distance = other_distance
                            min_oppo_pos = pos
                    delta_oppo = np.array(min_oppo_pos) - np.array(crash_pos) 
                    if np.prod(delta_oppo) > 0:
                        self.goal_theta = 0
                        self.goal_v = 70
                    else:
                        self.goal_theta = 180
                        self.goal_v = -70
                elif crash_pos[0] > 300:
                    self.goal_theta = 0
                    self.goal_v = 70
                else: 
                    self.goal_theta = 180
                    self.goal_v = -70
                self.stage = 2
            if self.stage == 2:
                if self.theta != self.goal_theta:
                    theta = self.goal_theta - self.theta
                    theta = np.clip(theta, -30, 30)
                    actions  = [0, theta]
                else: 
                    self.stage = 3
            if self.stage == 3:
                # 比例控制匀速运动
                k_gain = 15
                if self.goal_theta == 180:
                    force = -k_gain*(self.goal_v-(self.v[0]))
                else:
                    force = k_gain*(self.goal_v-(self.v[0]))
                force = np.clip(force, -100, 200)
                actions = [force, 0]
                ready_crush = False
                if np.abs(self.v[0] - self.goal_v) < 1:
                    if self.v[0] > 0:
                        future_pose = [self.pose[0] + self.fix_stop_length, self.pose[1]] 
                    else:
                        future_pose = [self.pose[0] - self.fix_stop_length, self.pose[1]] 
                    delta = np.array(crash_pos) - np.array(future_pose)
                    delta = delta.reshape(-1)
                    crush_radius = math.atan2(delta[0], delta[1])
                    for pos in self.other_pos:
                        delta = np.array(pos) - np.array(future_pose)
                        delta = delta.reshape(-1)
                        other_radius = math.atan2(delta[0], delta[1])
                        other_distance = np.linalg.norm(delta)
                        if np.abs(crush_radius-other_radius) * other_distance <= 36:
                            ready_crush = False
                            break
                        else: ready_crush=True
                    if ready_crush or self.pose[0]> 400 or self.pose[0]< 200:
                        self.stage = 4
            if self.stage == 4:
                # 比例控制减速到零
                k_gain = 15
                force = -k_gain*(0-(self.v[0]))
                if self.goal_theta == 0:
                    force = -force
                force = np.clip(force, -100, 200)
                actions = [force, 0]
                if np.abs(self.v[0] - 0) < 0.1:
                    self.stage = 5
            if self.stage == 5:
                delta = np.array(crash_pos) - np.array(self.pose)
                delta = delta.reshape(-1)
                radius = math.atan2(delta[0], delta[1])
                delta_theta = math.degrees(radius)
                self.goal_theta  = 90 - delta_theta
                self.stage = 6
            if self.stage == 6:
                if self.theta != self.goal_theta:
                    theta = self.goal_theta - self.theta
                    theta = np.clip(theta, -30, 30)
                    actions  = [0, theta]
                else: self.stage = 7
            if self.stage == 7:
                actions = [200, 0]
        elif crash_way == 2:
            if self.stage == 1:
                if crash_pos[0] > 300:
                    self.goal_theta = 0
                    self.goal_v = 70
                else: 
                    self.goal_theta = 180
                    self.goal_v = -70
                self.stage = 2
            if self.stage == 2:
                if self.theta != self.goal_theta:
                    theta = self.goal_theta - self.theta
                    theta = np.clip(theta, -30, 30)
                    actions  = [0, theta]
                else: 
                    self.stage = 3
            if self.stage == 3:
                # 比例控制匀速运动
                k_gain = 15
                if self.goal_theta == 180:
                    force = -k_gain*(self.goal_v-(self.v[0]))
                else:
                    force = k_gain*(self.goal_v-(self.v[0]))
                force = np.clip(force, -100, 200)
                actions = [force, 0]
                ready_crush = False
                if np.abs(self.v[0] - self.goal_v) < 1:
                    if np.abs(self.pose[0] - crash_pos[0])< self.fix_stop_length+12:
                        self.stage = 4
            if self.stage == 4:
                # 比例控制减速到零
                k_gain = 15
                force = -k_gain*(0-(self.v[0]))
                if self.goal_theta == 0:
                    force = -force
                force = np.clip(force, -100, 200)
                actions = [force, 0]
                if np.abs(self.v[0] - 0) < 0.1:
                    self.stage = 5
            if self.stage == 5:
                self.goal_theta  = 90
                self.stage = 6
            if self.stage == 6:
                if self.theta != self.goal_theta:
                    theta = self.goal_theta - self.theta
                    theta = np.clip(theta, -30, 30)
                    actions  = [0, theta]
                else: self.stage = 7
            if self.stage == 7:
                actions = [200, 0]
        return actions
    
    def _store_oppo_pos(self, color, obs):
        """
        存储有价值的冰壶位置信息
        """
        crash_pos = None
        opponent_color = 'green' if color=='purple' else 'purple'
        with torch.no_grad():
            relative_oppo_pos, oppo_pos = calculate_pos(obs.reshape(30, 30), opponent_color, self.pose)
            self.oppo_pos = oppo_pos
            # if len(relative_oppo_pos) > 0:
            #     self.oppo_pos = self.predict_model(torch.FloatTensor(relative_oppo_pos)).numpy()
            #     self.oppo_pos = self.oppo_pos.tolist()
            # else: self.oppo_pos = []
            relative_own_pos, team_pos = calculate_pos(obs.reshape(30, 30), color, self.pose)
            self.own_pos = team_pos
            # if len(relative_own_pos) > 0:
            #     self.own_pos = self.predict_model(torch.FloatTensor(relative_own_pos)).numpy()
            #     self.own_pos = self.own_pos.tolist()
            # else: self.own_pos = []
        if len(self.oppo_pos) > 0:
            oppo_temp = deepcopy(self.oppo_pos)
            # 选择距离中心点最近的对手作为撞击对象
            if len(oppo_temp) > 1:
                min_dist = np.inf 
                for i, pos in enumerate(oppo_temp):
                    dist = np.linalg.norm([pos[0] - 300, pos[1] - 500])
                    if dist < min_dist:
                        min_dist = dist
                        crash_pos = pos
            elif len(oppo_temp) > 0:crash_pos = oppo_temp[0]
            else: crash_pos=None
        self.crash_pos = crash_pos

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
            # # 加载新策略,仅用于第一个回合
            # if self.pose[1] < 310:
            #     actions = [15, 0]
            # elif np.abs(self.v[1] - 16) >= 0.1:
            #     k_gain = 15
            #     force = -k_gain*(self.v[1] - 16)
            #     force = np.clip(force, -100, 200)
            #     actions = [force, 0]
            # else:
            #     actions = [0,0]
            # self.ep_count += 1
            # return actions
            self.goal = [430, 620]
        if self.ep_count <= self.fix_forward_count+self.fix_backward_count:
            if self.ep_count <=self.fix_forward_count:
                actions = [self.fix_forward_force,0]
            else:
                actions = [self.fix_backward_force, 0]
                # 判断场上对方的位置信息（只做一次）
                if self.ep_count == self.fix_observe_count:
                    self._store_oppo_pos(color, obs)
        # 让智能体先停止下来(比例控制)
        elif self.stage == 0:
            if np.abs(self.v[1] - 0) >= 0.1:
                k_gain = 17
                force = -k_gain*(self.v[1] - 0)
                force = np.clip(force, -100, 200)
                actions = [force, 0]
            else: 
                self.stage = 1
                if self.crash_pos is not None:
                    self.crash_way = self._get_crash_way(self.crash_pos)
        if self.stage != 0:
            if self.policy == 0: 
                if self.crash_pos is not None:
                    actions = self.get_rule_action(self.crash_way)
                else:
                    if self.throw_left == 3:
                        self.goal = [180, 630]
                    elif self.throw_left == 2:
                        self.goal = [420, 630]
                    elif self.throw_left == 1:
                        self.goal = [280, 630]
                    elif self.throw_left == 0:
                        self.goal = [300, 500]
                    actions = self.get_model_action(deterministic)
            else: 
                actions = self.get_model_action(deterministic)
        self.ep_count += 1
        return actions 

    def load_model(self, pth):

        self.actor.load_model(pth)
    
    def save_model(self, pth):

        self.actor.save_model(pth)

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

    def get_extra_info(self):

        info =self.goal + self.pose + self.v + [self.theta/180*np.pi]
        info = np.array(info)
        info[:6] /= 10
        return info

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
            if len(self.own_pos) > 0:
                mindist = self._get_mindist(self.own_pos)
            else:
                mindist = 67
            if len(self.oppo_pos) > 0:
                for pos in self.oppo_pos:
                    distance = np.linalg.norm((np.array(pos) - np.array([300, 500])))
                    if mindist > distance:
                        point -= 1
        return point 


    def set_goal(self, goal):
        """
        设定冰壶落点目标
        """
        self.goal = goal 

    def switch(self, policy):

        if policy == 0:
            self.policy = 0
            self.goal = [300, 500]
        elif policy == 1:
            self.policy = 1

state_shape = [4, 30, 30]
action_num = 49
continue_space = Box(low=np.array([-100, -10]), high=np.array([200, 10]))   
discrete_space = Discrete(action_num)
load_model_pth = os.path.dirname(os.path.abspath(__file__)) + "/actor.pth"
low_controller = LowController(state_shape, discrete_space)
low_controller.load_model(load_model_pth)



