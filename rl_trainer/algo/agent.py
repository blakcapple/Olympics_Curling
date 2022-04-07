import random
from gym.spaces import Box, Discrete
from torch.distributions import Categorical, Normal
from rl_trainer.algo.cnn import CNNLayer
import torch.nn as nn
import numpy as np
import torch  
from helper import calculate_pos
from copy import deepcopy 
from algo.network import CNNGaussianActor,CNNCategoricalActor
import math 

# keep angle between [-pi, pi]
def wrap(angle):
    while angle >= 180:
        angle -= 2*180
    while angle < -180:
        angle += 2*180
    return angle

class RLAgent:
    # constant 
    gamma = 0.98
    delta_t = 0.1
    mass = 1 
    fix_forward_count = 21
    fix_forward_force = 50
    fix_backward_force = -100
    fix_backward_count = 21

    def __init__(self, state_shape, action_space, info_dim, device, mode='rule'):
        
        if isinstance(action_space, Box):
            self.is_act_continuous = True
            self.actor = CNNGaussianActor(state_shape, action_space.shape[0], nn.ReLU)
            self.action_space = action_space
        elif isinstance(action_space, Discrete):
            self.is_act_continuous = False
            self.actor = CNNCategoricalActor(state_shape, action_space.n, nn.ReLU, info_dim)
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
        # 要撞击的对手索引
        self.oppo_numner = 0
        # 规则阶段 1表示已经智能体停止，2表示智能体完成转向计算，3表示完成转向
        self.stage = 0
        # 表征是否是最后一个投掷回合（考虑对手的）
        self.last_throw = False
        # 表征是否是第一个投掷回合（考虑对手的）
        self.first_throw = False
        # 智能体的模型集合
        self.model_dict = dict()
        # 智能体行为模式 'train'：训练模式 'rule':规则模式
        self.mode = mode
        self.device = device

    def set_policy(self, policy):
        
        self.policy = policy 

    def set_model_dict(self, pth, name='normal'):

        self.model_dict[name] = pth

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
        # # 加载默认的模型
        # self.load_model(self.model_dict['normal'])
    
    def set_agent_idx(self, idx):

        self.agent_idx = idx 

    def _action_to_accel(self, action):
        """
        Convert action(force) to acceleration
        """
        self.theta = wrap(self.theta + action[1])
        # self.theta = self.theta + action[1]
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

    def get_rule_action(self):
        """
        得到基于规则的动作
        """
        # 选择距离中心点最近的对手作为撞击对象
        if self.ep_count == self.fix_forward_count+1:
            if len(self.oppo_pos) > 1:
                min_dist = -np.inf 
                for i, pos in enumerate(self.oppo_pos):
                    dist = np.linalg.norm([pos[0] - 300, pos[1] - 500])
                    if dist < min_dist:
                        dist = min_dist
                        self.oppo_numner = i
        # 让智能体完成对目标位置的转向
        if self.stage == 1:
            delta = np.array(self.oppo_pos[self.oppo_numner]) - np.array(self.pose)
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
        opponent_color = 'green' if color=='purple' else 'purple'
        self.oppo_pos = calculate_pos(obs.reshape(30, 30), opponent_color, self.pose)
        self.own_pos = calculate_pos(obs.reshape(30, 30), color, self.pose)
        if len(self.oppo_pos) > 0:
            temp = deepcopy(self.oppo_pos)
            for pos in temp:
                # if np.linalg.norm([pos[0] - 300, pos[1] - 500]) > 102 or (pos[0] < 234):
                #     self.oppo_pos.remove(pos)
                # 继续判断与目标冰球连线上是否存在己方冰球
                if len(self.own_pos) > 0 and len(self.oppo_pos) > 0:
                    delta = np.array(pos) - np.array(self.pose)
                    delta = delta.reshape(-1)
                    radius_oppo = math.atan2(delta[0], delta[1])
                    temp2 = deepcopy(self.own_pos)
                    for pos2 in temp2:
                        delta = np.array(pos2) - np.array(self.pose)
                        delta = delta.reshape(-1)
                        radius_own = math.atan2(delta[0], delta[1])
                        # 角度在5度之内则移除这个点
                        # if np.abs(radius_oppo-radius_own) <= 5/180*np.pi:
                        #     # pdb.set_trace()
                        #     self.oppo_pos.remove(pos)

    def choose_action(self, obs, throw_left, color, deterministic=False, goal=None):
        state = torch.from_numpy(obs).float()
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
        # if self.last_throw:
        #     # 加载新的策略 仅用于最后一个回合
        #     if self.ep_count == 0:
        #         self.load_model(self.model_dict['last'])
        # elif self.first_throw:
        #     # 加载新策略,仅用于第一个回合
        #     if self.ep_count == 0:
        #         self.load_model(self.model_dict['first'])
        if self.ep_count <= self.fix_forward_count+self.fix_backward_count:
            if self.ep_count <=self.fix_forward_count:
                actions = [self.fix_forward_force,0]
            else:
                actions = [self.fix_backward_force, 0]
                # 判断场上对方的位置信息（只做一次）
                if self.ep_count == self.fix_forward_count+1:
                    self._store_oppo_pos(color, obs)
        # 让智能体先停止下来(比例控制)
        elif self.stage == 0:
            if np.abs(self.v[1] - 0) >= 0.1:
                k_gain = 15
                force = -k_gain*(self.v[1] - 0)
                force = np.clip(force, -100, 200)
                actions = [force, 0]
            else: 
                self.stage = 1
                actions = [0, 0]
        elif self.stage != 0:
            if self.mode == 'rule':
                if len(self.oppo_pos) > 0:
                    actions = self.get_rule_action()
                else:
                    actions = self.get_model_action(deterministic)
            elif self.mode == 'train':
                obs_sequence = self.obs_sequence.unsqueeze(0)
                info = [self.goal[0]/10, self.goal[1]/10, self.pose[0]/10, self.pose[1]/10, self.v[0], self.v[1], self.theta]
                info = torch.tensor([info], dtype=torch.float32, device=self.device)
                actions, v, logp = self.policy.step(obs_sequence.to(self.device), info)
                self.ep_count += 1
                return actions, v, logp

        self.ep_count += 1
        return actions

    def load_model(self, pth):

        self.actor.load_model(pth)
    
    def save_model(self, pth):

        self.actor.save_model(pth)
    
    def get_stage(self):
        if self.ep_count > self.fix_forward_count+self.fix_backward_count and np.abs(self.v[1] - 0) < 0.1:
            return 1
        else: return self.stage 

    def get_info(self):
        return [self.goal[0]/10, self.goal[1]/10, self.pose[0]/10, self.pose[1]/10, self.v[0], self.v[1], self.theta]
    
    def set_goal(self, goal):

        self.goal = goal