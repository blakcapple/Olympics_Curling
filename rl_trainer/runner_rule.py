from math import floor
from typing import Dict
from numpy import newaxis
from pygame import key
import torch 
import numpy as np 
from rl_trainer.algo.agent import rl_agent
import time
import os 
import pdb
import wandb
from copy import deepcopy
from gym.spaces import Box, Discrete
import re
from torch.distributions import Categorical
from spinup.utils.mpi_pytorch import sync_params
from spinup.utils.mpi_tools import mpi_statistics_scalar, proc_id
import math
from helper import Physical_Agent, calculate_pos, get_reward, write_to_file


class Runner:

    def __init__(self, args, env, meta_policy, opponent, buffer, logger, device, 
                action_space, act_dim):
        
        self.total_epoch_step = args.epoch_step
        self.load_index = args.load_index
        self.load_opponent_index = args.load_opponent_index
        self.local_steps_per_epoch = int(self.total_epoch_step / args.cpu)
        self.eval_step = args.eval_step
        self.randomplay_epoch = args.randomplay_epoch
        self.randomplay_interval = args.randomplay_interval
        self.selfplay_interval = args.selfplay_interval
        self.save_interval = args.save_interval
        self.eval_interval = args.eval_interval
        self.render =  args.render 
        self.env = env
        self.meta_policy = meta_policy
        self.low_policy = rl_agent([4, 30, 30], action_space, device) 
        self.buffer = buffer
        self.logger = logger 
        self.ep_ret_history = [] 
        self.best_ep_ret = -np.inf
        self.device = device
        self.action_space = action_space
        self.act_dim = act_dim
        self.opponent = opponent
        self.save_dir = args.save_dir
        self.load_dir = os.path.join(args.save_path, 'models') # where to load models for opponent
        self.save_index = [] # the models pool
        self.id = proc_id()
        self.actions_map = self._set_actions_map(args.action_num)
        self.continue_train = True # if stop training 

        self.agent_idx = self.id % 2
        self.color = 'purple' if self.agent_idx==0 else 1

    def _read_history_models(self):
        
        patten = re.compile(r'actor_(?P<index>\d+)')
        files = os.listdir(self.load_dir)
        for file in files:
            index = patten.findall(file)
            if len(index) > 0 :
                self.save_index.append(int(index[0]))
        self.save_index.sort() # from low to high sorting
        self.model_score = torch.ones(len(self.save_index), dtype=torch.float64) # initialize scores 
        self.logger.info(f'model_score: {self.model_score}')

    def _set_actions_map(self, action_num):
        #dicretise action space
        forces = np.linspace(-100, 200, num=int(np.sqrt(action_num)), endpoint=True)
        thetas = np.linspace(-10, 10, num=int(np.sqrt(action_num)), endpoint=True)
        actions = [[force, theta] for force in forces for theta in thetas]
        actions_map = {i:actions[i] for i in range(action_num)}
        return actions_map
    
    def _wrapped_action(self, action, who_is_throwing):
        # 根据当前回合是谁在投掷冰球来设计动作；无意义的一方的动作为零向量
        if isinstance(self.action_space, Discrete):
            real_action = self.actions_map[action]
        elif isinstance(self.action_space, Box):
            action = np.clip(action, -1, 1)
            high = self.action_space.high
            low = self.action_space.low
            real_action = low + 0.5*(action + 1.0)*(high - low)
        wrapped_action = [[real_action[0]], [real_action[1]]]
        wrapped_opponent_action = [[0], [0]]
        if who_is_throwing == 0:
            wrapped_action = [wrapped_action, wrapped_opponent_action]
        else:
            wrapped_action = [wrapped_opponent_action, wrapped_action]

        return wrapped_action

    def get_info(self, obs, agent_index):
        info = np.zeros(14) # record game infomation

        if obs[0]['game round'] == 0:
            info[0] = 1
        else:
            info[1] = 1

        if obs[0]['throws left'][agent_index] == 4:
            pass
        elif obs[0]['throws left'][agent_index] == 3:
            info[2] = 1
        elif obs[0]['throws left'][agent_index] == 2:
            info[3] = 1
        elif obs[0]['throws left'][agent_index] == 1:
            info[4] = 1
        elif obs[0]['throws left'][agent_index] == 0:
            info[5] = 1

        if obs[0]['throws left'][1-agent_index] == 4:
            pass
        if obs[0]['throws left'][1-agent_index] == 3:
            info[6] = 1
        elif obs[0]['throws left'][1-agent_index] == 2:
            info[7] = 1
        elif obs[0]['throws left'][1-agent_index] == 1:
            info[8] = 1
        elif obs[0]['throws left'][1-agent_index] == 0:
            info[9] = 1
        if agent_index == 0:
            info[10:12] = np.array(obs[0]['score']) / 4
        else:
            info[10:12] = np.array(obs[0]['score'][::-1]) / 4
        if agent_index == 0:
            info[12] = 1
        else:
            info[13] = 1
        
        return info 

    def obs_transform(self, obs, obs_sequence_dict):
    
        ob_ctrl = obs[0]['obs'][0]
        ob_oppo = obs[1]['obs'][0]

        ob_ctrl = ob_ctrl.reshape(1, *ob_ctrl.shape)
        ob_oppo = ob_oppo.reshape(1, *ob_oppo.shape)

        release_ctrl = obs[0]['release']
        release_oppo = obs[1]['release']
    
        info_ctrl = self.get_info(obs, 0)
        info_oppo = self.get_info(obs, 1)
        
        if obs_sequence_dict is not None:
            obs_sequence1 = np.concatenate((np.delete(obs_sequence_dict['obs'][0], 0, axis=0), ob_ctrl), axis=0)
            obs_sequence2 = np.concatenate((np.delete(obs_sequence_dict['obs'][1], 0, axis=0), ob_oppo), axis=0)
        else:
            obs_sequence1 = np.repeat(ob_ctrl, 4, axis=0)
            obs_sequence2 = np.repeat(ob_oppo, 4, axis=0)
        obs_all = np.append(obs_sequence1[np.newaxis], obs_sequence2[np.newaxis], axis=0)
        release_all = np.array([release_ctrl, release_oppo])
        info_all = np.array([info_ctrl, info_oppo])
        dict = {'obs':obs_all, 'release':release_all, "info":info_all}

        return dict 

    def get_low_policy(self, x):
        r"""
        decide low policy from input
        """
        x = x if np.isscalar(x) else x[0]
        goalx = [234+i*33 for i in range(5)][floor(x/6)]
        goaly = [467+i*33 for i in range(6)][(x-floor(x/6)*6)%6]
        load_path = os.path.join(self.save_dir, f'{goalx}_{goaly}', 'models', 'actor.pth')
        self.low_policy.load_model(load_path)
        return self.low_policy

    def run_low_policy(self, policy, ob, agent_idx):

        r"""
        when meta-policy decides which low policy to run;
        this function runs the low policy till the round end
        """

        done = False
        while not done:
            obs = ob['obs'][agent_idx]
            a = policy.act(torch.as_tensor(obs[newaxis], dtype=torch.float32, device=self.device), True)
            env_a = self._wrapped_action(a, agent_idx)
            raw_next_o, reward, d, pos_info, info = self.env.step(env_a)
            next_ob = self.obs_transform(raw_next_o, ob)
            ob = next_ob
            if info == 'round_end' or info == 'game1_end' or info == 'game2_end' :
                done = True
            if self.render:
                self.env.env_core.render()
        next_ob = self.obs_transform(raw_next_o, None)
        return next_ob, reward, d

    def run_rule_policy(self, py_agent, agent_idx, purple_pos, ob):
        done = False
        k_gain = 9
        v = np.linalg.norm(py_agent.v)
        while np.abs(v - 0) >= 0.1:
            v = np.linalg.norm(py_agent.v)
            force = -k_gain*(v - 0)
            force = 200 if force > 200 else force 
            force = -100 if force < -100 else force 
            env_a = [[[force],[0]],[[0],[0]]] if agent_idx == 0 else [[[0],[0]],[[force],[0]]]
            raw_next_o, reward, d, pos_info, info = self.env.step(env_a)
            py_agent.step([force, 0])
            next_o = self.obs_transform(raw_next_o, ob)
            ob = next_o
            if self.render:
                self.env.env_core.render()
        delta = np.array(purple_pos) - np.array(py_agent.pose)
        delta = delta.reshape(-1)
        radius = math.atan2(delta[0], delta[1])
        delta_theta = math.degrees(radius)
        goal_theta  = py_agent.theta - delta_theta
        while (py_agent.theta != goal_theta):
            theta = goal_theta - py_agent.theta
            theta = np.clip(theta, -30, 30)
            py_agent.step([0, theta])
            env_a = [[[0],[theta]],[[0],[0]]] if agent_idx == 0 else [[[0],[0]],[[0],[theta]]]
            raw_next_o, reward, d, pos_info, info = self.env.step(env_a)
            next_o = self.obs_transform(raw_next_o, ob)
            ob = next_o
            if self.render:
                self.env.env_core.render()
        while not done:
            env_a = [[[200],[0]],[[0],[0]]] if agent_idx == 0 else [[[0],[0]],[[200],[0]]]
            py_agent.step([200, 0])
            raw_next_o, reward, d, pos_info, info = self.env.step(env_a)
            next_o = self.obs_transform(raw_next_o, ob)
            ob = next_o
            if info == 'round_end' or info == 'game1_end' or info == 'game2_end' :
                done = True
            if self.render:
                self.env.env_core.render()
        return ob, reward, d

    def rollout(self, epochs):

        best_ret = -np.inf
        ep_len = 0
        ep_ret = 0
        start_time = time.time()
        episode = 0
        epoch_reward = []
        raw_o = self.env.reset()
        o = self.obs_transform(raw_o, None)
        py_agent = Physical_Agent()
    # Main loop: collect experience in env and update/log each epoch
        for epoch in range(epochs):
            if not self.continue_train:
                break
            step = 0
            x = 0
            while(step < self.local_steps_per_epoch):
                if self.render:
                    self.env.env_core.render()
                if o['obs'][0][-1].all() == 1: # 若全为-1，说明是另一个智能体在投掷冰球
                    who_is_throwing = 1
                else:
                    who_is_throwing = 0
                py_agent.reset()
                # 前n步规则控制
                while (ep_len <= 12): 
                    env_a = [[[50],[0]],[[0],[0]]] if who_is_throwing == 0 else [[[0],[0]],[[50],[0]]]
                    raw_next_o, _, _, pos_info, info = self.env.step(env_a)
                    py_agent.step([50, 0])
                    next_o = self.obs_transform(raw_next_o, o)
                    o = next_o
                    ep_len += 1
                    if self.render:
                        self.env.env_core.render()

                info_ctrl = o['info'][self.agent_idx]
                info_oppo = o['info'][1-self.agent_idx]
                obs_ctrl = o['obs'][self.agent_idx][-1]
                obs_oppo = o['obs'][1-self.agent_idx][-1]

                if who_is_throwing == self.agent_idx:
                    color = 'green' if self.agent_idx == 0 else 'purple'
                    purple_pos = calculate_pos(obs_ctrl, color, py_agent.pose)
                    if len(purple_pos) > 0:
                        temp = deepcopy(purple_pos)
                        for pos in temp:
                            if np.linalg.norm([pos[0] - 300, pos[1] - 500]) > 100 and (pos[0] < 234 or pos[0] > 366):
                                purple_pos.remove(pos)
                    if len(purple_pos) > 0:
                        min_idx = 0
                        if len(purple_pos) > 1:
                            min_dist = -np.inf 
                            for i, pos in enumerate(purple_pos):
                                dist = np.linalg.norm([pos[0] - 300, pos[1] - 500])
                                if dist < min_dist:
                                    dist = min_dist
                                    min_idx = i
                        next_o, reward, done = self.run_rule_policy(py_agent, self.agent_idx, purple_pos[min_idx], o)
                    else:
                        a, v, logp = self.meta_policy.step(torch.as_tensor(obs_ctrl[newaxis][newaxis], dtype=torch.float32, device=self.device),
                                                    torch.as_tensor(info_ctrl[newaxis], dtype=torch.float32, device=self.device))
                        a = 13 
                        low_policy = self.get_low_policy(a)
                        next_o, reward, done = self.run_low_policy(low_policy, o, self.agent_idx)

                else:
                    color = 'green' if (1 - self.agent_idx) ==0 else 'purple'
                    green_pos = calculate_pos(obs_oppo, color, py_agent.pose)
                    if len(green_pos) > 0:
                        temp = deepcopy(green_pos)
                        for pos in temp:
                            if np.linalg.norm([pos[0] - 300, pos[1] - 500]) > 100 and (pos[0] < 234 or pos[0] > 366):
                                green_pos.remove(pos)
                    if len(green_pos) > 0:
                        min_idx = 0
                        if len(green_pos) > 1:
                            min_dist = -np.inf 
                            for i, pos in enumerate(green_pos):
                                dist = np.linalg.norm([pos[0] - 300, pos[1] - 500])
                                if dist < min_dist:
                                    dist = min_dist
                                    min_idx = 0
                        next_o, reward, done = self.run_rule_policy(py_agent, 1 - self.agent_idx, green_pos[min_idx], o)
                    else:
                        a = self.opponent.act(torch.as_tensor(obs_oppo[newaxis][newaxis], dtype=torch.float32, device=self.device),
                                            torch.as_tensor(info_oppo[newaxis], dtype=torch.float32, device=self.device))
                        a = 13
                        low_policy = self.get_low_policy(a)
                        next_o, reward, done = self.run_low_policy(low_policy, o, 1 - self.agent_idx)
                
                o = next_o
                ep_ret += reward[self.agent_idx]
                ep_len = 0 
                if done :
                    py_agent.reset()
                    raw_o = self.env.reset()
                    o = self.obs_transform(raw_o, None)                    









                






