from math import floor
from numpy import newaxis
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

def write_to_file(file, goal, reward):
    with open(file, 'a') as file_object:
        file_object.write(f'goal:{goal}; ')
        file_object.write(f'reward:{reward} \n')


def get_reward(pos, center):

    if pos[1] < 352:
        reward = -10
        return reward
    distance = math.sqrt((pos[0]-center[0])**2 + (pos[1]-center[1])**2)
    reward = 1 / (distance + 1) * 1000# distance reward 

    return reward

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

    def rollout(self, epochs):

        best_ret = -np.inf
        ep_len = 0
        ep_ret = 0
        start_time = time.time()
        episode = 0
        epoch_reward = []
        raw_o = self.env.reset()
        o = self.obs_transform(raw_o, None)
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

                # 前n步规则控制
                while (ep_len <= 12): 
                    env_a = [[[50],[0]],[[0],[0]]] if who_is_throwing == 0 else [[[0],[0]],[[50],[0]]]
                    raw_next_o, _, _, pos_info, info = self.env.step(env_a)
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
                    a, v, logp = self.meta_policy.step(torch.as_tensor(obs_ctrl[newaxis][newaxis], dtype=torch.float32, device=self.device),
                                                  torch.as_tensor(info_ctrl[newaxis], dtype=torch.float32, device=self.device))
                    step += 1
                    low_policy = self.get_low_policy(a)
                    next_o, reward, done = self.run_low_policy(low_policy, o, self.agent_idx)
                    self.buffer.store(obs_ctrl[newaxis], info_ctrl, a, reward[self.agent_idx], v, logp)
                    self.logger.store(VVals=v)
                else:
                    a = self.opponent.act(torch.as_tensor(obs_oppo[newaxis][newaxis], dtype=torch.float32, device=self.device),
                                        torch.as_tensor(info_oppo[newaxis], dtype=torch.float32, device=self.device))
                    low_policy = self.get_low_policy(a)
                    next_o, reward, done = self.run_low_policy(low_policy, o, 1 - self.agent_idx)
                
                o = next_o
                ep_ret += reward[self.agent_idx]
                ep_len = 0 
                epoch_ended = step==(self.local_steps_per_epoch)
                if done or epoch_ended:
                    if epoch_ended and not(done):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        _, v, _ = self.meta_policy.step(torch.as_tensor(o['obs'][self.agent_idx][-1][newaxis][newaxis], dtype=torch.float32, device=self.device),
                                                   torch.as_tensor(o['info'][self.agent_idx][newaxis], dtype=torch.float32, device=self.device) )
                    else:
                        v = 0
                    self.buffer.finish_path(v)
                    if done:
                        episode +=1
                        epoch_reward.append(ep_ret)
                        win_is = 1 if reward[self.agent_idx] == 100 else 0
                        lose_is = 1 if reward[1-self.agent_idx] == 100 else 0
                        self.logger.store(Win=win_is)
                        self.logger.store(Lose=lose_is)
                        self.logger.store(EpRet=ep_ret)
                        # 每一轮投掷结束之后，obs序列重置
                        o = self.env.reset()
                        o = self.obs_transform(raw_next_o, None)
                        ep_ret = 0

            data = self.buffer.get()
            # update policy
            self.meta_policy.learn(data)
            # Log info about epoch
            avg_KL, _ = mpi_statistics_scalar(self.logger.epoch_dict['KL'])
            self.logger.log_tabular('Win', average_only=True)
            self.logger.log_tabular('Lose', average_only=True)
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('VVals', with_min_and_max=True)
            self.logger.log_tabular('TotalEnvInteracts', (epoch+1)*self.total_epoch_step)
            self.logger.log_tabular('LossPi', average_only=True)
            self.logger.log_tabular('LossV', average_only=True)
            self.logger.log_tabular('DeltaLossPi', average_only=True)
            self.logger.log_tabular('DeltaLossV', average_only=True)
            self.logger.log_tabular('Entropy', average_only=True)
            self.logger.log_tabular('KL', average_only=True)
            self.logger.log_tabular('ClipFrac', average_only=True)
            self.logger.log_tabular('Time', time.time()-start_time)
            self.logger.dump_tabular()
            avg_ret,_ = mpi_statistics_scalar(np.array(epoch_reward))
            epoch_reward = []
            # 保存最好的模型
            if best_ret < avg_ret:
                best_ret = avg_ret
                sync_params(self.meta_policy.ac)
                if self.id == 0:
                    self.meta_policy.save_models()
            # KL收敛后则停止训练
            if avg_KL <= 0.0001 or epoch == (epochs-1):
                # self.continue_train = False
                # 记录最好的奖励值到文件中
                if self.id == 0:
                    write_to_file('best_reward.txt', epoch, best_ret)
            if epoch % self.save_interval == 0 and epoch > 0 or not self.continue_train:
                sync_params(self.meta_policy.ac)
                if self.id == 0:
                    self.meta_policy.save_models(index=epoch)





                






