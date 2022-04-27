from numpy import newaxis
import torch 
import numpy as np 
import time
import os 
import pdb
from copy import deepcopy
from spinup.utils.mpi_tools import num_procs, proc_id
import math

from agents.rl.control import low_controller
import pickle
from os import path
import torch.nn as nn
from rl_trainer.algo.network import mlp 

def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets'))

def write_to_file(file, collect_samples):
    with open(file, 'a') as file_object:
        file_object.write(f'{collect_samples}')

class PredictNet(nn.Module):
    """
    网络：根据观察到的点，预测冰壶位置
    """
    def __init__(self, lr, device):
        self.linear_layer = mlp([2]+[32]+[32]+[2], nn.LeakyReLU())
        self.to(device)
        self.lose_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters, lr=lr)
    def froward(self, input):
        out = self.linear_layer(input)
        return out 

class Runner:

    def __init__(self, args, env, logger):
        
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
        self.logger = logger
        self.render =  args.render 
        self.env = env
        self.ep_ret_history = [] 
        self.best_ep_ret = -np.inf
        self.save_dir = args.save_dir
        self.load_dir = os.path.join(args.save_path, 'models') # where to load models for opponent
        self.save_index = [] # the models pool
        self.id = proc_id()
        self.continue_train = True # if stop training 

        self.agent_idx = 1
        self.algo_list = ['oppo', 'team'] if self.agent_idx == 1 else ['team', 'oppo']
        self.color = 'purple' if self.agent_idx == 0 else 'green'
        # 底层控制器
        self.team_controller = deepcopy(low_controller)
        self.oppo_controller = deepcopy(low_controller)

        self.goalx_num = 80
        self.goaly_num = 80

        self.start_x = 180 + (420-180)*proc_id()/num_procs()
        self.end_x = 180 + (420-180)*(proc_id()+1)/num_procs()
        self.start_y = 380 + (650-400)*proc_id()/num_procs()
        self.end_y = 380 + (650-400)*(proc_id()+1)/num_procs()
        self.goals_map = self._set_goal_map()


    def _set_goal_map(self):
        #dicretise action space
        goalx_set = np.linspace(self.start_x, self.end_x, num=int((self.goalx_num)), endpoint=True)
        goaly_set = np.linspace(self.start_y, self.end_y, num=int((self.goaly_num)), endpoint=True)
        goals = [[goalx, goaly] for goalx in goalx_set for goaly in goaly_set]
        goals_map = {i:goals[i] for i in range(self.goalx_num*self.goaly_num)}
        return goals_map

    def get_actions(self, state):

        joint_actions = []

        for agent_idx in range(len(self.algo_list)):

            if self.algo_list[agent_idx] == 'oppo':

                obs = state[agent_idx]['obs']
                index = state[agent_idx]['controlled_player_index']
                throws_left = state[agent_idx]['throws left']
                color = state[agent_idx]['team color']
                score_point = state[agent_idx]['score']
                game_round = state[agent_idx]['game round']
                self.oppo_controller.set_game_information(score_point, game_round)
                self.oppo_controller.set_agent_idx(index)
                obs = np.array(obs)
                actions = self.oppo_controller.choose_action(obs, throws_left, color, True)
                joint_actions.append([[actions[0]], [actions[1]]])
                self.oppo_controller.step([actions[0], actions[1]])

            elif self.algo_list[agent_idx] == 'team':
                obs = state[agent_idx]['obs']
                index = state[agent_idx]['controlled_player_index']
                throws_left = state[agent_idx]['throws left']
                color = state[agent_idx]['team color']
                score_point = state[agent_idx]['score']
                game_round = state[agent_idx]['game round']
                self.team_controller.set_game_information(score_point, game_round)
                self.team_controller.set_agent_idx(index)
                obs = np.array(obs)
                actions = self.team_controller.choose_action(obs, throws_left, color, True)
                joint_actions.append([[actions[0]], [actions[1]]])
                self.team_controller.step([actions[0], actions[1]])

        return joint_actions      

    def run_round_to_end(self, state):
        """
        执行一个冰壶投掷回合
        """
        round_done = False
        while not round_done:
            joint_actions = self.get_actions(state)
            next_state, reward, done, pos_info, info = self.env.step(joint_actions)
            state = next_state
            if info == 'round_end' or info == 'game1_end' or info == 'game2_end' :
                round_done = True
            if self.render:
                self.env.env_core.render()
        self.team_controller.ep_count = 0 
        self.oppo_controller.ep_count = 0
        return next_state, reward, done, pos_info

    def rollout(self):
        collect_samples = []
        start_time = time.time()
        for epoch in range(self.goalx_num*self.goaly_num):
            obs = self.env.reset()
            self.oppo_controller.reset()
            self.team_controller.reset()
            if self.render:
                self.env.env_core.render()
            self.oppo_controller.set_goal(self.goals_map[epoch])
            next_obs, _, _, pos_info = self.run_round_to_end(obs)
            while self.team_controller.ep_count < 47:
                joint_action = self.get_actions(obs)
                next_obs,_,_,_,_=self.env.step(joint_action)
                obs = next_obs
                if self.render:
                    self.env.env_core.render()
            if self.team_controller.ep_count == 47:
                if len(self.team_controller.oppo_pos) == 0:
                    continue
                predict_pose = self.team_controller.oppo_pos[0]
                real_pos = pos_info
                collect_pair = [predict_pose, real_pos]
                collect_samples.append(collect_pair)
                run_time = time.time()
                print(f'Epoch:{epoch} Thread ID:{proc_id()} point:{collect_pair} goal:{self.goals_map[epoch]}, time:, {run_time-start_time:.3f}')
            if epoch % 100 == 0:
                write_to_file(os.path.join(assets_dir(), 'expert_traj/log_{}.txt'.format(proc_id())), collect_samples)
        pickle.dump((collect_samples), open(os.path.join(assets_dir(), 'expert_traj/expert_traj_{}.p'.format(proc_id())), 'wb'))


    # def train(self, path, epoch, optim_batch_size):
    #     for i in range(num_procs()):
    #         load_pth = os.path.join(assets_dir(), 'expert_traj/expert_traj_{}.p'.format(proc_id()))
    #         sampels =  
    #     samples = pickle.load(open(path, "rb"))
    #     pdb.set_trace()
    #     optim_iter_num = int(math.ceil(samples.shape[0] / optim_batch_size))
    #     for i in range(epoch):
    #         perm = np.arrage(samples.shape[0])
    #         np.random.shuffle(perm)
    #         for _ in range(optim_iter_num):
    #             ind = slice(i * optim_batch_size, min((i + 1) * optim_batch_size, samples.shape[0]))
    #             # samples[ind]






                






