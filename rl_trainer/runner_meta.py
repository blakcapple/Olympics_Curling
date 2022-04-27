from numpy import newaxis
import torch 
import numpy as np 
import time
import os 
import pdb
import wandb
from copy import deepcopy
from gym.spaces import Box, Discrete
import re
from spinup.utils.mpi_pytorch import sync_params
from spinup.utils.mpi_tools import mpi_statistics_scalar, proc_id
import math
from agents.rl.control import low_controller


def write_to_file(file, goal, reward):
    with open(file, 'a') as file_object:
        file_object.write(f'goal:{goal}; ')
        file_object.write(f'reward:{reward} \n')

class Runner:

    def __init__(self, args, env, policy, opponent, buffer, logger, device, 
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
        self.policy = policy
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
        self.goals_map = self._set_goal_map(args.action_num-1)
        self.continue_train = True # if stop training 

        self.agent_idx = self.id % 2
        self.algo_list = ['oppo', 'team'] if self.agent_idx == 1 else ['team', 'oppo']
        self.color = 'purple' if self.agent_idx == 0 else 'green'
        # 底层控制器
        self.team_controller = deepcopy(low_controller)
        self.oppo_controller = deepcopy(low_controller)

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

    def _set_goal_map(self, action_num):
        #dicretise action space
        goalx_set = np.linspace(200, 400, num=int(np.sqrt(action_num)), endpoint=True)
        goaly_set = np.linspace(400, 700, num=int(np.sqrt(action_num)), endpoint=True)
        goals = [[goalx, goaly] for goalx in goalx_set for goaly in goaly_set]
        goals_map = {i:goals[i] for i in range(action_num)}
        return goals_map

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

    def obs_transform(self, controller, obs, agent_idx):
        """
        提取观察到的冰壶信息和游戏信息
        """
        team_pos = controller.own_pos
        oppo_pos = controller.oppo_pos
        team_info = np.zeros((4, 3))
        oppo_info = np.zeros((4, 3))
        for i, pos in enumerate(team_pos):
            team_info[i] = [pos[0]/10, pos[1]/10, 1]
        for i, pos in enumerate(oppo_pos):
            oppo_info[i] = [pos[0]/10, pos[1]/10, 0]
                    
        game_info = self.get_info(obs, agent_idx)
        state = np.concatenate([team_info.reshape(-1), oppo_info.reshape(-1), game_info])
        
        return state 

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
            next_state, reward, done, _, info = self.env.step(joint_actions)
            state = next_state
            if info == 'round_end' or info == 'game1_end' or info == 'game2_end' :
                round_done = True
            if self.render:
                self.env.env_core.render()
        self.team_controller.ep_count = 0 
        self.oppo_controller.ep_count = 0
        return next_state, reward, done, info

    def rollout(self, epochs):

        best_ret = -np.inf
        ep_len = 0
        ep_ret = 0
        start_time = time.time()
        episode = 0
        epoch_reward = []
        obs = self.env.reset()
    # Main loop: collect experience in env and update/log each epoch
        for epoch in range(epochs):
            if not self.continue_train:
                break
            for step in range(self.local_steps_per_epoch):
                if self.render:
                    self.env.env_core.render()
                if self.agent_idx == 0:
                    while self.team_controller.ep_count < 47:
                        joint_action = self.get_actions(obs)
                        next_obs, _,_,_,_=self.env.step(joint_action)
                        obs = next_obs
                        if self.render:
                            self.env.env_core.render()
                    # 根据策略选择底层策略；底层策略执行到该投掷回合结束
                    if self.team_controller.ep_count == 47:
                        state = self.obs_transform(self.team_controller, obs, self.agent_idx)
                        state = torch.as_tensor(state[newaxis], dtype=torch.float32, device=self.device)
                        action, value, logp = self.policy.step(state)
                        if action.item() == 49:
                            self.team_controller.switch(0) # 转换到规则策略
                        else:
                            goal = self.goals_map[action.item()]
                            self.team_controller.set_goal(goal)
                            self.team_controller.switch(1) # 转换到RL
                        next_obs, reward, done, info = self.run_round_to_end(obs)
                    next_obs, reward, done, info = self.run_round_to_end(next_obs)
                else:
                    next_obs, reward, done, info = self.run_round_to_end(obs)
                    while self.team_controller.ep_count < 47:
                        joint_action = self.get_actions(obs)
                        next_obs,_,_,_,_ = self.env.step(joint_action)
                        obs = next_obs
                        if self.render:
                            self.env.env_core.render()
                    # 根据策略选择底层策略；底层策略执行到该投掷回合结束
                    if self.team_controller.ep_count == 47:
                        state = self.obs_transform(self.team_controller, obs, self.agent_idx)
                        state = torch.as_tensor(state[newaxis], dtype=torch.float32, device=self.device)
                        action, value, logp = self.policy.step(state)
                        if action.item() == 49:
                            self.team_controller.switch(0) # 转换到规则策略
                        else:
                            goal = self.goals_map[action.item()]
                            self.team_controller.set_goal(goal)
                            self.team_controller.switch(1) # 转换到RL
                        next_obs, reward, done, info = self.run_round_to_end(obs)
                self.buffer.store(state, action, reward[self.agent_idx], value, logp)
                self.logger.store(VVals=value)
                obs = next_obs
                ep_ret += reward[self.agent_idx]
                epoch_ended = step==(self.local_steps_per_epoch-1)
                if done or epoch_ended:
                    if epoch_ended and not(done):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        state = self.obs_transform(self.team_controller, obs, self.agent_idx)
                        state = torch.as_tensor(state[newaxis], dtype=torch.float32, device=self.device)
                        _, value, _ = self.policy.step(state)
                    else:
                        value = 0
                    self.buffer.finish_path(value)
                    if done:
                        episode +=1
                        epoch_reward.append(ep_ret)
                        win_is = 1 if reward[self.agent_idx] > reward[1-self.agent_idx] else 0
                        lose_is = 1 if reward[self.agent_idx] < reward[1-self.agent_idx] else 0
                        self.logger.store(Win=win_is)
                        self.logger.store(Lose=lose_is)
                        self.logger.store(EpRet=ep_ret)
                        # 每一轮投掷结束之后，obs序列重置
                        obs = self.env.reset()
                        ep_ret = 0

            data = self.buffer.get()
            # update policy
            self.policy.learn(data)
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
                sync_params(self.policy.ac)
                if self.id == 0:
                    self.policy.save_models()
            # KL收敛后则停止训练
            if avg_KL <= 0.0001 or epoch == (epochs-1):
                # self.continue_train = False
                # 记录最好的奖励值到文件中
                if self.id == 0:
                    write_to_file('best_reward.txt', epoch, best_ret)
            if epoch % self.save_interval == 0 and epoch > 0 or not self.continue_train:
                sync_params(self.policy.ac)
                if self.id == 0:
                    self.policy.save_models(index=epoch)





                






