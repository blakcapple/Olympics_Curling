from curses import raw
from random import random
from numpy import average, newaxis
from pathlib import Path
import torch 
import numpy as np 
from rl_trainer.algo.opponent import random_agent, rl_agent
import time
import os 
import pdb
import wandb
from copy import deepcopy
from gym.spaces import Box, Discrete
import re
from torch.distributions import Categorical
from spinup.utils.mpi_pytorch import sync_params
from spinup.utils.mpi_tools import proc_id, mpi_avg


def action_check(a):
    a = a if np.isscalar(a) else a[0]

    return a 
    
class Runner:

    def __init__(self, args, env, policy, icm_agent, opponent, buffer, logger, device, 
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
        self.log_interval = args.log_interval
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
        self.load_dir = os.path.join(args.save_dir, 'models') # where to load models for opponent
        self.id = proc_id()
        self.render = args.render 
        if isinstance(action_space, Discrete):
            self.actions_map = self._set_actions_map(action_space.n)
        else:
            self.actions_map = None
        # self-play variable  
        self.save_index = [] # the model pool
        self.model_score = [] # the score of the historical models (used to sample)

        if args.load_opponent_index > 0:
            self.random_play_flag = False
            self.self_play_flag = True  
        else:
            self.self_play_flag = False
            self.random_play_flag = True

        if self.self_play_flag:
            self.begin_self_play = True
        else:
            self.begin_self_play = False

        self._read_history_models() # read history models from dir
        self.last_epoch = 0
        # ICM intrinsic reward 
        self.icm = icm_agent
        self.icm_buffer = []

    def _read_history_models(self):
        
        patten = re.compile(r'actor_(?P<index>\d+)')
        files = os.listdir(self.load_dir)
        for file in files:
            index = patten.findall(file)
            if len(index) > 0 :
                self.save_index.append(int(index[0]))
        self.save_index.sort() # from low to high sorting
        self.model_score = torch.ones(len(self.save_index), dtype=torch.float64) # initialize scores 
        if self.id == 0:
            print(f'model_score: {self.model_score}')

    def _set_actions_map(self, action_num):
        #dicretise action space
        forces = np.linspace(-100, 200, num=int(np.sqrt(action_num)), endpoint=True)
        thetas = np.linspace(-10, 10, num=int(np.sqrt(action_num)), endpoint=True)
        actions = [[force, theta] for force in forces for theta in thetas]
        actions_map = {i:actions[i] for i in range(action_num)}
        return actions_map
    
    def _wrapped_action(self, action, opponent_action):

        if isinstance(self.action_space, Discrete):
            real_action = self.actions_map[action]
            real_opponent_action = self.actions_map[opponent_action]
        elif isinstance(self.action_space, Box):
            action = np.clip(action, -1, 1)
            opponent_action = np.clip(opponent_action, -1, 1)
            high = self.action_space.high
            low = self.action_space.low
            real_action = low + 0.5*(action + 1.0)*(high - low)
            real_opponent_action = low + 0.5*(opponent_action + 1.0)*(high - low)
        wrapped_action = [[real_action[0]], [real_action[1]]]
        wrapped_opponent_action = [[real_opponent_action[0]], [real_opponent_action[1]]]
        wrapped_action= [wrapped_action, wrapped_opponent_action]

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

    def rollout(self, epochs):
        
        agent_index = np.random.randint(2)
        agent_index = 0
        ep_len = 0
        ep_ret = 0
        start_time = time.time()
        episode = 0
        epoch_reward = []
        raw_o = self.env.reset()
        action_end = False
        o = self.obs_transform(raw_o, None)
    # Main loop: collect experience in env and update/log each epoch
        for epoch in range(epochs):

            # this for self-play trigger from random-play
            if (self.load_index+epoch) > self.randomplay_epoch and not self.begin_self_play:
                self.begin_self_play = True
                self.self_play_flag = True
                self.last_epoch = 0

            # sample model from the model pool
            if self.begin_self_play:
                
                # with 0.8 probability sample the lateset model and 0.2 probability sample historical model 
                p = np.random.rand(1)
                if p > 0.2:
                    opponent_number = len(self.save_index) - 1   
                    sample_distribution = None                 
                else:
                    sample_distribution = Categorical(logits=self.model_score)
                    opponent_number = sample_distribution.sample()
                load_path = os.path.join(self.load_dir, f'actor_{self.save_index[opponent_number]}.pth')
                self.opponent = rl_agent([4, 30, 30], self.action_space, self.device)
                while not (Path(load_path).exists()):
                    pass 
                self.opponent.load_model(load_path)
                print(f'{proc_id()}:load actor_{self.save_index[opponent_number]} as opponent')

            t = 0
            g = 0
            stored_temp = False ## 该标志位表示是否存储了释放冰球时的临界经验数组
            while (t < self.local_steps_per_epoch):
                if self.render:
                    self.env.env_core.render()
                if o['obs'][0][-1].all() == 1: # 若全为-1，说明是另一个智能体在投掷冰球
                    who_is_throwing = 1
                else:
                    who_is_throwing = 0
                info_ctrl = o['info'][agent_index]
                info_oppo = o['info'][1-agent_index]
                obs_ctrl = o['obs'][agent_index]
                obs_oppo = o['obs'][1-agent_index]
                release = o['release'][agent_index]
                action_end = True if release or who_is_throwing != agent_index else False
                # 在没有过释放冰球前正常采取动作，释放冰球后，所有动作将无效
                if not action_end:
                    a, v, logp = self.policy.step(torch.as_tensor(obs_ctrl[newaxis], dtype=torch.float32, device=self.device),
                                                  torch.as_tensor(info_ctrl[newaxis], dtype=torch.float32, device=self.device))
                opponent_a = self.opponent.act(torch.as_tensor(obs_oppo[newaxis], dtype=torch.float32, device=self.device),
                                                  torch.as_tensor(info_oppo[newaxis], dtype=torch.float32, device=self.device))
                if action_end:
                    a = opponent_a
                env_a = self._wrapped_action(action_check(a), action_check(opponent_a))
                raw_next_o, r, d, info_before, info_after = self.env.step(env_a)
                next_o = self.obs_transform(raw_next_o, o)
                # 更新release状态
                next_release = next_o['release'][agent_index] 
                next_action_end = True if next_release else False

                # collect data for icm training ; estimate intrinsic reward
                if not action_end:
                    icm_data = {'obs':torch.as_tensor(obs_ctrl[-1][newaxis], dtype=torch.float32, device=self.device), 
                                'action':torch.as_tensor(a, dtype=torch.long, device=self.device), 
                                'next_obs': torch.as_tensor(next_o['obs'][agent_index][-1][newaxis], dtype=torch.float32, device=self.device)}
                    # 加入buffer中
                    self.icm_buffer.append(icm_data)


                # 记忆临界状态
                if next_action_end and not stored_temp and who_is_throwing == agent_index:
                    temp_experience = [obs_ctrl, info_ctrl, a, r[agent_index], v, logp]
                    stored_temp = True 
                if info_after == 'round_end' or info_after == 'game1_end' or info_after == 'game2_end':
                    false_d = True
                    stored_temp = False  # False 
                else: false_d = False
                ep_ret += r[agent_index]
                ep_len += 1 

                # save and log
                # 只存储非临界状态前的经验数组
                if not stored_temp and who_is_throwing == agent_index:
                    # 冰球得到延迟奖励后，更新临界时的经验数组，并存储到buffer中
                    if action_end:
                        temp_experience[3] = r[agent_index]
                        self.buffer.store(*(temp_experience))
                        self.logger.store(VVals=temp_experience[4])
                        t += 1
                        # if proc_id() == 2:
                        #     print(proc_id(), 'buffer:', self.buffer.ptr-self.buffer.path_start_idx)
                    else:
                        self.buffer.store(obs_ctrl, info_ctrl, a, r[agent_index], v, logp)
                        self.logger.store(VVals=v)
                        t += 1
                        # if proc_id() == 2:
                        #     print(proc_id(), 'buffer:', self.buffer.ptr-self.buffer.path_start_idx)

                o = next_o
                epoch_ended = t==(self.local_steps_per_epoch)
                if d or epoch_ended:
                    if epoch_ended and not(d):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        _, v, _ = self.policy.step(torch.as_tensor(o['obs'][agent_index][newaxis], dtype=torch.float32, device=self.device),
                                                   torch.as_tensor(o['info'][agent_index][newaxis], dtype=torch.float32, device=self.device) )
                    else:
                        v = 0
                    # icm collect and estimate
                    self.icm.collect_data(self.icm_buffer)
                    self.icm.estimate(self.icm_buffer)
                    intrinsic_reward = [self.icm_buffer[i]['intrinsic_reward'] for i in range(len(self.icm_buffer))]
                    self.icm_buffer.clear()
                    self.buffer.finish_path(v, intrinsic_reward)
                    if d:
                        episode +=1
                        epoch_reward.append(ep_ret)
                        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                        win_is = 1 if ep_ret >= 100 else 0 
                        self.logger.store(Win=win_is)
                        # 每一轮投掷结束之后，obs序列重置
                    ep_ret, ep_len =  0, 0
                if false_d:
                    o = self.obs_transform(raw_next_o, None)
                if d:
                    o = self.env.reset()
                    o = self.obs_transform(o, None)

            self.icm.train()
            self.icm_buffer.clear()
            data = self.buffer.get()
            # update policy
            self.policy.learn(data)
            # # Log info about epoch
            if (epoch+1) % self.log_interval == 0:
                self.logger.log_tabular('Win', average_only=True)
                self.logger.log_tabular('EpRet', with_min_and_max=True)
                self.logger.log_tabular('EpLen', average_only=True)
                self.logger.log_tabular('Epoch', epoch)
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
                self.logger.log_tabular('IcmForwardLoss', average_only=True)
                self.logger.log_tabular('IcmInverseLoss', average_only=True)
                self.logger.dump_tabular()

            # update the model score 
            # mean_win = np.mean(win_is)
            # if mean_win >= 0.5:
            #     if sample_distribution is not None: 
            #         self.model_score[opponent_number] -= 0.01 / (len(self.save_index) * sample_distribution.probs[opponent_number]) * (mean_win-0.5)
            #     else:
            #         self.model_score[opponent_number] -= 0.01 * (mean_win-0.5)
            # if self.id == 0:
            #     print("model_score", self.model_score)

            if epoch % self.save_interval == 0 or epoch == (epochs-1) and epoch > 0:
                sync_params(self.policy.ac)
                sync_params(self.icm.reward_model)
                self.save_index.append(self.load_index+epoch)
                self.model_score = torch.cat((self.model_score, torch.tensor([1])))
                assert self.model_score.shape[0] == len(self.save_index), print('the score length is not equal with the model length')
                if self.id == 0:
                    self.policy.save_models(self.load_index+epoch)
                    self.icm.save_models(self.load_index+epoch)
            



                






