from curses import raw
from random import random
from numpy import newaxis
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
from spinup.utils.mpi_tools import num_procs, proc_id

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
        self.save_index = [] # the models pool
        self.id = proc_id()
        self.actions_map = self._set_actions_map(args.action_num)
        self.render = args.render 

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

    def obs_transform(self, obs, agent_index, obs_sequence_dict):
    
        ob_ctrl = obs[0]['obs'][0]
        ob_oppo = obs[1]['obs'][0]

        ob_ctrl = ob_ctrl.reshape(1, *ob_ctrl.shape)
        ob_oppo = ob_oppo.reshape(1, *ob_oppo.shape)

        release_ctrl = obs[0]['release']
        release_oppo = obs[1]['release']

        info = np.zeros(14) # record game infomation

        if obs[0]['game round'] == 0:
            info[0] = 1
        else:
            info[1] = 1

        if obs[0]['throws left'][agent_index] == 3:
            info[2] = 1
        elif obs[0]['throws left'][agent_index] == 2:
            info[3] = 1
        elif obs[0]['throws left'][agent_index] == 1:
            info[4] = 1
        elif obs[0]['throws left'][agent_index] == 0:
            info[5] = 1

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
            info[10:12] = np.array(obs[0]['score'].reverse) / 4
        if agent_index == 0:
            info[12] = 1
        else:
            info[13] = 1

        if obs_sequence_dict is not None:
            obs_sequence1 = np.concatenate((np.delete(obs_sequence_dict['obs'][0], 0, axis=0), ob_ctrl), axis=0)
            obs_sequence2 = np.concatenate((np.delete(obs_sequence_dict['obs'][1], 0, axis=0), ob_oppo), axis=0)
        else:
            obs_sequence1 = np.repeat(ob_ctrl, 4, axis=0)
            obs_sequence2 = np.repeat(ob_oppo, 4, axis=0)
        obs_all = np.append(obs_sequence1[np.newaxis], obs_sequence2[np.newaxis], axis=0)
        release_all = np.array([release_ctrl, release_oppo])
        dict = {'obs':obs_all, 'release':release_all, "info":info}
        return dict 

    def rollout(self, epochs):
        
        agent_index = 0
        ep_len = 0
        ep_ret = 0
        start_time = time.time()
        episode = 0
        epoch_reward = []
        raw_o = self.env.reset()
        action_end = False
        o = self.obs_transform(raw_o, agent_index, None)
    # Main loop: collect experience in env and update/log each epoch
        for epoch in range(epochs):
            t = 0
            stored_temp = False ## 该标志位表示是否存储了释放冰球时的临界经验数组
            while (t < self.local_steps_per_epoch):
                if self.render:
                    self.env.env_core.render()
                if o['obs'][0][-1].all() == 1: # 若全为-1，说明是另一个智能体在投掷冰球
                    who_is_throwing = 1
                else:
                    who_is_throwing = 0
                info = o['info']
                obs = o['obs'][agent_index]
                release = o['release'][agent_index]
                action_end = True if release or who_is_throwing != agent_index else False
                # 在没有过释放冰球前正常采取动作，释放冰球后，所有动作将无效
                if not action_end:
                    a, v, logp = self.policy.step(torch.as_tensor(obs[newaxis], dtype=torch.float32, device=self.device),
                                                  torch.as_tensor(info[newaxis], dtype=torch.float32, device=self.device))
                opponent_a = self.opponent.act(o['obs'][1-agent_index])
                env_a = self._wrapped_action(a[0], opponent_a)
                raw_next_o, r, d, info_before, info_after = self.env.step(env_a)
                next_o = self.obs_transform(raw_next_o, agent_index, o)
                # 更新release状态
                next_release = next_o['release'][agent_index] 
                next_action_end = True if next_release else False
                # 记忆临界状态
                if next_action_end and not stored_temp:
                    temp_experience = [obs, info, a, r[agent_index], v, logp]
                    stored_temp = True 
                if any(r) !=0 :
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
                    else:
                        self.buffer.store(obs, info, a, r[agent_index], v, logp)
                        self.logger.store(VVals=v)
                        t += 1

                o = next_o
                epoch_ended = t==(self.local_steps_per_epoch)
                if d or epoch_ended:
                    if epoch_ended and not(d):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        _, v, _ = self.policy.step(torch.as_tensor(o['obs'][agent_index][newaxis], dtype=torch.float32, device=self.device),
                                                   torch.as_tensor(o['info'][newaxis], dtype=torch.float32, device=self.device) )
                    else:
                        v = 0
                    self.buffer.finish_path(v)
                    if d:
                        episode +=1
                        epoch_reward.append(ep_ret)
                        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                        # 每一轮投掷结束之后，obs序列重置
                    ep_ret, ep_len =  0, 0
                if false_d:
                    o = self.obs_transform(raw_next_o, agent_index, None)
                if d:
                    o = self.env.reset()
                    o = self.obs_transform(o, agent_index, None)
            data = self.buffer.get()
            # update policy
            self.policy.learn(data)
            # Log info about epoch
            self.logger.log_tabular('Epoch', epoch)
            self.logger.log_tabular('EpRet', with_min_and_max=True)
            self.logger.log_tabular('EpLen', average_only=True)
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
            # wandb.log({'Reward':np.mean(epoch_reward)}, step=epoch)
            if epoch % self.save_interval == 0 or epoch == (epochs-1) and epoch > 0:
                sync_params(self.policy.ac)
                if self.id == 0:
                    self.policy.save_models(index=epoch)




                






