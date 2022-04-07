from dis import dis
from numpy import newaxis
import torch 
import numpy as np 
import time
import pdb
from copy import deepcopy
from gym.spaces import Box, Discrete
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
    # if distance <= 1:
    #     reward = 10
    # elif distance <= 5:
    #     reward = 5
    # elif distance <= 10:
    #     reward = 1
    # else:
    #     reward = 0
    reward = 1 / (distance + 1) * 1000# distance reward 

    return reward

class Runner:

    fix_forward_count = 21
    fix_forward_force = 50
    fix_backward_force = -100
    fix_backward_count = 24

    def __init__(self, args, env, agent, buffer, logger, device, 
                action_space, act_dim):
        
        self.total_epoch_step = args.epoch_step
        self.local_steps_per_epoch = int(args.epoch_step / args.cpu)
        self.save_interval = args.save_interval
        self.render =  args.render 
        self.env = env
        self.agent = agent 
        self.buffer = buffer
        self.logger = logger 
        self.ep_ret_history = [] 
        self.best_ep_ret = -np.inf
        self.device = device
        self.action_space = action_space
        self.act_dim = act_dim
        self.id = proc_id()
        self.actions_map = self._set_actions_map(args.action_num)
        self.goal = [300, 500]
        self.continue_train = True # if stop training 
        self.agent.set_agent_idx(0)
        self.agent.set_goal(self.goal)

    def _set_actions_map(self, action_num):
        #dicretise action space
        forces = np.linspace(-100, 200, num=int(np.sqrt(action_num)), endpoint=True)
        thetas = np.linspace(-10, 10, num=int(np.sqrt(action_num)), endpoint=True)
        actions = [[force, theta] for force in forces for theta in thetas]
        actions_map = {i:actions[i] for i in range(action_num)}
        return actions_map
    
    def _wrapped_action(self, action, who_is_throwing):
        # 根据当前回合是谁在投掷冰球来设计动作；无意义的一方的动作为零向量
        wrapped_action = []
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
            return [wrapped_action, wrapped_opponent_action]
        else:
            return [wrapped_opponent_action, wrapped_action]

    def obs_transform(self,obs):
    
        ob_ctrl = np.array(obs[0]['obs'])
        ob_oppo = np.array(obs[1]['obs'])

        release_ctrl = obs[0]['release']
        release_oppo = obs[1]['release']

        throws_left = obs[0]['throws left']
        
        obs_all = np.array([ob_ctrl, ob_oppo])
        release_all = np.array([release_ctrl, release_oppo])
        dict = {'obs':obs_all, 'release':release_all, 'throws_left':throws_left}

        return dict 

    def rollout(self, epochs):

        best_ret = -np.inf
        ep_len = 0
        ep_ret = 0
        start_time = time.time()
        episode = 0
        epoch_reward = []
        raw_o = self.env.reset()
        action_end = False
        o = self.obs_transform(raw_o)
    # Main loop: collect experience in env and update/log each epoch
        for epoch in range(epochs):
            if not self.continue_train:
                break
            t = 0
            stored_temp = False ## 该标志位表示是否存储了释放冰球时的临界经验数组
            while (t < self.local_steps_per_epoch):
                if self.render:
                    self.env.env_core.render()
                if o['obs'][0].all() == 1: # 若全为-1，说明是另一个智能体在投掷冰球
                    who_is_throwing = 1
                else:
                    who_is_throwing = 0

                while self.agent.stage == 0:
                    obs = o['obs'][who_is_throwing]
                    release = o['release'][who_is_throwing]
                    throws_left = o["throws_left"]
                    a = self.agent.choose_action(obs,'purple',throws_left)
                    env_a = [[[a[0]], [a[1]]], [[0],[0]]] if who_is_throwing == 0 else [[[0], [0]], [[a[0]],[a[1]]]]
                    self.agent.step([a[0], a[1]])
                    raw_next_o, _, d, pos_info, info = self.env.step(env_a)
                    next_o = self.obs_transform(raw_next_o)
                    o = next_o
                    if self.render:
                        self.env.env_core.render()
                obs = o['obs'][who_is_throwing]
                release = o['release'][who_is_throwing]
                throws_left = o["throws_left"]
                action_end = True if release else False # 冰球投掷出去后，不受动作控制
                # 在没有过释放冰球前正常采取动作，释放冰球后，所有动作将无效
                if not action_end:
                    extra_info = self.agent.get_info()
                    a, v, logp = self.agent.choose_action(obs,'purple',throws_left, goal=self.goal)
                    self.agent.step(self.actions_map[a.item()])
                    env_a = self._wrapped_action(a.item(), who_is_throwing)
                raw_next_o, _, d, pos_info, info = self.env.step(env_a)
                next_o = self.obs_transform(raw_next_o)
                # 更新release状态
                next_release = next_o['release'][who_is_throwing] 
                next_action_end = True if next_release else False
                # 记忆临界状态
                if next_action_end and not stored_temp:
                    temp_experience = [self.agent.obs_sequence, extra_info, a, 0, v, logp]
                    stored_temp = True 
                if info == 'round_end' or info == 'game1_end' or info == 'game2_end' :
                    false_d = True
                    stored_temp = False  # False 
                else: false_d = False
                if false_d:
                    # 根据预期目标点和pos计算奖励
                    reward = get_reward(pos_info, self.goal)
                else: reward = 0

                ep_ret += reward

                # save and log
                # 只存储非临界状态前的经验数组
                if not stored_temp:
                    # 冰球得到延迟奖励后，更新临界时的经验数组，并存储到buffer中
                    if action_end:
                        temp_experience[3] = reward
                        self.buffer.store(*(temp_experience))
                        self.logger.store(VVals=temp_experience[4])
                        t += 1
                    else:
                        self.buffer.store(self.agent.obs_sequence, extra_info, a, reward, v, logp)
                        self.logger.store(VVals=v)
                        t += 1
                ep_len += 1
                o = next_o
                epoch_ended = t==(self.local_steps_per_epoch)
                obs = o['obs'][who_is_throwing]
                if false_d or epoch_ended:
                    if epoch_ended and not(false_d):
                        print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                    # if trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        _, v, _ = self.agent.choose_action(obs, 'purple', throws_left, goal=self.goal)
                    else:
                        v = 0
                    self.buffer.finish_path(v)
                    if false_d:
                        episode +=1
                        epoch_reward.append(ep_ret)
                        self.logger.store(EpRet=ep_ret, EpLen=ep_len)
                        raw_o = self.env.reset()
                        o = self.obs_transform(raw_o)
                        self.agent.reset()
                        ep_ret, ep_len =  0, 0

            data = self.buffer.get()
            # update policy
            self.agent.policy.learn(data)
            # goalx = np.random.randint(250, 350)
            # goaly = np.random.randint(450, 550)
            # self.goal = [goalx, goaly]
            # self.agent.set_goal(self.goal)
            # Log info about epoch
            avg_KL, _ = mpi_statistics_scalar(self.logger.epoch_dict['KL'])
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
            avg_ret,_ = mpi_statistics_scalar(np.array(epoch_reward))
            epoch_reward = []
            # 保存最好的模型
            if best_ret < avg_ret:
                best_ret = avg_ret
                sync_params(self.agent.policy.ac)
                if self.id == 0:
                    self.agent.policy.save_models()
            # # KL收敛后则停止训练
            if avg_KL <= 0.0001 or epoch == (epochs-1):
                self.continue_train = False
                # 记录最好的奖励值到文件中
                if self.id == 0:
                    write_to_file('recorde.txt', self.goal, best_ret)
            if epoch % self.save_interval == 0 and epoch > 0 or not self.continue_train:
                sync_params(self.agent.policy.ac)
                if self.id == 0:
                    self.agent.policy.save_models(index=epoch)





                






