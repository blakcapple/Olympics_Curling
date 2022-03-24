
import numpy as np
import torch
import random
from agents.rl.rl_model import agent, agent_base
from env.chooseenv import make
from tabulate import tabulate
import argparse
from torch.distributions import Categorical
import os
from gym.spaces import Box
import pdb

def get_join_actions(state, algo_list):
    
    joint_actions = []

    for agent_idx in range(len(algo_list)):
        if algo_list[agent_idx] == 'random':
            driving_force = random.uniform(-100, 200)
            turing_angle = random.uniform(-10, 10)
            joint_actions.append([[driving_force], [turing_angle]])

        elif algo_list[agent_idx] == 'rl_base':

            obs = state[agent_idx]['obs']
            info = get_info(state, agent_idx)
            obs = np.array(obs)
            actions_raw = agent_base.choose_action(obs, info, True)
            if agent_base.is_act_continuous:
                actions_raw = actions_raw.detach().cpu().numpy().reshape(-1)
                action = np.clip(actions_raw, -1, 1)
                high = agent_base.action_space.high
                low = agent_base.action_space.low
                actions = low + 0.5*(action + 1.0)*(high - low)
            else:
                actions = agent_base.actions_map[actions_raw.item()]
            joint_actions.append([[actions[0]], [actions[1]]])

        elif algo_list[agent_idx] == 'rl':
            obs = state[agent_idx]['obs']
            info = get_info(state, agent_idx)
            obs = np.array(obs)
            actions_raw = agent.choose_action(obs, info, True)
            if agent.is_act_continuous:
                actions_raw = actions_raw.detach().cpu().numpy().reshape(-1)
                action = np.clip(actions_raw, -1, 1)
                high = agent.action_space.high
                low = agent.action_space.low
                actions = low + 0.5*(action + 1.0)*(high - low)
            else:
                actions = agent.actions_map[actions_raw.item()]
            joint_actions.append([[actions[0]], [actions[1]]])

    return joint_actions


RENDER = True

def get_info(obs, agent_index):

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
            info[10:12] = np.array(list(reversed(obs[0]['score']))) / 4
        if agent_index == 0:
            info[12] = 1
        else:
            info[13] = 1

        return info 

def run_game(env, algo_list, episode, verbose=False):
    total_reward = np.zeros(2)
    num_win = np.zeros(3)       #agent 1 win, agent 2 win, draw
    episode = int(episode)
    for i in range(1, int(episode)+1):
        episode_reward = np.zeros(2)

        state = env.reset()
        if RENDER:
            env.env_core.render()
        step = 0


        while True:
            joint_action = get_join_actions(state, algo_list)
            next_state, reward, done, _, info = env.step(joint_action)
            reward = np.array(reward)
            episode_reward += reward
            if RENDER:
                env.env_core.render()
            
            if any(reward) !=0 :
                print(reward)
                # pdb.set_trace()
            if done:
                if reward[0] != reward[1]:
                    if reward[0]==100:
                        num_win[0] +=1
                    elif reward[1] == 100:
                        num_win[1] += 1
                    else:
                        raise NotImplementedError
                else:
                    num_win[2] += 1

                if not verbose:
                    print('.', end='')
                    if i % 100 == 0 or i==episode:
                        print()
                break
            state = next_state
            step += 1
        total_reward += episode_reward
    total_reward/=episode
    print("total reward: ", total_reward)
    print('Result within {} episode:'.format(episode))

    header = ['Name', algo_list[0], algo_list[1]]
    data = [['score', np.round(total_reward[0], 2), np.round(total_reward[1], 2)],
            ['win', num_win[0], num_win[1]]]
    print(tabulate(data, headers=header, tablefmt='pretty'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", default='random', help='rl/random')
    parser.add_argument("--opponent", default='rl', help='rl/random')
    parser.add_argument("--episode", default=1)
    args = parser.parse_args()

    env_type = "olympics-curling"
    game = make(env_type, conf=None, seed = 1)

    # torch.manual_seed(1)
    # np.random.seed(1)
    # random.seed(1)

    agent_list = [args.opponent, args.my_ai]        #your are controlling agent green
    run_game(game, algo_list=agent_list, episode=args.episode, verbose=False)

