
import numpy as np
import torch
import random
from agents.rl.submission_rule import agent
# from agents.rl.submission_origin import agent as agent_base
# from agents.rl.submission_center import agent as agent_base
# from agents.rl.submission import agent as agent_base
from agents.rl.submission_rule_oppo import agent as agent_base 
# from agents.rl.submission_1 import agent as agent
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
            turing_angle = random.uniform(-30, 30)
            joint_actions.append([[driving_force], [turing_angle]])

        elif algo_list[agent_idx] == 'rl_base':

            obs = state[agent_idx]['obs']
            index = state[agent_idx]['controlled_player_index']
            throws_left = state[agent_idx]['throws left']
            color = state[agent_idx]['team color']
            score_point = state[agent_idx]['score']
            game_round = state[agent_idx]['game round']
            agent_base.set_game_information(score_point, game_round)
            agent_base.set_agent_idx(index)
            obs = np.array(obs)
            actions = agent_base.choose_action(obs, throws_left, color, True)
            # actions = agent_base.choose_action(obs, throws_left, True)
            joint_actions.append([[actions[0]], [actions[1]]])
            agent_base.step([actions[0], actions[1]])

        elif algo_list[agent_idx] == 'rl':
            obs = state[agent_idx]['obs']
            index = state[agent_idx]['controlled_player_index']
            throws_left = state[agent_idx]['throws left']
            color = state[agent_idx]['team color']
            score_point = state[agent_idx]['score']
            game_round = state[agent_idx]['game round']
            agent.set_game_information(score_point, game_round)
            agent.set_agent_idx(index)
            obs = np.array(obs)
            actions = agent.choose_action(obs, throws_left, color, True)
            joint_actions.append([[actions[0]], [actions[1]]])
            agent.step([actions[0], actions[1]])

    return joint_actions


RENDER = True


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

            if done:
                # if reward[0] != reward[1]:
                #     if reward[0]==100:
                #         num_win[0] +=1
                #     elif reward[1] == 100:
                #         num_win[1] += 1
                #     else:
                #         raise NotImplementedError
                # else:
                #     num_win[2] += 1

                # if not verbose:
                #     print('.', end='')
                #     if i % 100 == 0 or i==episode:
                #         print()
                break
            state = next_state
            step += 1
        total_reward += episode_reward
    total_reward/=episode
    print("total reward: ", total_reward)
    print('Result within {} episode:'.format(episode))

    # header = ['Name', algo_list[0], algo_list[1]]
    # data = [['score', np.round(total_reward[0], 2), np.round(total_reward[1], 2)],
    #         ['win', num_win[0], num_win[1]]]
    # print(tabulate(data, headers=header, tablefmt='pretty'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", default='rl_base', help='rl/random')
    parser.add_argument("--opponent", default='rl', help='rl/random')
    parser.add_argument("--episode", default=1)
    args = parser.parse_args()

    env_type = "olympics-curling"
    game = make(env_type, conf=None, seed = 1)

    # torch.manual_seed(1)
    # np.random.seed(1)
    # random.seed(1)

    agent_list = [args.opponent, args.my_ai]        #your are controlling agent green
    # agent_list = [args.my_ai, args.opponent]
    run_game(game, algo_list=agent_list, episode=args.episode, verbose=False)

