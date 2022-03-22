import json
from pathlib import Path
from random import random
import sys
import pdb
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
from utils.arguments import read_args
from env.chooseenv import make
from algo.ppo import PPO
from algo.buffer import PPOBuffer
import torch
from runner import Runner
import numpy as np
import os
from utils.log import init_log 
from vec_env.schmem_vec_env import ShmemVecEnv
from algo.opponent import rl_agent, random_agent
import wandb
from gym.spaces import Box, Dict, Discrete
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params
from spinup.utils.mpi_tools import mpi_fork, proc_id, num_procs


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def main(args):

    ## set logger
    logger_kwargs = setup_logger_kwargs(args.algo, args.seed, data_dir=args.save_dir)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    setup_pytorch_for_mpi()
    args.seed += 100 * proc_id()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = make(args.game_name, args.seed)

    state_shape = [4, 30, 30]
    action_num = args.action_num
    if args.action_type == 1:
        action_space = Box(low=np.array([-100, -10]), high=np.array([200, 10]))
        act_dim = 2
    elif args.action_type == 0:
        action_space = Discrete(action_num)
        act_dim = 1

    device = 'cpu' # spinning up mpi tools only support cpu 
    print('device', device)

    local_epoch_step = int(args.epoch_step / args.cpu)
    policy = PPO(state_shape, action_space, pi_lr=args.pi_lr, v_lr=args.v_lr, device=device, 
                entropy_c = args.entropy_c,logger=logger, clip_ratio=args.clip_ratio, 
                train_pi_iters=args.train_pi_iters, train_v_iters=args.train_v_iters, 
                target_kl=args.target_kl, save_dir=args.save_dir, max_grad_norm=args.max_grad_norm, 
                max_size=int(local_epoch_step), batch_size=int(args.epoch_step/args.mini_batch))
    sync_params(policy.ac) # Sync params across processes
    logger.setup_pytorch_saver(policy.ac)
    
    # Count variables
    var_counts = tuple(count_vars(module) for module in [policy.ac.pi, policy.ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    info_dim = 14
    buffer = PPOBuffer(state_shape, act_dim, info_dim, local_epoch_step, device, args.gamma, args.lamda)
    if args.load:
        policy.load_models(args.load_dir, args.load_index)
        if args.load_opponent_index > 0:
            opponent = rl_agent(state_shape, action_space, device)
            load_path = os.path.join(args.load_dir, f'actor_{args.load_opponent_index}.pth')
            opponent.load_model(load_path)
        else:
            opponent = random_agent(action_space)
    else:
        opponent = random_agent(action_space)

    runner = Runner(args, env, policy, opponent, buffer, logger, device, action_space, act_dim)

    runner.rollout(args.train_epoch)

if __name__ == '__main__':
    args = read_args()
    logger, save_path, log_file = init_log(args.save_dir, args.save_name)
    with open(save_path+'/arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # with open(save_path+'/arguments.txt', 'r') as f:
    #     args.__dict__ = json.load(f)
    args.save_dir = save_path
    mpi_fork(args.cpu)
    main(args)

