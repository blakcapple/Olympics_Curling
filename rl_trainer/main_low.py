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
from runner_low import Runner
import numpy as np
from utils.log import init_log 
from gym.spaces import Box, Discrete
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params
from spinup.utils.mpi_tools import mpi_fork, proc_id, num_procs
from algo.agent import RLAgent


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def main(args):

    ## set logger
    logger_kwargs = setup_logger_kwargs(args.algo, args.seed, data_dir=args.save_path)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    setup_pytorch_for_mpi()
    args.seed += 100 * proc_id()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = make(args.game_name, args.seed)

    state_shape = [1, 30, 30]
    action_num = args.action_num
    if args.action_type == 1:
        action_space = Box(low=np.array([-100, -10]), high=np.array([200, 10]))
        act_dim = 2
    elif args.action_type == 0:
        action_space = Discrete(action_num)
        act_dim = 1
    device = 'cpu' # spinning up mpi tools only support cpu 
    local_epoch_step = int(args.epoch_step / args.cpu)
    info_dim = 7 #[goalx,goaly,vx,vy,theta,posx, posy]

    agent = RLAgent(state_shape, action_space, info_dim, device, mode='train')
    policy = PPO(state_shape, action_space, pi_lr=args.pi_lr, v_lr=args.v_lr, device=device, 
                entropy_c = args.entropy_c,logger=logger, clip_ratio=args.clip_ratio, 
                train_pi_iters=args.train_pi_iters, train_v_iters=args.train_v_iters, 
                target_kl=args.target_kl, save_path=args.save_path, max_grad_norm=args.max_grad_norm, 
                max_size=int(local_epoch_step), batch_size=int(args.epoch_step/args.mini_batch), 
                info_dim=info_dim)

    sync_params(policy.ac) # Sync params across processes
    logger.setup_pytorch_saver(policy.ac)
    agent.set_policy(policy)
    
    # Count variables
    var_counts = tuple(count_vars(module) for module in [policy.ac.pi, policy.ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    buffer = PPOBuffer(state_shape, act_dim, info_dim, local_epoch_step, device, args.gamma, args.lamda)    
    if args.load:
        agent.policy.load_models(args.load_dir, args.load_index)
        print('load_model')

    runner = Runner(args, env, agent, buffer, logger, device, action_space, act_dim)

    runner.rollout(args.train_epoch)

if __name__ == '__main__':
    args = read_args()
    logger, save_path, log_file = init_log(args.save_dir, args.save_name)
    with open(save_path+'/arguments.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    # with open(save_path+'/arguments.txt', 'r') as f:
    #     args.__dict__ = json.load(f)
    args.save_path = save_path
    mpi_fork(args.cpu)
    main(args)

