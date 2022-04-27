import pickle
import json
from pathlib import Path
import sys
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
from utils.arguments import read_args
from env.chooseenv import make
import torch
from collect import Runner
import numpy as np
from utils.log import init_log 
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi
from spinup.utils.mpi_tools import mpi_fork, proc_id
import pdb
import os

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


    runner = Runner(args, env, logger)

    runner.rollout()

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

