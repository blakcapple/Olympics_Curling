import argparse

def read_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="olympics-curling", type=str)
    parser.add_argument('--algo', default="ppo", type=str, help="ppo/sac")
    parser.add_argument('--train_epoch', default=10000, type=int)
    parser.add_argument('--randomplay_epoch', default=2000, type=int, help='the opponent is random agent')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--load', default=False, action='store_true')
    parser.add_argument('--load_dir', default='load_model', type=str)
    parser.add_argument('--load_index', default=0, type=int, help='which index to load')
    parser.add_argument('--load_opponent_index', default=0, type=int, help='which index to load')
    parser.add_argument('--action_type', default=0, type=int, help='1 is continuous and 0 is discrete')
    parser.add_argument('--action_num', default=49, type=int, help='how many different actions')
    parser.add_argument('--render', default=False, action='store_true')
    # ppo parameters
    parser.add_argument('--pi_lr', default=3e-4, type=float)
    parser.add_argument('--v_lr', default=5e-4, type=float)
    parser.add_argument('--mini_batch', default=3, type=int)
    parser.add_argument('--entropy_c', default=0.0, type=float)
    parser.add_argument('--train_pi_iters', default=10, type=int)
    parser.add_argument('--train_v_iters', default=10, type=int)
    parser.add_argument('--lamda', default=0.97, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--target_kl', default=0.03, type=float)
    parser.add_argument('--clip_ratio', default=0.2, type=float)
    parser.add_argument('--max_grad_norm', default=0.5, type=float)
    parser.add_argument('--epoch_step', default=900, type=int)
    parser.add_argument('--eval_step', default=1000, type=int)
    parser.add_argument('--save_dir', default='data', type=str)
    parser.add_argument('--save_name', default=None, type=str)
    # interval parameters
    parser.add_argument('--save_interval', default=50, type=int)
    parser.add_argument('--eval_interval', default=100, type=int)
    parser.add_argument('--selfplay_interval', default=40, type=int)
    parser.add_argument('--randomplay_interval', default=10, type=int)


    args = parser.parse_args()

    return args
