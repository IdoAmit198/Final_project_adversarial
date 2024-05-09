# import sys
import argparse


def get_args(description=""):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-s', '--seed', type=int, default=0, help='Give random seed')
    # parser.add_argument('-nw', '--num_workers', type=int, default=0, help='Number of workers to fetch samples from memory.')
    parser.add_argument('-d', '--dataset',
                        choices=['food101', 'flowers102', 'dtd', 'fgvcaircraft', 'imagenet', 'stanfordcars', 'sun397'],
                        help='Give the dataset name from the choices.', default='cifar10')
    parser.add_argument('-m', '--model_name', default='wide_resnet50_2')
    parser.add_argument('-opt', '--optimizer', choices=['SGD', 'ADAM'], help='Choose an optimizer', default='SGD')
    parser.add_argument('-e', '--max_epochs', type=int, default=200, help='Give number of epochs for training')
    parser.add_argument('--pgd_num_steps', type=int, default=20, help='Number of PGD training iterations')
    parser.add_argument('--max_epsilon', type=float, default=8/255, help='Maximum epsilon value for adaptive adversarial training')
    parser.add_argument('--epsilon_step_size', type=float, default=0.005, help='Epsilon step increcement for adaptive adversarial training')
    # We aim to use a pre-defined scheduler which lower by 10 factor at 80, 140, 180.
    parser.add_argument('-bs', '--batch_size', type=int, default=256,
                        help='Batch size.')

    # parser.add_argument('--device', type=str, choices=['cuda:0', 'cpu'], help='GPU or CPU ', default='cuda:0')

    parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help='learning rate for training')
    parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float, help='weight decay for training')
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument('-tm', '--train_method', choices=['train','adaptive', 'eval'], help='Training method. \
                        Will be to decide whether to measure baseline, adaptive, and such.', default='train')

    args = parser.parse_args()
    return args