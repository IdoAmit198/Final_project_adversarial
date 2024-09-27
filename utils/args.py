# import sys
import argparse


def get_args(description=""):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-s', '--seed', type=int, default=42, help='Give random seed')
    parser.add_argument('-d', '--dataset',
                        choices=['cifar10', 'cifar100', 'flowers102', 'mnist', 'imagenet'],
                        help='Give the dataset name from the choices. Currently, only Cifar-10 is supported.', default='cifar10')
    parser.add_argument('-m', '--model_name', choices=['wide_resnet50_2', 'resnet18'] ,default='resnet18')
    parser.add_argument('-opt', '--optimizer', choices=['SGD', 'ADAM'], help='Choose an optimizer', default='SGD')
    parser.add_argument('-e', '--max_epochs', type=int, default=200, help='Give number of epochs for training')
    parser.add_argument('--pgd_num_steps', type=int, default=10, help='Number of PGD training iterations')
    parser.add_argument('--max_epsilon', type=int, default=8, help='Maximum epsilon value for adaptive adversarial training.\n\
                                                Notice that the value should be in the range of 0-255, and a value of x actually is x/255.')
    parser.add_argument('--epsilon_step_size', type=float, default=0.005, help='Epsilon step increcement for adaptive adversarial training')
    parser.add_argument('--re_introduce_prob', type=float, default=0.2, help='During re-introduce method, the probability to re-introduce smaller epsilon value.')
    # We aim to use a pre-defined scheduler which lower by 10 factor at 80, 140, 180.
    parser.add_argument('-bs', '--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('-lr', '--learning_rate', default=1e-1, type=float, help='learning rate for training')
    parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float, help='weight decay for training')
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument('-tm', '--train_method', choices=['train','adaptive', 're_introduce', 'eval'], help='Training method. \
                        Will be to decide whether to measure baseline, adaptive, and such.', default='re_introduce')
    parser.add_argument('--agnostic_loss', action='store_true', help='Use agnostic loss')
    # Eval epsilons args
    parser.add_argument('--eval_epsilons', action='store_true', help='Whether to evaluate the model on different epsilons.\n \
                                                                    Defaults to false, and will train the model.')
    parser.add_argument('--sanity_check', action='store_true', help='Whether we make a sanity check now. If we do, it changes the save dir.\n \
                                                                    Defaults to false.')
    parser.add_argument('--eval_epsilon_max', type=int, default=32, help='Maximum epsilon value for evaluation of trained models')
    parser.add_argument('--eval_model_path', type=str, help='Path for the model to be evaluated.')
    parser.add_argument('--eval_uncertainty', action='store_true',
                        help='Whether to evaluate the uncertainty estimation abilities of the trained model.\n \
                        Uncertainty estimation will be assessed only using the test set.')
    args = parser.parse_args()
    return args