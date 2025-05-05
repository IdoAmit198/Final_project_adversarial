# import sys
import argparse


def get_args(description=""):
    parser = argparse.ArgumentParser(description=description)
    # parser.add_argument('--Train', action='store_false', help='Whether to train a model or not.')
    parser.add_argument('--Train', action='store_true', help='Whether to train a model or not.')
    parser.add_argument('--Inference', action='store_true', help='Whether to evaluate the model on different epsilons. Defaults to True.')
    parser.add_argument('--fine_tune', type=str, default=None, choices=['clean', 'adversarial', None],
                        help='Whether to fine-tune a clean or an adversarial checkpoint model. Defaults to None, which does not fine-tune at all.')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Give random seed')
    parser.add_argument('-d', '--dataset',
                        choices=['cifar10', 'cifar100', 'flowers102', 'mnist', 'imagenet', 'imagenet100'],
                        help='Give the dataset name from the choices. Currently, only Cifar-10 is supported.', default='imagenet')
    parser.add_argument('--augment', action='store_true', help='Whether to augment the training data')
    parser.add_argument('--use_clean_loss', action='store_true', help='Whether to use clean loss')
    parser.add_argument('-m', '--model_name', help=' model_name could be one of the following:\nWideResNet28_10, WideResNet34_10, WideResNet34_20, resnet18, resnet34, preact_resnet18, resnet50\n' \
    '                           Or of the following `robust_bench:<name>` / timm_<name>' ,default='resnet50')
    parser.add_argument('-opt', '--optimizer', choices=['SGD', 'ADAM'], help='Choose an optimizer', default='SGD')
    parser.add_argument('--scheduler', type=str, choices=['MultiStepLR', 'WarmupCosineLR', 'CyclicLR', 'CosineAnnealingWarmRestarts'],
                        help='The scheduler type for the learning rate.', default='WarmupCosineLR')
    parser.add_argument('--warmup_ratio', type=float, default=0.5, help='The warmup ratio for the WarmupCosineLR scheduler. A float between 0 and 1.')
    parser.add_argument('-e', '--max_epochs', type=int, default=50, help='Give number of epochs for training')
    parser.add_argument('--pgd_num_steps', type=int, default=2, help='Number of PGD training iterations')
    parser.add_argument('--pgd_step_size_factor', type=float, default=1.0, help='Step size factor for PGD training')
    parser.add_argument('--max_epsilon', type=int, default=16, help='Maximum epsilon value for adaptive adversarial training.\n\
                                                Notice that the value should be in the range of 0-255, and a value of x actually is x/255.')
    parser.add_argument('--epsilon_step_size', type=float, default=1/255, help='Epsilon step increcement for adaptive adversarial training')
    # parser.add_argument('--re_introduce_prob', type=float, default=0.2, help='During re-introduce method, the probability to re-introduce smaller epsilon value.')
    parser.add_argument('-bs', '--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('-lr', '--learning_rate', default=1e-1, type=float, help='learning rate for training')
    parser.add_argument('-wd', '--weight_decay', default=5e-4, type=float, help='weight decay for training')
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument('-tm', '--train_method', choices=['train','adaptive', 're_introduce', 'eval'], help='Training method. \
                        Will be to decide whether to measure baseline, adaptive, and such.', default='re_introduce')
    parser.add_argument('--agnostic_loss', action='store_true', help='Use agnostic loss')
    # Eval epsilons args
    parser.add_argument('--sanity_check', action='store_true', help='Whether we make a sanity check now. If we do, it changes the save dir.\n \
                                                                    Defaults to false.')
    parser.add_argument('--eval_epsilon_max', type=int, default=32, help='Maximum epsilon value for evaluation of trained models')
    parser.add_argument('--eval_model_path', type=str, help='Path for the model to be evaluated.', default=None)
    parser.add_argument('--eval_uncertainty', action='store_true',
                        help='Whether to evaluate the uncertainty estimation abilities of the trained model. Defauled to True.\n \
                        Uncertainty estimation will be assessed only using the test set.')
    # parser.add_argument('--ATAS', action='store_false', help='Whether to use ATAS or not.')
    # GradAlign args
    parser.add_argument('--GradAlign', action='store_true', help='Whether to use GradAlign or not.')
    parser.add_argument('--grad_align_lambda', default=2.0, type=float, help='regularization hyper-parameter for GradAlign')
    #
    # ATAS args
    parser.add_argument('--ATAS', action='store_true', help='Whether to use ATAS or not.')
    parser.add_argument('--atas_beta', default=0.5, type=float, help='hardness momentum')
    parser.add_argument('--atas_c', default=0.1, type=float, help='constant for ATAS')
    parser.add_argument('--atas_max_step_size', default=8/255, type=float, help='maximum perturb step size')
    parser.add_argument('--atas_min_step_size', default=4/255, type=float, help='minimum perturb step size')
    # AutoAttack inference args
    parser.add_argument('--AutoAttackInference', action='store_true', help='Whether to evaluate the model using AutoAttack.')
    parser.add_argument('--aa_attacks_list', nargs="+", type=str, default=None,
                        help='List of AutoAttack attacks to be used. Defaults to apgd-ce.\n\
                            Full list of avilable attacks can be found in the AutoAttack repository:\n\
                            https://github.com/fra31/auto-attack/tree/master')
    parser.add_argument('--checkpoint_author', type=str, default=None,
                        help='The authors of the checkpoint model. Used for naming the saved model and later for comparison.')
    args = parser.parse_args()
    return args