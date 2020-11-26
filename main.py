import argparse

import numpy as np
import torch
import logging

from solver import Solver
from utils.utils import str2bool

import os
from utils.helpers import (get_config_section)

CONFIG_FILE = "hyperparam.ini"
LOG_LEVELS = list(logging._levelToName.values())

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(arguments):
    seed = arguments.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = Solver(arguments)

    if arguments.train:
        net.train()
    else:
        net.traverse()


if __name__ == "__main__":

    default_config = get_config_section([CONFIG_FILE], "Custom")

    parser = argparse.ArgumentParser(description='toy Beta-VAE')

    parser.add_argument('-L', '--log-level', help="Logging levels.",
                        default=default_config['log_level'], choices=LOG_LEVELS)

    parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=1e12, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--random_percentage', default=0.2, type=float, help='random_percentage')

    parser.add_argument('--second_layer_C', default=False, type=str2bool, help='add a linear layer for L3 classidier')

    parser.add_argument('--latent_spec_cont', type=int, default=None, metavar='integer value',
                        help='Capacity of continue latent space')
    parser.add_argument('--latent_spec_disc', nargs='+', type=int, default=None, metavar='integer list',
                        help='Capacity of discrete latent space: 2 variables, the first for one-hot vector number and '
                             'the second for one-hot vector size')
    parser.add_argument('--latent_spec_cont_var', type=int, default=None, metavar='integer value',
                        help='Capacity of continue latent space')
    parser.add_argument('--latent_spec_cont_class', type=int, default=None, metavar='integer value',
                        help='Capacity of continue latent space')
    parser.add_argument('--latent_spec_disc_var', nargs='+', type=int, default=None, metavar='integer value',
                        help='Capacity of discret latent space')
    parser.add_argument('--latent_spec_disc_class', nargs='+', type=int, default=None, metavar='integer value',
                        help='Capacity of discret latent space')

    parser.add_argument('--L3_without_random', default=False, type=str2bool, help='use classification without random')
    parser.add_argument('--is_C', default=True, type=str2bool,
                        help='use classificatino regularizer with random continue latent space')
    parser.add_argument('--is_partial_rand_class', default=False, type=str2bool,
                        help='use classificatino regularizer with partial random class latent space')
    parser.add_argument('--is_E1', default=False, type=str2bool,
                        help='use encoder E1')
    parser.add_argument('--is_binary_structural_latent', default=False, type=str2bool,
                        help='use binary_structural_latent')
    parser.add_argument('--is_zvar_sim_loss', default=False, type=str2bool, help='use zvar_sim_loss')
    parser.add_argument('--BN', default=False, type=str2bool, help='use BN in model')
    parser.add_argument('--E1_conv', default=False, type=str2bool, help='use E1_conv in model')
    parser.add_argument('--E1_dense', default=False, type=str2bool, help='use E1_dense in model')
    parser.add_argument('--is_noise_stats', default=False, type=str2bool, help='Compute stat with bits noised '
                                                                               'during score')
    parser.add_argument('--zvar_sim_loss_only_for_encoder', default=False, type=str2bool, help='If the zvar_sim_loss loss is propagate for'
                                                                                     ' only encoder or for both encoder '
                                                                                     'and decoder')
    parser.add_argument('--zvar_sim_loss_for_all_model', default=False, type=str2bool, help='If the zvar_sim_loss loss is propagate in'
                                                                                     ' all the model')

    # --- Params to VAnilla VAE test:
    parser.add_argument('--hidden_filters_layer1', type=int, default=32, metavar='integer value',
                        help='hidden_filters_layer1')
    parser.add_argument('--hidden_filters_layer2', type=int, default=32, metavar='integer value',
                        help='hidden_filters_layer2')
    parser.add_argument('--hidden_filters_layer3', type=int, default=32, metavar='integer value',
                        help='hidden_filters_layer3')
    parser.add_argument('--stride_size', type=int, default=2, metavar='integer value',
                        help='stride_size')
    parser.add_argument('--kernel_size', type=int, default=4, metavar='integer value',
                        help='kernel_size')
    # ------------------------------------

    parser.add_argument('--is_perturbed_score', default=False, type=str2bool, help='Compute stat with z perturbed '
                                                                                                'during score')
    parser.add_argument('--W_Lzvar_sim', default=1, type=float, help='W_zvar_sim_loss parameter for zvar_sim_loss loss')
    parser.add_argument('--beta', default=1, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--W_Lc', default=1, type=float, help='W_L3 parameter for classification loss')
    parser.add_argument('--W_Lr', default=1, type=float, help='Wreconstruction parameter for reconstruction loss')
    parser.add_argument('--W_recon_wt_rand', default=1, type=float,
                        help='W_recon_wt_rand parameter for reconstruction loss')
    parser.add_argument('--W_Kl_var', default=1, type=float, help='W_Kl_cont parameter for Kl_cont loss')
    parser.add_argument('--W_Kl_struct', default=1, type=float, help='W_Kl_disc parameter for Kl_disc loss')
    parser.add_argument('--W_Lpc', default=1, type=float, help='W_class_partial_rand parameter for loss')
    parser.add_argument('--gamma', default=1000, type=float, help='gamma parameter for KL-term in understanding '
                                                                  'beta-VAE')
    parser.add_argument('--C_max', default=25, type=float, help='capacity parameter(C) of bottleneck channel')
    parser.add_argument('--C_stop_iter', default=1e5, type=float, help='when to stop increasing the capacity')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')
    parser.add_argument('--dataset', default='mnist', type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')

    parser.add_argument('--expe_name', default='main', type=str, help='expes name')

    parser.add_argument('--display_step', default=10000, type=int, help='number of iterations after which loss data is'
                                                                        ' printed and visdom is updated')
    parser.add_argument('--save_step', default=10000, type=int, help='number of iterations after which a checkpoint is'
                                                                     ' saved')
    parser.add_argument('--just_train', default=False, type=str2bool, help='if it is just for training without save')

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_dir_scores', default='checkpoints_scores', type=str, help='checkpoint sample_scores '
                                                                                          'directory')

    parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="GPU devices available")

    args = parser.parse_args()

    print(args)

    if not args.just_train:
        # save arguments parser:
        with open('args_parser/commandline_args_test' + args.expe_name + '.txt', 'w') as f:
            f.write(str(str(args).split('Namespace(')[1]).replace(', ', '\n').replace(')', ''))

    if args.gpu_devices is not None:
        gpu_devices = ','.join([str(idx) for idx in args.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        print('CUDA Visible devices !')

    main(args)
