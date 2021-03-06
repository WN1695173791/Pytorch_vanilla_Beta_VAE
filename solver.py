import warnings
import os
from tqdm import tqdm
import logging
import torch
import torch.optim as optimizer
import torch.nn.functional as F
import numpy as np

from models.model import BetaVAE
from dataset.dataset_2 import get_dataloaders
from scores import compute_scores

warnings.filterwarnings("ignore")
EPS = 1e-12


def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x)
    elif distribution == 'gaussian':
        # x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0

    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    # KLD is Kullback–Leibler divergence -- how much does one learned
    # distribution deviate from another, in this specific case the
    # learned distribution from the unit Gaussian

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # note the negative D_{KL} in appendix B of the paper
    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    # Normalise by same number of elements as in reconstruction
    # KLD /= nb_pixels
    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    return total_kld


def _kl_multiple_discrete_loss(alphas):
    """
    Calculates the KL divergence between a set of categorical distributions
    and a set of uniform categorical distributions.
    Parameters
    ----------
    alphas : list
        List of the alpha parameters of a categorical (or gumbel-softmax)
        distribution. For example, if the categorical atent distribution of
        the model has dimensions [2, 5, 10] then alphas will contain 3
        torch.Tensor instances with the parameters for each of
        the distributions. Each of these will have shape (N, D).
    """
    # Calculate kl losses for each discrete latent
    kl_losses = [_kl_discrete_loss(alpha) for alpha in alphas]

    # Total loss is sum of kl loss for each discrete latent
    kl_loss = torch.sum(torch.cat(kl_losses))
    return float(kl_loss)


def _kl_discrete_loss(alpha):
    """
    Calculates the KL divergence between a categorical distribution and a
    uniform categorical distribution.
    Parameters
    ----------
    alpha : torch.Tensor
        Parameters of the categorical or gumbel-softmax distribution.
        Shape (N, D)
    """
    disc_dim = int(alpha.size()[-1])
    log_dim = torch.Tensor([np.log(disc_dim)])
    if torch.cuda.is_available():
        log_dim = log_dim.cuda()
    # Calculate negative entropy of each row
    neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
    # Take mean of negative entropy across batch
    mean_neg_entropy = torch.mean(neg_entropy, dim=0)
    # KL loss of alpha with uniform categorical variable
    kl_loss = log_dim + mean_neg_entropy
    return kl_loss


def compute_scores_and_loss(net, train_loader, valid_loader, device, latent_spec, train_loader_size,
                            test_loader_size, is_partial_rand_class, random_percentage, is_E1, is_zvar_sim_loss, is_C,
                            is_noise_stats, is_perturbed_score, zvar_sim_var_rand, zvar_sim_normal,
                            zvar_sim_change_zvar, old_weighted, E1_VAE, E1_AE):
    """
    compute some sample_scores and loss on data train and data set and save it for plot after training
    :return:
    """

    # print("Train")
    scores_train, loss_train, mean_proba_per_class_train, std_proba_per_class_train, \
    mean_proba_per_class_noised_train, \
    std_proba_per_class_noised_train = compute_scores(net,
                                                      train_loader,
                                                      device,
                                                      latent_spec,
                                                      train_loader_size,
                                                      is_partial_rand_class,
                                                      random_percentage,
                                                      is_E1,
                                                      is_zvar_sim_loss,
                                                      is_C,
                                                      is_noise_stats,
                                                      is_perturbed_score,
                                                      zvar_sim_var_rand,
                                                      zvar_sim_normal,
                                                      zvar_sim_change_zvar,
                                                      old_weighted,
                                                      E1_VAE,
                                                      E1_AE)
    # print("Test:")
    scores_test, loss_test, mean_proba_per_class_test, std_proba_per_class_test, \
    mean_proba_per_class_noised_test, \
    std_proba_per_class_noised_test = compute_scores(net,
                                                     valid_loader,
                                                     device,
                                                     latent_spec, test_loader_size,
                                                     is_partial_rand_class,
                                                     random_percentage,
                                                     is_E1,
                                                     is_zvar_sim_loss,
                                                     is_C,
                                                     is_noise_stats,
                                                     is_perturbed_score,
                                                     zvar_sim_var_rand,
                                                     zvar_sim_normal,
                                                     zvar_sim_change_zvar,
                                                     old_weighted,
                                                     E1_VAE,
                                                     E1_AE)

    return scores_train, scores_test, loss_train, loss_test, mean_proba_per_class_train, std_proba_per_class_train, \
           mean_proba_per_class_test, std_proba_per_class_test, mean_proba_per_class_noised_train, \
           std_proba_per_class_noised_train, mean_proba_per_class_noised_test, std_proba_per_class_noised_test


def gpu_config(model):
    use_gpu = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if use_gpu:
        if gpu_count > 1:
            print('use {} gpu who named:'.format(gpu_count))
            for i in range(gpu_count):
                print(torch.cuda.get_device_name(i))
            model = torch.nn.DataParallel(model, device_ids=[device_id for device_id in range(gpu_count)])
        else:
            print('use 1 gpu who named: {}'.format(torch.cuda.get_device_name(0)))
    else:
        print('no gpu available !')
    model.to(device)
    return model, device


class OutputHook(list):
    """ Hook to capture module outputs.
    """
    def __call__(self, module, input, output):
        self.append(output)


class Solver(object):
    def __init__(self, args):

        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.expe_name = args.expe_name
        self.beta = args.beta
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.is_C = args.is_C
        self.second_layer_C = args.second_layer_C
        # Lambda:
        self.lambda_recons = args.lambda_recons
        self.lambda_class = args.lambda_class
        self.lambda_Kl_var = args.lambda_Kl_var
        self.lambda_Kl_struct = args.lambda_Kl_struct
        self.lambda_partial_class = args.lambda_partial_class
        self.lambda_zvar_sim = args.lambda_zvar_sim
        self.lambda_VAE = args.lambda_VAE
        self.display_step = args.display_step
        self.save_step = args.save_step
        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.is_partial_rand_class = args.is_partial_rand_class
        self.is_E1 = args.is_E1
        self.is_binary_structural_latent = args.is_binary_structural_latent
        self.random_percentage = args.random_percentage
        self.is_zvar_sim_loss = args.is_zvar_sim_loss
        self.BN = args.BN
        self.just_train = args.just_train
        self.E1_dense = args.E1_dense
        self.E1_conv = args.E1_conv
        self.is_noise_stats = args.is_noise_stats
        self.is_perturbed_score = args.is_perturbed_score
        self.zvar_sim_loss_only_for_encoder = args.zvar_sim_loss_only_for_encoder
        self.zvar_sim_loss_for_all_model = args.zvar_sim_loss_for_all_model
        self.hidden_filters_layer1 = args.hidden_filters_layer1
        self.hidden_filters_layer2 = args.hidden_filters_layer2
        self.hidden_filters_layer3 = args.hidden_filters_layer3
        self.stride_size = args.stride_size
        self.kernel_size = args.kernel_size
        self.zvar_sim_var_rand = args.zvar_sim_var_rand
        self.zvar_sim_normal = args.zvar_sim_normal
        self.zvar_sim_change_zvar = args.zvar_sim_change_zvar
        self.old_weighted = args.old_weighted
        self.adapt_lr = args.adapt_lr
        self.E1_second_conv = args.E1_second_conv
        self.E1_second_conv_adapt = args.E1_second_conv_adapt
        self.L1_norm_weights = args.L1_norm_weights
        self.L1_norm_act = args.L1_norm_act
        self.E1_VAE = args.E1_VAE
        self.E1_AE = args.E1_AE
        self.two_encoder = args.two_encoder
        self.big_kernel = args.big_kernel
        self.big_kernel_size = args.big_kernel_size
        self.GMP = args.GMP
        self.zeros_W_Classif = args.zeros_W_Classif
        self.lambda_L1 = args.lambda_L1

        if self.zvar_sim_loss_only_for_encoder or self.zvar_sim_loss_for_all_model:
            list_uniq_choice = [self.zvar_sim_loss_only_for_encoder, self.zvar_sim_loss_for_all_model]
            assert any(
                iter(list_uniq_choice)), "We must have only one choice for zvar_sim_loss propagation: in all the " \
                                         "model, in only encoder or in encoder and decoder !"

        if self.zvar_sim_var_rand or self.zvar_sim_normal or self.zvar_sim_change_zvar:
            list_uniq_choice_zvar_strategie = [self.zvar_sim_var_rand, self.zvar_sim_normal, self.zvar_sim_change_zvar]
            assert any(
                iter(list_uniq_choice_zvar_strategie)), "We must have only one choice for zvar_sim_loss strategie: zvar " \
                                                        "rand normal or change zvar !"

        if self.old_weighted:
            self.normalize_weights = (self.beta * (self.lambda_Kl_var + self.lambda_Kl_struct)) + self.lambda_class + \
                                     self.lambda_recons
        else:
            self.normalize_weights = (self.lambda_VAE * (self.beta * (self.lambda_Kl_struct + self.lambda_Kl_var))) + \
                                     (self.lambda_VAE * self.lambda_recons) + self.lambda_class

        self.lambda_Kl_var_normalized = self.lambda_Kl_var / self.normalize_weights
        self.lambda_Kl_struct_normalized = self.lambda_Kl_struct / self.normalize_weights
        self.lambda_recons_normalized = self.lambda_recons / self.normalize_weights
        self.beta_normalized = self.beta / self.normalize_weights
        self.lambda_class_normalized = self.lambda_class / self.normalize_weights
        self.lambda_partial_class_normalized = self.lambda_partial_class / self.normalize_weights
        self.lambda_zvar_sim_normalized = self.lambda_zvar_sim
        self.lambda_VAE_normalized = self.lambda_VAE / self.normalize_weights

        self.four_conv = True
        if args.dataset.lower() == 'dsprites':
            self.img_size = (1, 32, 32)
            self.nb_class = 6
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        elif args.dataset.lower() == 'chairs':
            self.img_size = (3, 64, 64)
            self.nb_class = 1393
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'celeba':
            self.img_size = (3, 64, 64)
            self.nb_class = 10177
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'mnist':
            self.img_size = (1, 32, 32)
            self.nb_class = 10
            self.nc = 1
            self.decoder_dist = 'gaussian'
            self.four_conv = False
        elif args.dataset.lower() == 'fashion':
            self.img_size = (1, 32, 32)
            self.nb_class = 10
            self.nc = 1
            self.decoder_dist = 'gaussian'
            self.four_conv = False
        else:
            raise NotImplementedError

        self.nb_pixels = self.img_size[1] * self.img_size[2]

        # get latent space:
        self.latent_spec = {}
        if args.latent_spec_cont is not None:
            self.latent_spec["cont"] = args.latent_spec_cont
        if args.latent_spec_disc is not None:
            disc = args.latent_spec_disc
            self.latent_spec["disc"] = (np.ones(disc[0]) * disc[1]).astype(int)
        if args.latent_spec_cont_var is not None:
            self.latent_spec["cont_var"] = args.latent_spec_cont_var
        if args.latent_spec_cont_class is not None:
            self.latent_spec["cont_class"] = args.latent_spec_cont_class
        if args.latent_spec_disc_var is not None:
            disc_var = args.latent_spec_disc_var
            self.latent_spec["disc_var"] = (np.ones(disc_var[0]) * disc_var[1]).astype(int)
        if args.latent_spec_disc_class is not None:
            disc_class = args.latent_spec_disc_class
            self.latent_spec["disc_class"] = (np.ones(disc_class[0]) * disc_class[1]).astype(int)

        self.is_continuous = 'cont' in self.latent_spec
        self.is_discrete = 'disc' in self.latent_spec
        self.is_both_continue = 'cont_var' in self.latent_spec
        self.is_both_discrete = 'disc_var' in self.latent_spec

        # Keep track of divergence values for each latent variable
        self.scores_train = 0
        self.scores_test = 0
        self.loss_train = 0
        self.loss_test = 0
        self.mean_proba_per_class_train = 0
        self.std_proba_per_class_train = 0
        self.mean_proba_per_class_test = 0
        self.std_proba_per_class_test = 0
        self.mean_proba_per_class_noised_train = 0
        self.std_proba_per_class_noised_train = 0
        self.mean_proba_per_class_noised_test = 0
        self.std_proba_per_class_noised_test = 0
        self.epochs = 0
        self.global_iter = 0
        self.Lc = 0
        self.Lpc = 0
        self.Lm = 0
        self.L_KL_var = 0
        self.L_KL_struct = 0
        self.L_KL = 0
        self.Lr = 0
        self.L_Total = 0
        self.L_AE = 0
        self.L_Total_wt_weights = 0
        self.Lm_1 = 0
        self.Lm_2 = 0

        list_condition = [self.is_continuous, self.is_discrete, self.is_both_continue, self.is_both_discrete]
        assert any(iter(list_condition)), "only one condition may be True"

        # logger
        formatter = logging.Formatter('%(asc_time)s %(level_name)s - %(funcName)s: %(message)s', "%H:%M:%S")
        logger = logging.getLogger(__name__)
        logger.setLevel(args.log_level.upper())
        stream = logging.StreamHandler()
        stream.setLevel(args.log_level.upper())
        stream.setFormatter(formatter)
        logger.addHandler(stream)

        # dataset:
        # PREPARES DATA
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(args.dataset,
                                                                                 batch_size=self.batch_size,
                                                                                 logger=logger)

        self.train_loader_size = len(self.train_loader.dataset)
        self.valid_loader_size = len(self.valid_loader.dataset)
        self.test_loader_size = len(self.test_loader.dataset)

        logger.info("Train {} with {} train samples, {} valid samples and {}"
                    " test samples".format(args.dataset,
                                           self.train_loader_size,
                                           self.valid_loader_size,
                                           self.test_loader_size))

        # device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # create model
        net = BetaVAE(self.latent_spec, self.nb_class, self.is_C, device, nc=self.nc, four_conv=self.four_conv,
                      second_layer_C=self.second_layer_C, is_E1=self.is_E1,
                      is_binary_structural_latent=self.is_binary_structural_latent, BN=self.BN, E1_conv=self.E1_conv,
                      E1_dense=self.E1_dense, hidden_filters_1=self.hidden_filters_layer1,
                      hidden_filters_2=self.hidden_filters_layer2, hidden_filters_3=self.hidden_filters_layer3,
                      stride_size=self.stride_size, kernel_size=self.kernel_size, E1_second_conv=self.E1_second_conv,
                      E1_second_conv_adapt=self.E1_second_conv_adapt, E1_VAE=self.E1_VAE, E1_AE=self.E1_AE,
                      two_encoder=self.two_encoder, big_kernel_size=self.big_kernel_size[0], big_kernel=self.big_kernel,
                      GMP=self.GMP, zeros_W_Classif=self.zeros_W_Classif)

        # print model characteristics:
        print(net)
        num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('The number of parameters of model is', num_params)

        # config gpu:
        self.net, self.device = gpu_config(net)
        self.optimizer = optimizer.Adam(self.net.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

        if 'parallel' in str(type(self.net)):
            self.net = self.net.module
        else:
            self.net = self.net

        if not self.just_train:
            # experience name
            self.checkpoint_dir = os.path.join(args.ckpt_dir, args.expe_name)
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir, exist_ok=True)

            self.checkpoint_dir_scores = os.path.join(args.ckpt_dir_scores, args.expe_name)
            if not os.path.exists(self.checkpoint_dir_scores):
                os.makedirs(self.checkpoint_dir_scores, exist_ok=True)

            # checkpoint save
            self.file_path_checkpoint_scores = os.path.join(self.checkpoint_dir_scores, 'last')
            if not os.path.exists(self.file_path_checkpoint_scores):
                self.checkpoint_scores = {'iter': [],
                                          'epochs': [],
                                          'Zc_Zd_train': [],
                                          'score_zc_zd_partial_train': [],
                                          'score_zc_zd_partial_test': [],
                                          'Zc_random_Zd_train': [],
                                          'Zc_Zd_random_train': [],
                                          'Zc_pert_Zd_train': [],
                                          'Zc_Zd_pert_train': [],
                                          'Zc_Zd_test': [],
                                          'Zc_random_Zd_test': [],
                                          'Zc_Zd_random_test': [],
                                          'Zc_pert_Zd_test': [],
                                          'Zc_Zd_pert_test': [],
                                          'classification_loss_train': [],
                                          'classification_loss_test': [],
                                          'classification_partial_rand_loss_train': [],
                                          'classification_partial_rand_loss_test': [],
                                          'recon_loss_train': [],
                                          'recon_loss_test': [],
                                          'zvar_sim_loss_train': [],
                                          'zvar_sim_loss_test': [],
                                          'kl_var_loss_train': [],
                                          'kl_var_loss_test': [],
                                          'kl_class_loss_train': [],
                                          'kl_class_loss_test': [],
                                          'total_kld_train': [],
                                          'total_kld_test': [],
                                          'vae_loss_train': [],
                                          'vae_loss_test': [],
                                          'one_bit_rand_mean_pred_train': [],
                                          'one_bit_rand_std_pred_train': [],
                                          'one_bit_rand_mean_pred_test': [],
                                          'one_bit_rand_std_pred_test': [],
                                          'one_bit_rand_noised_mean_pred_train': [],
                                          'one_bit_rand_noised_std_pred_train': [],
                                          'one_bit_rand_noised_mean_pred_test': [],
                                          'one_bit_rand_noised_std_pred_test': []
                                          }
                with open(self.file_path_checkpoint_scores, mode='wb+') as f:
                    torch.save(self.checkpoint_scores, f)

            # load checkpoints:
            self.load_checkpoint_scores('last')
            self.load_checkpoint('last')

        self.loop = 0
        self.i = 0

        if not self.old_weighted and self.adapt_lr:
            self.lr = self.lr * self.nb_pixels  # we adapt lr to compare with old expe: indeed, in new expe we compute
            # loss by element, but before we commputed loss by image (1024 mroe big).

        if self.L1_norm_act:
            # Register hook to capture the layer outputs. Non-trivial networks will often
            # require hooks to be applied more judiciously.
            self.output_hook = OutputHook()
            if self.big_kernel:
                if self.two_encoder:
                    num_layer_L1 = 3
                elif self.E1_second_conv_adapt:
                    num_layer_L1 = 0
            else:
                if self.two_encoder:
                    num_layer_L1 = 6
                elif self.E1_second_conv_adapt:
                    num_layer_L1 = 3
            self.net.E1[num_layer_L1].register_forward_hook(self.output_hook)

    def train(self):
        self.net_mode(train=True)
        out = False

        print_bar = tqdm(total=self.max_iter)
        print_bar.update(self.epochs)
        while not out:
            self.loop += 1
            if self.loop % 2 == 0:  # if pair loop
                self.i = 0
            else:  # if impair loop
                self.i = 1
            for data, labels in self.train_loader:
                torch.autograd.set_detect_anomaly(True)

                self.i += 1
                self.global_iter += 1
                self.epochs = self.global_iter / len(self.train_loader)
                print_bar.update(1)

                data = data.to(self.device)  # Variable(data.to(self.device))
                labels = labels.to(self.device)  # Variable(labels.to(self.device))

                x_recon, _, _, latent_representation, latent_sample, _, _, latent_sample_random_continue, prediction, \
                pred_noised, prediction_partial_rand_class, prediction_random_variability, _, _, \
                prediction_zc_zd_pert, z_var, \
                z_var_reconstructed = self.net(data,
                                               both_continue=self.is_both_continue,
                                               both_discrete=self.is_both_discrete,
                                               is_partial_rand_class=self.is_partial_rand_class,
                                               random_percentage=self.random_percentage,
                                               is_E1=self.is_E1,
                                               is_zvar_sim_loss=self.is_zvar_sim_loss,
                                               var_rand=self.zvar_sim_var_rand,
                                               normal=self.zvar_sim_normal,
                                               change_zvar=self.zvar_sim_change_zvar)

                # Reconstruction and Classification losses
                # self.Lr = F.mse_loss(x_recon, data)
                if self.old_weighted:
                    self.Lr = F.mse_loss(x_recon, data, size_average=False).div(self.batch_size)
                else:
                    self.Lr = F.mse_loss(x_recon,
                                         data)  # pytorch default: size_average=True: averages over "each atomic
                    # element for which loss is computed for"
                if self.is_C:
                    if self.old_weighted:
                        self.Lc = F.nll_loss(prediction_random_variability,
                                             labels, size_average=False).div(self.batch_size)
                    else:
                        # classificatino loss
                        self.Lc = F.nll_loss(prediction_random_variability,
                                             labels)  # averaged over each loss element in the batch
                if self.is_partial_rand_class:
                    self.Lpc = F.nll_loss(prediction_partial_rand_class, labels)

                # zvar_sim_loss loss:
                if self.is_zvar_sim_loss:
                    if self.old_weighted:
                        self.Lm = F.mse_loss(z_var_reconstructed, z_var, size_average=False).div(self.batch_size)
                    else:
                        self.Lm = F.mse_loss(z_var_reconstructed, z_var)

                # print(z_var[0])
                # print(z_var_reconstructed[0])
                # print(self.Lm)

                if self.is_both_continue:
                    mu_var, logvar_var = latent_representation['cont_var']
                    kl_cont_loss_var = kl_divergence(mu_var, logvar_var)
                    # print(latent_representation['cont_class'][1].shape)
                    if self.E1_VAE:
                        mu_class, logvar_class = latent_representation['cont_class']
                        kl_cont_loss_class = kl_divergence(mu_class, logvar_class)
                        self.L_KL_struct = kl_cont_loss_class.item()
                    elif self.E1_AE:
                        self.L_KL_struct = 0
                    self.L_KL_var = kl_cont_loss_var
                elif self.is_both_discrete:
                    kl_disc_loss_var = _kl_multiple_discrete_loss(latent_representation['disc_var'])
                    kl_disc_loss_class = _kl_multiple_discrete_loss(latent_representation['disc_class'])
                    self.L_KL_var = kl_disc_loss_var
                    self.L_KL_struct = kl_disc_loss_class
                else:
                    if self.is_continuous:
                        # Calculate KL divergence
                        mu, logvar = latent_representation['cont']
                        kl_cont_loss = kl_divergence(mu, logvar)
                        self.L_KL_var = kl_cont_loss
                    if self.is_discrete:
                        # Calculate KL divergence
                        kl_disc_loss = _kl_multiple_discrete_loss(latent_representation['disc'])
                        self.L_KL_struct = kl_disc_loss

                # Calculate total kl value to record it
                if self.old_weighted:
                    self.L_KL = (self.lambda_Kl_var_normalized * self.L_KL_var.item()) + \
                                (self.lambda_Kl_struct_normalized * self.L_KL_struct)

                    self.L_Total = (self.lambda_recons_normalized * self.Lr) + \
                                   (self.beta_normalized * self.L_KL) + \
                                   (self.lambda_class_normalized * self.Lc) + \
                                   (self.lambda_partial_class_normalized * self.Lpc)

                    self.L_Total_wt_weights = self.Lr + self.L_KL_var.item() + self.L_KL_struct + self.Lc + self.Lpc
                else:
                    self.L_KL = self.beta_normalized * (self.L_KL_var.item() + self.L_KL_struct)
                    self.L_AE = self.L_KL + self.Lr

                    # Warning: if we add a new lambda: update normalize weight !!!!
                    self.L_Total = (self.lambda_VAE_normalized * self.L_AE) + (self.lambda_class_normalized * self.Lc)
                    self.L_Total_wt_weights = self.Lr + self.L_KL_var.item() + self.L_KL_struct + self.Lc + self.L_AE

                if self.is_zvar_sim_loss and self.zvar_sim_loss_for_all_model:
                    self.L_Total += (self.Lm * self.lambda_zvar_sim_normalized)
                    self.L_Total_wt_weights += self.Lm

                if self.L1_norm_weights:
                    l1_regularization = 0.
                    if self.big_kernel:
                        if self.two_encoder:
                            num_layer_L1 = 3
                        elif self.E1_second_conv_adapt:
                            num_layer_L1 = 0
                    else:
                        if self.two_encoder:
                            num_layer_L1 = 6
                        elif self.E1_second_conv_adapt:
                            num_layer_L1 = 3
                    for param in self.net.E1[num_layer_L1].parameters():
                        print('hereee')
                        print(param.shape)
                        l1_regularization += param.abs().sum()
                        print(l1_regularization)
                        l1_regularization = self.lambda_L1 * l1_regularization
                    print(self.L_Total)
                    self.L_Total += l1_regularization
                    print(self.L_Total)
                elif self.L1_norm_act:
                    l1_penalty = 0.
                    for output in self.output_hook:
                        print(output.shape)
                        l1_penalty += torch.norm(output, 1)
                        print(l1_penalty)
                        l1_penalty *= self.lambda_L1
                    self.L_Total += l1_regularization


                # plot parameters:
                # print('-----------::::::::::::Before:::::::-----------------:', self.i)
                # print(self.net.encoder[0].weight[0][0])
                # print(self.net.decoder[0].weight[0])
                # print(self.net.E1[0].weight[0][0])
                # print(self.net.L3_classifier[0].weight[0])

                if self.is_zvar_sim_loss and not self.zvar_sim_loss_for_all_model:
                    if self.i % 2 == 0:
                        self.optimizer.zero_grad()
                        self.L_Total.backward()  # retain_graph=True: if I use an another one backward
                        self.optimizer.step()
                    else:
                        # Backward gradient only for Encoder with the zvar_sim_loss loss:
                        # we want to freeze some modules
                        if self.zvar_sim_loss_only_for_encoder:
                            # we freeze also decoder
                            for params in self.net.decoder.parameters():
                                params.requires_grad = False
                        if self.is_E1:
                            for params in self.net.E1.parameters():
                                params.requires_grad = False
                        if self.is_C:
                            for params in self.net.L3_classifier.parameters():
                                params.requires_grad = False

                        # passing only those parameters that explicitly requires grad
                        self.optimizer = optimizer.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                                        lr=self.lr)
                        # backward zvar_sim loss:
                        self.optimizer.zero_grad()
                        # print('lmmmmmmmmmmmmmmmmmmmmmm', self.Lm)
                        # print('lmmwwwwwwwwwwwwwwwwwwwwwwwwwwww', self.Lm * self.lambda_zvar_sim_normalized)
                        (self.Lm * self.lambda_zvar_sim_normalized).backward()
                        self.optimizer.step()

                        # Then unfreeze layers:
                        if self.zvar_sim_loss_only_for_encoder:
                            for params in self.net.decoder.parameters():
                                params.requires_grad = True
                        if self.is_E1:
                            for params in self.net.E1.parameters():
                                params.requires_grad = True
                        if self.is_C:
                            for params in self.net.L3_classifier.parameters():
                                params.requires_grad = True

                        # add the unfrozen weight to the current optimizer
                        if self.zvar_sim_loss_only_for_encoder:
                            self.optimizer.add_param_group({'params': self.net.decoder.parameters()})
                        if self.is_E1:
                            self.optimizer.add_param_group({'params': self.net.E1.parameters()})
                        if self.is_C:
                            self.optimizer.add_param_group({'params': self.net.L3_classifier.parameters()})
                else:
                    self.optimizer.zero_grad()
                    self.L_Total.backward()  # retain_graph=True: if I use an another one backward
                    self.optimizer.step()

                # print('-----------::::::::::::After:::::::-----------------:')
                # print(self.net.encoder[0].weight[0][0])
                # print(self.net.decoder[0].weight[0])
                # print(self.net.E1[0].weight[0][0])
                # print(self.net.L3_classifier[0].weight[0])

                # display step
                if self.global_iter % self.display_step == 0:
                    print_bar.write('epoch: [{:.1f}], vae_loss:{:.2f}, Lr:{:.2f}, L_kl:{:.2f}, L_kl_var:{:.2f}, '
                                    'L_kl_struct:{:.2f}, L_c:{:.2f}, L_pc:{:.2f}, L_m:{:.2f}'.format(
                        self.epochs,
                        self.L_Total_wt_weights.item(),
                        self.Lr,
                        self.L_KL_var.item() + self.L_KL_struct,
                        self.L_KL_var.item(),
                        self.L_KL_struct,
                        self.Lc,
                        self.Lpc,
                        self.Lm))

                # save step
                if not self.just_train:
                    if self.global_iter % self.save_step == 0:
                        # save checkpoint
                        self.save_checkpoint('last')

                        self.net_mode(train=False)
                        self.scores_train, self.scores_test, self.loss_train, self.loss_test, \
                        self.mean_proba_per_class_train, self.std_proba_per_class_train, \
                        self.mean_proba_per_class_test, \
                        self.std_proba_per_class_test, self.mean_proba_per_class_noised_train, \
                        self.std_proba_per_class_noised_train, self.mean_proba_per_class_noised_test, \
                        self.std_proba_per_class_noised_test = compute_scores_and_loss(self.net,
                                                                                       self.train_loader,
                                                                                       self.valid_loader,
                                                                                       self.device,
                                                                                       self.latent_spec,
                                                                                       self.train_loader_size,
                                                                                       self.test_loader_size,
                                                                                       self.is_partial_rand_class,
                                                                                       self.random_percentage,
                                                                                       self.is_E1,
                                                                                       self.is_zvar_sim_loss,
                                                                                       self.is_C,
                                                                                       self.is_noise_stats,
                                                                                       self.is_perturbed_score,
                                                                                       self.zvar_sim_var_rand,
                                                                                       self.zvar_sim_normal,
                                                                                       self.zvar_sim_change_zvar,
                                                                                       self.old_weighted,
                                                                                       self.E1_VAE,
                                                                                       self.E1_AE)
                        self.save_checkpoint_scores_loss()
                        self.net_mode(train=True)

                        if self.is_C:
                            print_bar.write(
                                'iter: [{:.1f}], epoch: [{:.3f}], score_Zc_Zd_train:{:.5f}, score_Zc_Zd_test:{:.5f},'
                                'score_Zc_rand_Zd_train:{:.5f},score_Zc_rand_Zd_test:{:.5f},'
                                'score_Zc_Zd_rand_train:{:.5f}, score_Zc_Zd_rand_test:{:.5f}, '
                                'score_zc_zd_partial_train:{:.5f}, score_zc_zd_partial_test:{:.5f},'
                                'score_Zc_pert_Zd_train:{}, score_Zc_pert_Zd_test:{}, score_Zc_Zd_pert_train:{},'
                                'score_Zc_Zd_pert_test:{}'.format(self.global_iter, self.epochs,
                                                                  self.scores_train['Zc_Zd'],
                                                                  self.scores_test['Zc_Zd'],
                                                                  self.scores_train['Zc_random_Zd'],
                                                                  self.scores_test['Zc_random_Zd'],
                                                                  self.scores_train['Zc_Zd_random'],
                                                                  self.scores_test['Zc_Zd_random'],
                                                                  self.scores_train['score_zc_zd_partial'],
                                                                  self.scores_test['score_zc_zd_partial'],
                                                                  self.scores_train['Zc_pert_Zd'],
                                                                  self.scores_test['Zc_pert_Zd'],
                                                                  self.scores_train['Zc_Zd_pert'],
                                                                  self.scores_test['Zc_Zd_pert']))

                if self.epochs >= self.max_iter:
                    out = True
                    break

        print_bar.write("[Training Finished]")
        print_bar.close()

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise Exception('Only bool type is supported. True or False')
        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint_scores_loss(self, silent=False):
        """
        save all sample_scores and loss values
        """
        self.checkpoint_scores['iter'].append(self.global_iter)
        self.checkpoint_scores['epochs'].append(self.epochs)
        if self.is_C:
            # sample_scores
            self.checkpoint_scores['Zc_Zd_train'].append(self.scores_train['Zc_Zd'])
            self.checkpoint_scores['Zc_random_Zd_train'].append(self.scores_train['Zc_random_Zd'])
            self.checkpoint_scores['score_zc_zd_partial_train'].append(self.scores_train['score_zc_zd_partial'])
            self.checkpoint_scores['score_zc_zd_partial_test'].append(self.scores_test['score_zc_zd_partial'])
            self.checkpoint_scores['Zc_Zd_random_train'].append(self.scores_train['Zc_Zd_random'])
            self.checkpoint_scores['Zc_pert_Zd_train'].append(self.scores_train['Zc_pert_Zd'])
            self.checkpoint_scores['Zc_Zd_pert_train'].append(self.scores_train['Zc_Zd_pert'])
            self.checkpoint_scores['Zc_Zd_test'].append(self.scores_test['Zc_Zd'])
            self.checkpoint_scores['Zc_random_Zd_test'].append(self.scores_test['Zc_random_Zd'])
            self.checkpoint_scores['Zc_Zd_random_test'].append(self.scores_test['Zc_Zd_random'])
            self.checkpoint_scores['Zc_pert_Zd_test'].append(self.scores_test['Zc_pert_Zd'])
            self.checkpoint_scores['Zc_Zd_pert_test'].append(self.scores_test['Zc_Zd_pert'])
            # sample_scores with one bit random
            self.checkpoint_scores['one_bit_rand_mean_pred_train'].append(self.mean_proba_per_class_train)
            self.checkpoint_scores['one_bit_rand_std_pred_train'].append(self.std_proba_per_class_train)
            self.checkpoint_scores['one_bit_rand_mean_pred_test'].append(self.mean_proba_per_class_test)
            self.checkpoint_scores['one_bit_rand_std_pred_test'].append(self.std_proba_per_class_test)
            self.checkpoint_scores['one_bit_rand_noised_mean_pred_train'].append(self.mean_proba_per_class_noised_train)
            self.checkpoint_scores['one_bit_rand_noised_std_pred_train'].append(self.std_proba_per_class_noised_train)
            self.checkpoint_scores['one_bit_rand_noised_mean_pred_test'].append(self.mean_proba_per_class_noised_test)
            self.checkpoint_scores['one_bit_rand_noised_std_pred_test'].append(self.std_proba_per_class_noised_test)
        # losses
        self.checkpoint_scores['classification_loss_train'].append(self.loss_train['classification_loss'])
        self.checkpoint_scores['classification_loss_test'].append(self.loss_test['classification_loss'])
        self.checkpoint_scores['classification_partial_rand_loss_train'].append(
            self.loss_train['classification_partial_rand_loss'])
        self.checkpoint_scores['classification_partial_rand_loss_test'].append(
            self.loss_test['classification_partial_rand_loss'])
        self.checkpoint_scores['recon_loss_train'].append(self.loss_train['recon_loss'])
        self.checkpoint_scores['recon_loss_test'].append(self.loss_test['recon_loss'])
        self.checkpoint_scores['kl_var_loss_train'].append(self.loss_train['kl_var_loss'])
        self.checkpoint_scores['kl_var_loss_test'].append(self.loss_test['kl_var_loss'])
        self.checkpoint_scores['kl_class_loss_train'].append(self.loss_train['kl_class_loss'])
        self.checkpoint_scores['kl_class_loss_test'].append(self.loss_test['kl_class_loss'])
        self.checkpoint_scores['total_kld_train'].append(self.loss_train['total_kld'])
        self.checkpoint_scores['total_kld_test'].append(self.loss_test['total_kld'])
        self.checkpoint_scores['zvar_sim_loss_train'].append(self.loss_train['zvar_sim_loss'])
        self.checkpoint_scores['zvar_sim_loss_test'].append(self.loss_test['zvar_sim_loss'])
        self.checkpoint_scores['vae_loss_train'].append(self.loss_train['vae_loss'])
        self.checkpoint_scores['vae_loss_test'].append(self.loss_test['vae_loss'])

        with open(self.file_path_checkpoint_scores, mode='wb+') as f:
            torch.save(self.checkpoint_scores, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {}, epoch {})".format(self.file_path_checkpoint_scores,
                                                                        self.global_iter, self.epochs))

    def save_checkpoint(self, filename, silent=False):
        model = {'net': self.net.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'iter': self.global_iter,
                 'epochs': self.epochs
                 }
        file_path = os.path.join(self.checkpoint_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(model, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {}, epoch {})".format(file_path, self.global_iter, self.epochs))

    def load_checkpoint_scores(self, filename):
        file_path = os.path.join(self.checkpoint_dir_scores, filename)
        if os.path.isfile(file_path):
            if os.path.getsize(file_path) > 0:  # check that the file is not empty
                with open(file_path, "rb"):
                    if not torch.cuda.is_available():
                        self.checkpoint_scores = torch.load(file_path, map_location=torch.device('cpu'))
                    else:
                        self.checkpoint_scores = torch.load(file_path)
                    print("=> loaded sample_scores and losses")
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.checkpoint_dir, filename)
        if os.path.isfile(file_path):
            if not torch.cuda.is_available():
                checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.epochs = checkpoint['epochs']
            self.net.load_state_dict(checkpoint['net'])
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
