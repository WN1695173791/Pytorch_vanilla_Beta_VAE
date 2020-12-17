from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
import random
from utils.activations import DeterministicBinaryActivation

EPS = 1e-12


def reparameterization_trick(mu, logvar):
    """
    Samples from a normal distribution using the reparameterization trick.
    :param mu: torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)
    :param logvar: torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
    :return:
    """
    """
    if mu.training:
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps
    else:
        # Reconstruction mode
        return mu
    """
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, latent_spec, nb_class, is_C, device, temperature=0.67, nc=3, four_conv=True, 
                 second_layer_C=False, is_E1=False,  is_binary_structural_latent=False, 
                 BN=False, E1_conv=False, E1_dense=False, batch_size=64, hidden_filters_1=32,
                 hidden_filters_2=32, hidden_filters_3=32, stride_size=2, kernel_size=4, E1_second_conv=False):
        """
        Class which defines model and forward pass.
        Parameters
        ----------
        latent_spec : dict
            Specifies latent distribution. For example:
            {'cont': 10, 'disc': [10, 4, 3]} encodes 10 normal variables and
            3 gumbel softmax variables of dimension 10, 4 and 3. A latent spec
            can include both 'cont' and 'disc' or only 'cont' or only 'disc'.
        temperature : float
            Temperature for gumbel softmax distribution.
            param: binary_structural_latent: bool: use with binary/continue latent space
        """
        super(BetaVAE, self).__init__()

        # Parameters
        self.E1_second_conv = E1_second_conv
        self.BN = BN
        self.is_binary_structural_latent = is_binary_structural_latent
        self.nc = nc
        self.is_continuous = 'cont' in latent_spec
        self.is_discrete = 'disc' in latent_spec
        self.is_both_continue = 'cont_var' in latent_spec
        self.is_both_discrete = 'disc_var' in latent_spec
        self.is_E1 = is_E1
        self.E1_conv = E1_conv
        self.E1_dense = E1_dense
        self.batch_size = batch_size

        self.latent_spec = latent_spec
        self.temperature = temperature
        self.device = device
        self.second_layer_C = second_layer_C

        # Calculate dimensions of latent distribution
        self.latent_struct_dim = 0
        self.latent_var_dim = 0
        if self.is_continuous:
            self.latent_var_dim = self.latent_spec['cont']
        if self.is_discrete:
            self.latent_struct_dim += sum([dim for dim in self.latent_spec['disc']])
            self.num_disc_latents = len(self.latent_spec['disc'])
        if self.is_both_continue:
            self.latent_var_dim = self.latent_spec['cont_var']
            self.latent_struct_dim = self.latent_spec['cont_class']
        if self.is_both_discrete:
            self.latent_var_dim += sum([dim for dim in self.latent_spec['disc_var']])
            self.num_disc_latents_var = len(self.latent_spec['disc_var'])
            self.latent_struct_dim += sum([dim for dim in self.latent_spec['disc_class']])
            self.num_disc_latents_class = len(self.latent_spec['disc_class'])
        self.latent_dim = self.latent_var_dim + self.latent_struct_dim

        if self.is_both_continue:
            if self.is_E1:
                self.latent_dim_encoder = self.latent_var_dim * 2
            else:
                self.latent_dim_encoder = (self.latent_var_dim * 2) + (self.latent_struct_dim * 2)
            self.latent_dim = self.latent_var_dim + self.latent_struct_dim
        elif self.is_both_discrete:
            self.latent_dim_encoder = self.latent_var_dim + self.latent_struct_dim
            self.latent_dim = self.latent_dim_encoder
        else:
            self.latent_dim_encoder = (self.latent_var_dim * 2) + self.latent_struct_dim
            self.latent_dim = self.latent_var_dim + self.latent_struct_dim

        self.four_conv = four_conv
        self.structural_classifier_input_dim = 8192
        self.nb_class = nb_class
        self.is_C = is_C

        self.hidden_filters_1 = hidden_filters_1
        self.hidden_filters_2 = hidden_filters_2
        self.hidden_filters_3 = hidden_filters_3
        self.kernel_size = kernel_size
        self.stride_size = stride_size
        self.hidden_dim = 256
        self.padding = 0
        # Shape required to start transpose convs
        self.reshape = (self.hidden_filters_3, self.kernel_size, self.kernel_size)

        self.width_conv1_size = round(((32-self.kernel_size+(2*self.padding))/self.stride_size)+1)
        self.height_conv1_size = round(((32 - self.kernel_size + (2 * self.padding)) / self.stride_size) + 1)

        self.width_conv2_size = round(
            ((self.width_conv1_size - self.kernel_size + (2 * self.padding)) / self.stride_size) + 1)
        self.height_conv2_size = round(
            ((self.height_conv1_size - self.kernel_size + (2 * self.padding)) / self.stride_size) + 1)

        self.width_conv3_size = round(
            ((self.width_conv2_size - self.kernel_size + (2 * self.padding)) / self.stride_size) + 1)
        self.height_conv3_size = round(
            ((self.width_conv2_size - self.kernel_size + (2 * self.padding)) / self.stride_size) + 1)

        # ---------------------------------------- Define encoder E --------------------------------------------------
        # firsts conv layers
        self.encoder_conv_layer_1 = [
            nn.Conv2d(self.nc, self.hidden_filters_1, self.kernel_size, stride=self.stride_size, padding=1),
        ]
        if self.BN:
            self.encoder_conv_layer_1 += [
                nn.BatchNorm2d(self.hidden_filters_1),
                ]
        self.encoder_conv_layer_1 += [
            nn.ReLU(True)
        ]

        self.encoder_conv_layer_3 = [
            nn.Conv2d(self.hidden_filters_1, self.hidden_filters_2, self.kernel_size, stride=self.stride_size, padding=1),
        ]
        if self.BN:
            self.encoder_conv_layer_3 += [
                nn.BatchNorm2d(self.hidden_filters_2),
            ]
        self.encoder_conv_layer_3 += [
            nn.ReLU(True)
        ]

        self.encoder_conv_layer_4 = [
            nn.Conv2d(self.hidden_filters_2, self.hidden_filters_3, self.kernel_size, stride=self.stride_size, padding=1),
        ]
        if self.BN:
            self.encoder_conv_layer_4 += [
                nn.BatchNorm2d(self.hidden_filters_3),
            ]
        self.encoder_conv_layer_4 += [
            nn.ReLU(True)
        ]

        # Fully connected layers
        self.encoder_layer_5 = [
            View((-1, 512)),  # B, 512
            nn.Linear(np.product(self.reshape), self.hidden_dim),  # B, 256
            nn.ReLU(True)
        ]
        self.encoder_layer_6 = [
            nn.Linear(self.hidden_dim, self.hidden_dim),  # B, 256
        ]

        # if we want to use binary value for structural latent representation
        if self.is_binary_structural_latent:
            if self.four_conv:
                self.encoder_conv_layer_2 = [
                    nn.Conv2d(self.hidden_filters_1, self.hidden_filters_1, self.kernel_size),  # B,  32, 16, 16
                ]
                if self.BN:
                    self.encoder_conv_layer_2 += [
                        nn.BatchNorm2d(self.hidden_filters_1),
                    ]
                self.encoder_conv_layer_2 += [
                    nn.ReLU(True)
                ]
                self.encoder = nn.Sequential(*self.encoder_conv_layer_1,
                                             *self.encoder_conv_layer_2,
                                             *self.encoder_conv_layer_3,
                                             *self.encoder_conv_layer_4,
                                             *self.encoder_layer_5,
                                             *self.encoder_layer_6)
            else:
                self.encoder = nn.Sequential(*self.encoder_conv_layer_1,
                                             *self.encoder_conv_layer_3,
                                             *self.encoder_conv_layer_4,
                                             *self.encoder_layer_5,
                                             *self.encoder_layer_6)
            self.output_encoder_variability = [
                nn.ReLU(True),
                View((-1, self.hidden_dim)),  # B, self.hidden_dim
                nn.Linear(self.hidden_dim, self.latent_var_dim * 2)  # B, latent_dim
            ]
            self.output_encoder_structural = [
                # we binary the conv_layer5 output: dim: self.hidden_dim
                DeterministicBinaryActivation(estimator='ST'),
                View((-1, self.hidden_dim)),  # B, self.hidden_dim
                # after linear layer, the neurons are not binary
                nn.Linear(self.hidden_dim, self.latent_struct_dim)  # B, latent_dim
            ]
            self.encoder_output_encoder_variability = nn.Sequential(*self.output_encoder_variability)
            self.encoder_output_encoder_structural = nn.Sequential(*self.output_encoder_structural)
        else:
            # Fully connected layers for mean and variance
            self.mu_logvar_gen = [
                nn.ReLU(True),
                nn.Linear(self.hidden_dim, self.latent_dim_encoder),  # B, latent_dim * 2
            ]
            if self.four_conv:
                self.encoder_conv_layer_2 = [
                    nn.Conv2d(self.hidden_filters_1, self.hidden_filters_1, self.kernel_size, stride=self.stride_size, padding=1),
                    PrintLayer()
                ]
                if self.BN:
                    self.encoder_conv_layer_2 += [
                        nn.BatchNorm2d(self.hidden_filters_1),
                    ]
                self.encoder_conv_layer_2 += [
                    nn.ReLU(True)
                ]
                self.encoder = nn.Sequential(*self.encoder_conv_layer_1,
                                             *self.encoder_conv_layer_2,
                                             *self.encoder_conv_layer_3,
                                             *self.encoder_conv_layer_4,
                                             *self.encoder_layer_5,
                                             *self.encoder_layer_6,
                                             *self.mu_logvar_gen)
            else:
                self.encoder = nn.Sequential(*self.encoder_conv_layer_1,
                                             *self.encoder_conv_layer_3,
                                             *self.encoder_conv_layer_4,
                                             *self.encoder_layer_5,
                                             *self.encoder_layer_6,
                                             *self.mu_logvar_gen)
        # ---------------------------------------- end encoder E --------------------------------------------------

        # ---------------------------------------- Define Decoder D -------------------------------------------------
        self.decoder_list_layer = [
            # Fully connected layers with ReLu activations
            nn.Linear(self.latent_dim, self.hidden_dim),  # B, 256
            # PrintLayer(),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim),  # B, 256
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, np.product(self.reshape)),  # B, 512
            nn.ReLU(True),
            View((-1, *self.reshape)),
            # Convolutional layers with ReLu activations
            nn.ConvTranspose2d(self.hidden_filters_3, self.hidden_filters_2, self.kernel_size, stride=self.stride_size,
                               padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.hidden_filters_2, self.hidden_filters_1, self.kernel_size, stride=self.stride_size,
                               padding=1),
            nn.ReLU(True),
        ]
        if self.four_conv:
            self.decoder_list_layer += [
                nn.ConvTranspose2d(self.hidden_filters_1, self.hidden_filters_1, self.kernel_size,
                                   stride=self.stride_size, padding=1),
                nn.ReLU(True)
            ]
        else:
            # (32, 32) images are supported but do not require an extra layer
            pass
        self.decoder_list_layer += [
            nn.ConvTranspose2d(self.hidden_filters_1, self.nc, self.kernel_size, stride=self.stride_size,  padding=1),  # B, 32, 32, 32
            nn.Sigmoid()
        ]

        self.decoder = nn.Sequential(*self.decoder_list_layer)
        # ---------------------------------------- end Decoder --------------------------------------------------

        # ---------------------------------------- Define Classifier C -------------------------------------------------
        if self.is_C:
            if self.second_layer_C:
                self.L3_classifier = nn.Sequential(
                    nn.Linear(self.latent_dim, self.hidden_dim),
                    nn.ReLU(True),
                    nn.Linear(self.hidden_dim, self.nb_class),
                )
            else:
                self.L3_classifier = nn.Sequential(
                    nn.Linear(self.latent_dim, self.nb_class),
                )
        # ---------------------------------------- end Classifier C --------------------------------------------------

        # ----------------------------------------Define Encoder E1 --------------------------------------------------
        if self.is_E1:
            if self.E1_conv:
                if self.E1_second_conv:
                    self.E1 = nn.Sequential(
                        nn.Conv2d(32, 3, 3, 1, 1),
                        # PrintLayer(),
                        nn.BatchNorm2d(3),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),  # B, 3, 8, 8
                        # PrintLayer(),
                        View((-1, 48)),  # B, 192
                        nn.Linear(48, self.latent_struct_dim * 2)  # B, struct_dim*2
                    )
                else:
                    self.E1 = nn.Sequential(
                        nn.Conv2d(32, 3, 3, 1, 1),
                        nn.BatchNorm2d(3),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),  # B, 3, 8, 8
                        View((-1, 192)),  # B, 192
                        nn.Linear(192, self.latent_struct_dim * 2)  # B, struct_dim*2
                    )
            elif self.E1_dense:
                self.E1 = nn.Sequential(
                    # TODO: dense layer like schema
                )
            else:
                self.E1 = nn.Sequential(
                    View((-1, self.structural_classifier_input_dim)),
                    nn.Linear(self.structural_classifier_input_dim, 4096),
                    nn.ReLU(True),
                    nn.Linear(4096, 1024),
                    nn.ReLU(True),
                    nn.Linear(1024, self.latent_struct_dim * 2)
                )
        # ---------------------------------------- end Classifier E1 --------------------------------------------------
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x, reconstruction_rand=False, is_perturbed_score=False, is_noise_stats=False,
                is_prediction=False, both_continue=False,
                both_discrete=False, is_partial_rand_class=False, random_percentage=0.2, is_E1=False,
                is_zvar_sim_loss=False, var_rand=False, normal=False, change_zvar=False):
        """
        Forward pass of model.
        Parameters
        ----------
        :param is_zvar_sim_loss: 
        :param reconstruction_rand: mode reconstruction: do not propagate gradient
        :param is_E1: Boolean: if we use output information of down conv layer
        :param random_percentage: float between 0 et 1: represent percentage of neuron to randomize in the latent
         structural representation
        :param is_partial_rand_class: Boolean: if we use random neuron to randomize in the vector latent representation
        :param both_discrete: boolean: if the latent representation is whole discrete
        :param both_continue: boolean: if the latent representation is whole continue
        :param x: torch.Tensor of data. Shape (N, C, H, W)
        :param is_prediction: boolean: if the model is in eval mode
        """
        
        if self.is_binary_structural_latent:
            latent_representation = self._encode(x, both_continue=both_continue,
                                                 is_E1=is_E1,
                                                 binary_structural_latent=self.is_binary_structural_latent)
            latent_sample, latent_sample_cont, latent_sample_disc = self.reparameterization(latent_representation)
            latent_sample_variability = latent_sample_cont
            latent_sample_class = latent_sample_disc

            latent_sample_random_continue = self.representation_random_continue(latent_sample_class,
                                                                                self.latent_var_dim)
            latent_sample_random_discrete = self.representation_random_discrete(latent_sample_variability)

        elif self.is_E1:
            # in this mode: we must have to be in both_continue mode:
            latent_sample = []
            latent_representation = {}

            # for z_var we use whole E: return [mu, logvar]
            latent_representation_var = self._encode(x, both_continue=both_continue,
                                                     is_E1=is_E1)
            # for z_struct we use only the first conv layer: [mu, logvar]
            latent_representation_structural_layer_1 = self._encode_layer1(x, both_continue=both_continue,
                                                                           E1_second_conv=self.E1_second_conv)

            latent_representation['cont_var'] = latent_representation_var
            latent_representation['cont_class'] = latent_representation_structural_layer_1

            # reparametrize var:
            latent_sample_variability = self.reparameterization(latent_representation_var, both_continue=both_continue,
                                                           is_E1=is_E1)
            # reparametrize var:
            latent_sample_class = self.reparameterization(latent_representation_structural_layer_1,
                                                          both_continue=both_continue,
                                                          is_E1=is_E1)

            latent_sample.append(latent_sample_variability)

            latent_sample.append(latent_sample_class)
            latent_sample = torch.cat(latent_sample, dim=1)

            latent_sample_random_continue = self.representation_random_continue(latent_sample_class,
                                                                                self.latent_var_dim)
            latent_sample_random_discrete = self.representation_random_continue(latent_sample_variability,
                                                                                self.latent_struct_dim,
                                                                                is_random_class=True)
        elif both_continue:
            latent_representation = self._encode(x, both_continue=both_continue)

            latent_sample, latent_sample_cont_var, latent_sample_cont_class = \
                self.reparameterization(latent_representation, both_continue=both_continue)
            latent_sample_variability = latent_sample_cont_var
            latent_sample_class = latent_sample_cont_class

            latent_sample_random_continue = self.representation_random_continue(latent_sample_class,
                                                                                self.latent_var_dim)
            latent_sample_random_discrete = self.representation_random_continue(latent_sample_variability,
                                                                                self.latent_struct_dim,
                                                                                is_random_class=True)
        elif both_discrete:
            latent_representation = self._encode(x, both_discrete=both_discrete)
            latent_sample, latent_sample_disc_var, latent_sample_disc_class = \
                self.reparameterization(latent_representation, both_discrete=both_discrete)
            latent_sample_variability = latent_sample_disc_var
            latent_sample_class = latent_sample_disc_class

            latent_sample_random_continue = self.representation_random_discrete(latent_sample_class,
                                                                                both_discrete=both_discrete,
                                                                                is_random_var=True)
            latent_sample_random_discrete = self.representation_random_discrete(latent_sample_variability,
                                                                                both_discrete=both_discrete)
        else:
            # continue  and discrete
            latent_representation = self._encode(x)
            latent_sample, latent_sample_cont, latent_sample_disc = self.reparameterization(latent_representation)
            latent_sample_variability = latent_sample_cont
            latent_sample_class = latent_sample_disc

            latent_sample_random_continue = self.representation_random_continue(latent_sample_class,
                                                                                self.latent_var_dim)
            latent_sample_random_discrete = self.representation_random_discrete(latent_sample_variability,
                                                                                both_discrete=both_discrete)

        # prediction for partial rand classifier:
        prediction_partial_rand_class = 0
        if is_partial_rand_class:
            prediction_partial_rand_class = self.classification_partial_random_class(
                                                     latent_sample_class,
                                                     self.latent_var_dim,
                                                     self.latent_struct_dim,
                                                     random_percentage=random_percentage,
                                                     both_continue=both_continue)  # loss Cl_partial            

        # Reconstruction:
        x_recon = self._decode(latent_sample)

        # Reconstruction with only one part of latent representation:
        recons_random_variability = 0
        recons_random_class = 0
        if reconstruction_rand:
            recons_random_variability = self._decode(latent_sample_random_continue)
            recons_random_class = self._decode(latent_sample_random_discrete)
         
        # zvar_sim_loss:
        z_var = 0
        z_var_reconstructed = 0
        z_var_reconstructed_z1 = 0
        z_var_reconstructed_z2 = 0
        if is_zvar_sim_loss:
            if var_rand:
                recons_random_variability = self._decode(latent_sample_random_continue)
                latent_representation_recons_random_class = self._encode(recons_random_variability,
                                                                         both_continue=both_continue, is_E1=is_E1)
                if is_E1:
                    z_var_reconstructed = latent_representation_recons_random_class
                else:
                    z_var_reconstructed = latent_representation_recons_random_class['cont_var']
                z_var_reconstructed = torch.cat(z_var_reconstructed, dim=1)
                z_var = latent_representation['cont_var']
                z_var = torch.cat(z_var, dim=1)
            elif normal:
                recons = x_recon
                latent_representation_recons = self._encode(recons, both_continue=both_continue, is_E1=is_E1)
                if is_E1:
                    z_var_reconstructed = latent_representation_recons
                else:
                    z_var_reconstructed = latent_representation_recons['cont_var']
                z_var_reconstructed = torch.cat(z_var_reconstructed, dim=1)
                z_var = latent_representation['cont_var']
                z_var = torch.cat(z_var, dim=1)
            elif change_zvar:
                z1 = []
                z2 = []
                latent_sample_change = []
                middle = int(len(latent_sample_variability)/2)
                latent_sample_variability_1 = latent_sample_variability[:middle]
                latent_sample_class_1 = latent_sample_class[:middle]
                latent_sample_variability_2 = latent_sample_variability[middle:]
                latent_sample_class_2 = latent_sample_class[middle:]
                z1.append(latent_sample_variability_2)
                z1.append(latent_sample_class_1)
                z1 = torch.cat(z1, dim=1)
                z2.append(latent_sample_variability_1)
                z2.append(latent_sample_class_2)
                z2 = torch.cat(z2, dim=1)
                latent_sample_change.append(z1)
                latent_sample_change.append(z2)
                latent_sample_change = torch.cat(latent_sample_change, dim=0)
                recons = self._decode(latent_sample_change)
                latent_representation_recons = self._encode(recons, both_continue=both_continue, is_E1=is_E1)
                if is_E1:
                    z_var_reconstructed = latent_representation_recons
                else:
                    z_var_reconstructed = latent_representation_recons['cont_var']
                z_var_reconstructed = torch.cat(z_var_reconstructed, dim=1)
                z_var = latent_representation['cont_var']
                z_var = torch.cat(z_var, dim=1)

        # Classification:
        prediction = 0
        prediction_random_variability = 0
        prediction_random_discrete = 0
        prediction_zc_pert_zd = 0
        prediction_zc_zd_pert = 0
        pred_noised = 0

        if self.is_C:
            if both_continue:
                prediction_random_variability, latent_sample_random_continue = self.classification_random_continue(
                                                                                    latent_sample_class,
                                                                                    self.latent_var_dim)  # loss Cl
            elif both_discrete:
                prediction_random_variability = self.classification_random_discrete(latent_sample_class, 
                                                                                    both_discrete=both_discrete,
                                                                                    is_random_var=True)
            else:
                prediction_random_variability, latent_sample_random_continue = self.classification_random_continue(
                                                                                    latent_sample_class,
                                                                                    self.latent_var_dim)

        # Evaluation mode:
        if is_prediction:
            if self.is_C:
                prediction = self.classification(latent_sample)
                if both_continue:
                    prediction_random_discrete, latent_sample_random_continue_struct = self.classification_random_continue(
                                                                                                latent_sample_variability,
                                                                                                self.latent_struct_dim,
                                                                                                is_random_class=True)
                    if is_noise_stats:
                        pred_noised = self.classification_struct_noised(latent_sample_random_continue)
                    if is_perturbed_score:
                        prediction_zc_pert_zd = self.classification_zc_pert_zd(latent_sample_variability,
                                                                               latent_sample_class,
                                                                               both_continue=both_continue)
                        prediction_zc_zd_pert = self.classification_zc_pert_zd(latent_sample_variability,
                                                                               latent_sample_class,
                                                                               both_continue=both_continue,
                                                                               reverse=True)
                elif both_discrete:
                    prediction_random_discrete = self.classification_random_discrete(latent_sample_variability,
                                                                                     both_discrete=both_discrete)
                else:
                    prediction_random_discrete = self.classification_random_discrete(latent_sample_variability)
                    if is_perturbed_score:
                        prediction_zc_pert_zd = self.classification_zc_pert_zd(latent_sample_variability, latent_sample_class)
                        prediction_zc_zd_pert = self.classification_zc_zd_pert(latent_sample_variability, latent_sample_class)

        return x_recon, recons_random_variability, recons_random_class, latent_representation, latent_sample, \
               latent_sample_variability, latent_sample_class, latent_sample_random_continue, prediction, pred_noised, \
               prediction_partial_rand_class, prediction_random_variability, prediction_random_discrete, \
               prediction_zc_pert_zd, prediction_zc_zd_pert, z_var, z_var_reconstructed

    def _encode(self, x, both_continue=False, both_discrete=False, is_E1=False,
                binary_structural_latent=False):
        """
        Encodes an image into parameters of a latent distribution defined in
        self.latent_spec.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data, shape (N, C, H, W)
        """
        latent_dist = {}
        distributions = self.encoder(x)
        if binary_structural_latent:
            output_encoder_variability = self.encoder_output_encoder_variability(distributions)
            output_encoder_structural = self.encoder_output_encoder_structural(distributions)
            mu = output_encoder_variability[:, :self.latent_var_dim]
            logvar = output_encoder_variability[:, self.latent_var_dim:]
            latent_dist['cont'] = [mu, logvar]
            # Linear layer for each of the categorical distributions
            latent_dist['disc'] = []
            for i, disc_dim in enumerate(self.latent_spec['disc']):
                start = sum(self.latent_spec['disc'][:i])
                end = sum(self.latent_spec['disc'][:i + 1])
                latent_dist['disc'].append(F.softmax(output_encoder_structural[:, start:end], dim=1))
        elif is_E1:
            if both_continue:
                mu = distributions[:, :self.latent_var_dim]
                logvar = distributions[:, self.latent_var_dim:]

                latent_dist = [mu, logvar]
        else:
            if both_continue:
                mu_var = distributions[:, :self.latent_var_dim]
                logvar_var = distributions[:, self.latent_var_dim:self.latent_var_dim * 2]

                begin_mu_class = self.latent_var_dim * 2
                end_mu_class = (self.latent_var_dim * 2) + self.latent_struct_dim

                mu_class = distributions[:, begin_mu_class:end_mu_class]
                logvar_class = distributions[:, end_mu_class:]

                latent_dist['cont_var'] = [mu_var, logvar_var]
                latent_dist['cont_class'] = [mu_class, logvar_class]
            elif both_discrete:
                latent_dist['disc_var'] = []
                latent_dist['disc_class'] = []

                for i, disc_dim in enumerate(self.latent_spec['disc_var']):
                    start = sum(self.latent_spec['disc_var'][:i])
                    end = sum(self.latent_spec['disc_var'][:i + 1])
                    latent_dist['disc_var'].append(F.softmax(distributions[:, start:end], dim=1))

                for i, disc_dim in enumerate(self.latent_spec['disc_class']):
                    start = self.latent_var_dim + sum(self.latent_spec['disc_class'][:i])
                    end = self.latent_var_dim + sum(self.latent_spec['disc_class'][:i + 1])
                    latent_dist['disc_class'].append(F.softmax(distributions[:, start:end], dim=1))
            else:
                if self.is_continuous:
                    mu = distributions[:, :self.latent_var_dim]
                    logvar = distributions[:, self.latent_var_dim:self.latent_var_dim * 2]
                    latent_dist['cont'] = [mu, logvar]
                if self.is_discrete:
                    # Linear layer for each of the categorical distributions
                    latent_dist['disc'] = []
                    for i, disc_dim in enumerate(self.latent_spec['disc']):
                        start = self.latent_var_dim * 2 + sum(self.latent_spec['disc'][:i])
                        end = self.latent_var_dim * 2 + sum(self.latent_spec['disc'][:i + 1])
                        latent_dist['disc'].append(F.softmax(distributions[:, start:end], dim=1))

        return latent_dist

    def _encode_layer1(self, x, both_continue=False, E1_second_conv=False):

        if both_continue:
            if E1_second_conv:
                output_layer = self.encoder[:5](x)
            else:
                output_layer = self.encoder[:3](x)

            output_structure_classifier = self.E1(output_layer)

            mu = output_structure_classifier[:, :self.latent_struct_dim]
            logvar = output_structure_classifier[:, self.latent_struct_dim:]

            latent_dist_structural = [mu, logvar]

        return latent_dist_structural

    def reparameterization(self, latent_dist, both_continue=False, both_discrete=False, is_E1=False):
        """
        Samples from latent distribution using the reparametrization trick.
        Parameters
        ----------
        latent_dist : dict
            Dict with keys 'cont' or 'disc' or both, containing the parameters
            of the latent distributions as torch.Tensor instances.
            :param is_E1:
            :param latent_dist:
            :param both_continue:
            :param both_discrete:
        """
        if is_E1:
            mu, logvar = latent_dist
            sample = reparameterization_trick(mu, logvar)
            return sample
        else:
            if both_continue:
                latent_sample = []
                latent_sample_cont_var = []
                latent_sample_cont_class = []

                mu_var, logvar_var = latent_dist['cont_var']
                cont_sample_var = reparameterization_trick(mu_var, logvar_var)
                latent_sample_cont_var.append(cont_sample_var)
                latent_sample.append(cont_sample_var)

                mu_class, logvar_class = latent_dist['cont_class']
                cont_sample_class = reparameterization_trick(mu_class, logvar_class)
                latent_sample_cont_class.append(cont_sample_class)
                latent_sample.append(cont_sample_class)
            elif both_discrete:
                latent_sample = []
                latent_sample_disc_var = []
                latent_sample_disc_class = []

                for alpha in latent_dist['disc_var']:
                    disc_sample = self.sample_gumbel_softmax(alpha)
                    latent_sample.append(disc_sample)
                    latent_sample_disc_var.append(disc_sample)

                for alpha in latent_dist['disc_class']:
                    disc_sample = self.sample_gumbel_softmax(alpha)
                    latent_sample.append(disc_sample)
                    latent_sample_disc_class.append(disc_sample)
            else:
                latent_sample = []
                latent_sample_cont = []
                latent_sample_disc = []

                if self.is_continuous:
                    mu, logvar = latent_dist['cont']
                    cont_sample = reparameterization_trick(mu, logvar)
                    latent_sample_cont.append(cont_sample)
                    latent_sample.append(cont_sample)

                if self.is_discrete:
                    for alpha in latent_dist['disc']:
                        disc_sample = self.sample_gumbel_softmax(alpha)
                        latent_sample.append(disc_sample)
                        latent_sample_disc.append(disc_sample)

            # Concatenate continuous and discrete samples into one large sample
            if both_continue:
                return torch.cat(latent_sample, dim=1), torch.cat(latent_sample_cont_var, dim=1), \
                       torch.cat(latent_sample_cont_class, dim=1)
            elif both_discrete:
                return torch.cat(latent_sample, dim=1), torch.cat(latent_sample_disc_var, dim=1), \
                       torch.cat(latent_sample_disc_class, dim=1)
            elif self.is_discrete:
                return torch.cat(latent_sample, dim=1), torch.cat(latent_sample_cont, dim=1), torch.cat(
                    latent_sample_disc,
                    dim=1)
            else:
                return torch.cat(latent_sample, dim=1), torch.cat(latent_sample_cont, dim=1), 0

    def sample_gumbel_softmax(self, alpha):
        """
        Samples from a gumbel-softmax distribution using the reparameterization trick.
        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (N, D)
        """
        if self.training:
            # Sample from gumbel distribution
            unif = torch.rand(alpha.size())
            if torch.cuda.is_available():
                unif = unif.cuda()
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            # reparameterization to create gumbel softmax sample
            log_alpha = torch.log(alpha + EPS)
            # self.temperature: to create more or less big gap between values in logit
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1)
        else:
            # In reconstruction mode, pick most likely sample
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha.
            # Note the view is because scatter_ only accepts 2D tensors.
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
            if torch.cuda.is_available():
                one_hot_samples = one_hot_samples.cuda()
            return one_hot_samples

    def _decode(self, z):
        return self.decoder(z)

    def classification(self, latent_sample):
        """
        classify sample from latent distribution into an image.
        Parameters
        ----------
        latent_sample : torch.Tensor
            Sample from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
        """
        output = self.L3_classifier(latent_sample)
        # print("output:", output)
        predict = F.log_softmax(output, dim=1)
        # print("predict:",  predict)

        return predict

    def representation_random_continue(self, latent_class, latent_var_dim, is_random_class=False):
        """
        classify sample from latent distribution into an image.
        Parameters
        ----------
        latent_class : torch.Tensor
            Sample from latent distribution. Shape (N, L) where L is dimension
            of latent distribution.
            :param latent_class:
            :param is_random_class: if randomise latent representaiton for class (rather than variability
            latent representation
            :param latent_var_dim:
        """
        latent_random_variability = []
        random_variability_latent = torch.randn((latent_class.shape[0], latent_var_dim))
        random_variability_latent = random_variability_latent.to(self.device)

        if is_random_class:
            latent_random_variability.append(latent_class)
            latent_random_variability.append(random_variability_latent)
        else:
            latent_random_variability.append(random_variability_latent)
            latent_random_variability.append(latent_class)

        latent_sample_random_continue = torch.cat(latent_random_variability, dim=1)

        return latent_sample_random_continue

    def classification_random_continue(self, latent_class, latent_var_dim, is_random_class=False):

        latent_sample_random_continue = self.representation_random_continue(latent_class, latent_var_dim,
                                                                            is_random_class=is_random_class)
        predicted_random_variability = self.L3_classifier(latent_sample_random_continue)
        return F.log_softmax(predicted_random_variability, dim=1), latent_sample_random_continue

    def classification_struct_noised(self, latent_sample_random_continue, nb_repeat=100):

        z_struct_noised_multiple = []
        for repeat in range(nb_repeat):
            z_struct_noised = []
            for i in range(self.latent_struct_dim):
                z_rand_var = latent_sample_random_continue.clone()
                z_rand_var[:, self.latent_var_dim + i] = torch.randn((z_rand_var.shape[0]))
                pred = self.L3_classifier(z_rand_var)
                z_struct_noised.append(np.array(F.log_softmax(pred, dim=1).cpu()))  # shape: (latent_struct_dim, batch_size, nb_class)
            z_struct_noised_multiple.append(np.array(z_struct_noised))  # shape: (nb_repeat, latent_struct_dim, batch_size, nb_class)
        z_struct_noised_multiple = np.array(z_struct_noised_multiple)
        return z_struct_noised_multiple

    def representation_random_discrete(self, latent_sample_cont, both_discrete=False, is_random_var=False):

        latent_random_discrete = []
        batch_size = latent_sample_cont.shape[0]

        if both_discrete:
            if is_random_var:
                latent_disc_len = self.latent_var_dim
                nb_alpha = len(self.latent_spec['disc_var'])
            else:
                latent_disc_len = self.latent_struct_dim
                nb_alpha = len(self.latent_spec['disc_class'])
            alpha_size = self.latent_spec['disc_var'][0]
        else:
            latent_disc_len = self.latent_struct_dim
            alpha_size = self.latent_spec['disc'][0]
            nb_alpha = len(self.latent_spec['disc'])

        random_discrete_latent = np.zeros((batch_size, latent_disc_len))

        for i in range(nb_alpha):
            start = i * alpha_size
            end = (i + 1) * alpha_size
            random_discrete_latent[:, start:end] = np.eye(alpha_size)[np.random.choice(alpha_size, batch_size)]

        random_discrete_latent = torch.FloatTensor(random_discrete_latent)
        random_discrete_latent = random_discrete_latent.to(self.device)

        if is_random_var:
            # in this case, we randomize variability part of all discrete latent representation
            latent_random_discrete.append(random_discrete_latent)
            latent_random_discrete.append(latent_sample_cont)
        else:
            latent_random_discrete.append(latent_sample_cont)
            latent_random_discrete.append(random_discrete_latent)

        latent_sample_random_discrete = torch.cat(latent_random_discrete, dim=1)

        return latent_sample_random_discrete

    def classification_random_discrete(self, latent_sample_cont, both_discrete=False, is_random_var=False):

        latent_sample_random_discrete = self.representation_random_discrete(latent_sample_cont,
                                                                            both_discrete=both_discrete,
                                                                            is_random_var=is_random_var)

        predicted_random_discrete = self.L3_classifier(latent_sample_random_discrete)

        return F.log_softmax(predicted_random_discrete, dim=1)

    def classification_zc_pert_zd(self, latent_sample_cont, latent_sample_disc, both_continue=False, reverse=False):
        """
        classify sample from latent distribution with latent continue code pertubed:
        10 codes with 10% then 20% then .... 100% of latent code random
        """
        predicted_partial_random_continue = []
        if both_continue:
            if reverse:
                latent_cont_size = self.latent_struct_dim
            else:
                latent_cont_size = self.latent_var_dim
        else:
            latent_cont_size = self.latent_var_dim

        for i in range(10):
            latent_random_continue = []
            random_size = int(latent_cont_size * (0.1 * i))
            if reverse:
                latent_cont_size = self.latent_struct_dim
                latent_sample_disc[:, :random_size] = torch.randn((latent_sample_cont.shape[0], random_size))
                partial_random_struct_latent = latent_sample_disc.to(self.device)

                latent_random_continue.append(latent_sample_cont)
                latent_random_continue.append(partial_random_struct_latent)
                latent_sample = torch.cat(latent_random_continue, dim=1)
            else:
                latent_sample_cont[:, :random_size] = torch.randn((latent_sample_cont.shape[0], random_size))
                partial_random_continue_latent = latent_sample_cont.to(self.device)

                # print("random perturbate continue latent:", i, partial_random_continue_latent[0])
                latent_random_continue.append(partial_random_continue_latent)
                latent_random_continue.append(latent_sample_disc)
                latent_sample = torch.cat(latent_random_continue, dim=1)

            predicted_partial_random_cont = self.L3_classifier(latent_sample)
            predicted_partial_random_continue.append(F.log_softmax(predicted_partial_random_cont, dim=1))

        return predicted_partial_random_continue

    def classification_zc_zd_pert(self, latent_sample_cont, latent_sample_disc):
        """
        classify sample from latent distribution with latent discrete code pertub:
        10 codes with 10% then 20% then .... 100% of latent code random
        """
        predicted_partial_random_discrete = []
        latent_disc_size = self.latent_struct_dim
        alpha_size = self.latent_spec['disc'][0]
        batch_size = latent_sample_cont.shape[0]
        sample_disc = np.array(latent_sample_disc.cpu())

        for i in range(10):
            latent_random_discrete = []
            nb_alpha = int(latent_disc_size * (0.1 * i) / alpha_size)
            random_discrete_latent = sample_disc

            for j in range(nb_alpha):
                start = j * alpha_size
                end = (j + 1) * alpha_size
                random_discrete_latent[:, start:end] = np.eye(alpha_size)[np.random.choice(alpha_size, batch_size)]

            random_discrete_latent = torch.FloatTensor(random_discrete_latent)
            random_discrete_latent = random_discrete_latent.to(self.device)

            # print("random perturb discrete latent:", i, random_discrete_latent[0])
            latent_random_discrete.append(latent_sample_cont)
            latent_random_discrete.append(random_discrete_latent)

            latent_sample = torch.cat(latent_random_discrete, dim=1)
            predicted_random_discrete = self.L3_classifier(latent_sample)

            predicted_partial_random_discrete.append(F.log_softmax(predicted_random_discrete, dim=1))

        return predicted_partial_random_discrete

    def representation_partial_random_class(self, latent_class, latent_var_dim, latent_class_dim,
                                            random_percentage=0.5, both_continue=False):
        """
        randomize x percent of latent vector.
        :param both_continue: 
        :param latent_class_dim: 
        :param latent_var_dim: 
        :param latent_class: 
        :param random_percentage: percent between 0 and 1
        :return:
        """
        batch_size = latent_class.shape[0]

        if both_continue:
            latent_partial_random_class = []
            random_variability_latent = torch.randn((batch_size, latent_var_dim))
            random_variability_latent = random_variability_latent.to(self.device)

            # calculate number of latent_class's neurons must have randomize
            number_rand_neurons = int(random_percentage * latent_class_dim)

            # choose number_rand_number random neurone in latent_class:
            random_index = random.sample(range(latent_class_dim), number_rand_neurons)

            for i in random_index:
                latent_class[:, i] = torch.randn(1)

            latent_partial_random_class.append(random_variability_latent)
            latent_partial_random_class.append(latent_class)
        else:
            # TODO: create partial random vector for other representation
            pass

        latent_partial_random_class = torch.cat(latent_partial_random_class, dim=1)

        return latent_partial_random_class

    def classification_partial_random_class(self, latent_class, latent_var_dim, latent_class_dim,
                                            random_percentage=0.5, both_continue=False):

        latent_partial_random_class = self.representation_partial_random_class(latent_class,
                                                                               latent_var_dim,
                                                                               latent_class_dim,
                                                                               random_percentage=random_percentage,
                                                                               both_continue=both_continue)
        predicted_partial_random_class = self.L3_classifier(latent_partial_random_class)

        return F.log_softmax(predicted_partial_random_class, dim=1)


class PrintLayer(nn.Module, ABC):

    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()
