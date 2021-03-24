from abc import ABC
import torch.nn as nn
from custom_Layer import Flatten, View, PrintLayer, kaiming_init
import numpy as np
from torch.autograd import Variable
import torch

EPS = 1e-12


class VAE(nn.Module, ABC):

    def __init__(self,
                 z_struct_size=32,
                 big_kernel_size=8,
                 stride_size=1,
                 hidden_filters_1=32,
                 hidden_filters_2=64,
                 hidden_filters_3=64,
                 BK_in_first_layer=True,
                 two_conv_layer=False,
                 three_conv_layer=False,
                 BK_in_second_layer=False,
                 BK_in_third_layer=False,
                 z_var_size=5,
                 var_hidden_filters_1=32,
                 var_hidden_filters_2=32,
                 var_hidden_filters_3=32,
                 var_kernel_size_1=3,
                 var_kernel_size_2=3,
                 var_kernel_size_3=3,
                 var_stride_size_1=1,
                 var_stride_size_2=1,
                 var_stride_size_3=1,
                 var_hidden_dim=256,
                 var_three_conv_layer=False):
        """
        Class which defines model and forward pass.
        """
        super(VAE, self).__init__()

        # struct encoder parameters:
        self.nc = 1  # number of channels
        self.hidden_filters_1 = hidden_filters_1
        self.hidden_filters_2 = hidden_filters_2
        self.hidden_filters_3 = hidden_filters_3
        self.kernel_size_1 = 3
        self.kernel_size_2 = 4
        self.kernel_size_3 = 4
        self.n_classes = 10
        self.hidden_filter_GMP = self.hidden_filters_1
        # custom parameters:
        self.z_struct_size = z_struct_size
        self.big_kernel_size = big_kernel_size
        self.stride_size = stride_size
        self.last_linear_layer_size = self.z_struct_size
        self.BK_in_first_layer = BK_in_first_layer
        self.BK_in_second_layer = BK_in_second_layer
        self.BK_in_third_layer = BK_in_third_layer
        self.two_conv_layer = two_conv_layer
        self.three_conv_layer = three_conv_layer

        # Var encoder parameters:
        self.padding = 0
        self.z_var_size = z_var_size
        self.var_hidden_filters_1 = var_hidden_filters_1
        self.var_hidden_filters_2 = var_hidden_filters_2
        self.var_hidden_filters_3 = var_hidden_filters_3
        self.var_kernel_size_1 = var_kernel_size_1
        self.var_kernel_size_2 = var_kernel_size_2
        self.var_kernel_size_3 = var_kernel_size_3
        self.var_stride_size_1 = var_stride_size_1
        self.var_stride_size_2 = var_stride_size_2
        self.var_stride_size_3 = var_stride_size_3
        self.var_hidden_dim = var_hidden_dim
        self.var_three_conv_layer = var_three_conv_layer

        # reshape size compute:
        w1 = ((32 - self.var_kernel_size_1 + (2 * self.padding)) / self.var_stride_size_1) + 1 - EPS
        self.width_conv1_size = round(w1)
        w2 = ((self.width_conv1_size - self.var_kernel_size_2 + (2 * self.padding)) / self.var_stride_size_2) + 1 - EPS
        self.width_conv2_size = round(w2)
        w3 = ((self.width_conv2_size - self.var_kernel_size_3 + (2 * self.padding)) / self.var_stride_size_3) + 1 - EPS
        self.width_conv3_size = round(w3)

        if self.var_three_conv_layer:
            self.var_reshape = (var_hidden_filters_3, self.width_conv3_size, self.width_conv3_size)
        else:
            self.var_reshape = (var_hidden_filters_2, self.width_conv2_size, self.width_conv2_size)

        # decoder:
        self.z_size = self.z_var_size + self.z_struct_size

        if self.BK_in_first_layer:
            self.kernel_size_1 = self.big_kernel_size
        elif self.BK_in_second_layer:
            self.kernel_size_2 = self.big_kernel_size
        elif self.BK_in_third_layer:
            self.kernel_size_3 = self.big_kernel_size

        # -----------_________________ define model: encoder_struct____________________________________________--------
        # ----------- add conv bloc:
        self.encoder_struct = [
            nn.Conv2d(self.nc, self.hidden_filters_1, self.kernel_size_1, stride=self.stride_size),
            nn.BatchNorm2d(self.hidden_filters_1),
            nn.ReLU(True),
            # PrintLayer(),  # B, 32, 25, 25
        ]
        if self.two_conv_layer:
            self.hidden_filter_GMP = self.hidden_filters_2
            numChannels = self.hidden_filters_2
            self.encoder_struct += [
                nn.Conv2d(self.hidden_filters_1, self.hidden_filters_2, self.kernel_size_2, stride=self.stride_size),
                nn.BatchNorm2d(self.hidden_filters_2),
                nn.ReLU(True),
                # PrintLayer(),  # B, 32, 25, 25
            ]
        if self.three_conv_layer:
            self.hidden_filter_GMP = self.hidden_filters_3
            numChannels = self.hidden_filters_3
            self.encoder_struct += [
                nn.Conv2d(self.hidden_filters_2, self.hidden_filters_3, self.kernel_size_3, stride=self.stride_size),
                nn.BatchNorm2d(self.hidden_filters_3),
                nn.ReLU(True),
                # PrintLayer(),  # B, 32, 25, 25
            ]
        # ----------- add GMP bloc:
        self.encoder_struct += [
            nn.AdaptiveMaxPool2d((1, 1)),  # B, hidden_filters_1, 1, 1
            # PrintLayer(),  # B, hidden_filters_1, 1, 1
            View((-1, self.hidden_filter_GMP)),  # B, hidden_filters_1
            # PrintLayer(),  # B, hidden_filters_1
        ]
        # -----------_________________ end encoder_struct____________________________________________------------

        # -----------_________________ define model: encoder_var____________________________________________--------
        self.encoder_var = [
            # PrintLayer(),
            nn.Conv2d(self.nc, self.var_hidden_filters_1, self.var_kernel_size_1, stride=self.var_stride_size_1),
            nn.BatchNorm2d(self.var_hidden_filters_1),
            nn.ReLU(True),
            # PrintLayer(),
            nn.Conv2d(self.var_hidden_filters_1, self.var_hidden_filters_2, self.var_kernel_size_2,
                      stride=self.var_stride_size_2),
            nn.BatchNorm2d(self.var_hidden_filters_2),
            nn.ReLU(True),
            # PrintLayer(),
        ]

        if self.var_three_conv_layer:
            self.encoder_var += [
                nn.Conv2d(self.var_hidden_filters_2, self.var_hidden_filters_3, self.var_kernel_size_3,
                          stride=self.var_stride_size_3),
                nn.BatchNorm2d(self.var_hidden_filters_3),
                nn.ReLU(True),
                # PrintLayer(),
            ]

        self.encoder_var += [
            View((-1, np.product(self.var_reshape))),
            # PrintLayer(),
            nn.Linear(np.product(self.var_reshape), self.var_hidden_dim),
            nn.BatchNorm1d(self.var_hidden_dim),
            nn.ReLU(True),
            nn.Dropout(0.4),
            # PrintLayer(),
            nn.Linear(self.var_hidden_dim, self.z_var_size * 2),
            # PrintLayer()
        ]
        # --------------------------------------- end encoder_var____________________________________________ ----

        # -----------_________________ define model: decoder ____________________________________________--------
        self.decoder = [
            # PrintLayer(),
            nn.Linear(self.z_size, self.var_hidden_dim),
            nn.ReLU(True),
            # PrintLayer(),
            nn.Linear(self.var_hidden_dim, np.product(self.var_reshape)),
            nn.ReLU(True),
            # PrintLayer(),
            View((-1, *self.var_reshape)),
            # PrintLayer(),
        ]
        if self.var_three_conv_layer:
            self.decoder += [
                nn.ConvTranspose2d(self.var_hidden_filters_3,
                                   self.var_hidden_filters_2,
                                   self.var_kernel_size_3,
                                   stride=self.var_stride_size_3),
                nn.ReLU(True),
                # PrintLayer(),
            ]

        self.decoder += [
            nn.ConvTranspose2d(self.var_hidden_filters_2,
                               self.var_hidden_filters_1,
                               self.var_kernel_size_2,
                               stride=self.var_stride_size_2),
            nn.ReLU(True),
            # PrintLayer(),
            nn.ConvTranspose2d(self.var_hidden_filters_1,
                               self.nc,
                               self.var_kernel_size_1,
                               stride=self.var_stride_size_1),
            # PrintLayer(),
            nn.Sigmoid()
        ]
        # --------------------------------------- end decoder ____________________________________________ ----

        self.encoder_struct = nn.Sequential(*self.encoder_struct)
        self.encoder_var = nn.Sequential(*self.encoder_var)
        self.decoder = nn.Sequential(*self.decoder)

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                # weight_init(m)
                kaiming_init(m)

    def forward(self, x):
        """
        Forward pass of model.
        """

        # z_struct:
        z_struct = self.encoder_struct(x)

        # z_var:
        z_var = self.encoder_var(x)
        latent_representation = self._encode(z_var, self.z_var_size)
        z_var_sample = self.reparametrize(latent_representation)

        # concatenate z_struct and z_var:
        z = torch.cat((z_var_sample, z_struct), dim=1)
        assert z.shape[-1] == self.z_size, "z concatenation doesn't match with expected z size"

        x_recons = self.decoder(z)

        return x_recons, z_struct, z_var, z_var_sample, latent_representation, z

    def _encode(self, z, z_size):
        """
        Encodes an image into parameters of a latent distribution.
        Parameters
        ----------
        x : torch.Tensor
            Batch of data, shape (N, C, H, W)
        """
        latent_dist = {}

        mu = z[:, :z_size]
        logvar = z[:, z_size:]

        latent_dist['mu'] = mu
        latent_dist['logvar'] = logvar

        return latent_dist

    def reparametrize(self, latent_dist):
        """
        Samples from a normal distribution using the reparameterization trick.
        :param mu: torch.Tensor
                Mean of the normal distribution. Shape (batch_size, latent_dim)
        :param logvar: torch.Tensor
                Diagonal log variance of the normal distribution. Shape (batch_size,
                latent_dim)
        :return:
        """

        mu = latent_dist['mu']
        logvar = latent_dist['logvar']

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        sample = mu + std * eps

        return sample
