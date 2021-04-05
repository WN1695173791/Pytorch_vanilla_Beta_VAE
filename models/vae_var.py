from abc import ABC
import torch.nn as nn
from custom_Layer import Flatten, View, PrintLayer, kaiming_init
import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-12


class VAE_var(nn.Module, ABC):

    def __init__(self,
                 z_var_size=5,
                 var_second_cnn_block=True,
                 var_third_cnn_block=False):
        """
        Class which defines model and forward pass.
        """
        super(VAE_var, self).__init__()

        # parameters:
        self.nc = 1
        self.z_var_size = z_var_size

        # number of CNN blocks:
        self.var_second_cnn_block = var_second_cnn_block
        self.var_third_cnn_block = var_third_cnn_block

        # reshape size compute:
        # computing padding size for have output shape equal to input shape (equivalent to padding="same"):
        # Same padding means the size of output feature-maps are the same as the input feature-maps 
        # (under the assumption of stride=1)
        # Formula: P = ((W-1)S - W + F) / 2 and with stride=1 we have: P = (F-1)/2
        self.padding_size = (3-1)/2
        self.padding_size = round(self.padding_size)
        
        # output tensor shape: ((W-F + 2P)/S) +1
        w1 = ((32 - 3 + (2 * self.padding_size)) / 2) + 1 - EPS
        self.width_conv1_size = round(w1)
        if self.var_second_cnn_block:
            w2 = ((self.width_conv1_size - 3 + (2 * self.padding_size)) / 2) + 1 - EPS
            self.width_conv2_size = round(w2)
        if self.var_third_cnn_block:
            w3 = ((self.width_conv2_size - 3 + (2 * self.padding_size)) / 1) + 1 - EPS
            self.width_conv3_size = round(w3)

        if self.var_second_cnn_block and not self.var_third_cnn_block:
            self.var_reshape = (64, self.width_conv2_size, self.width_conv2_size)
        elif self.var_third_cnn_block:
            self.var_reshape = (64, self.width_conv3_size, self.width_conv3_size)
        else:
            self.var_reshape = (32, self.width_conv3_size, self.width_conv3_size)

        # -----------_________________ define model: encoder var ____________________________________________--------
        self.encoder_var = [
            nn.Conv2d(in_channels=self.nc, out_channels=32, kernel_size=3, stride=2, padding=self.padding_size),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            # PrintLayer(),
        ]

        if self.var_second_cnn_block:
            self.encoder_var += [
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=self.padding_size),
                nn.ReLU(True),
                nn.BatchNorm2d(64),
                # PrintLayer(),
            ]

        if self.var_third_cnn_block:
            self.encoder_var += [
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=self.padding_size),
                nn.ReLU(True),
                nn.BatchNorm2d(64),
                # PrintLayer(),
            ]

        self.encoder_var += [
            View((-1, np.product(self.var_reshape))),
            nn.Dropout(0.4),
            # PrintLayer(),
            nn.Linear(np.product(self.var_reshape), 128),
            nn.ReLU(True),
            # PrintLayer(),
            nn.Linear(128, self.z_var_size * 2),
            # PrintLayer(),
        ]
        # --------------------------------------- end encoder_var____________________________________________ ----

        # -----------_________________ define model: decoder_var ____________________________________________--------
        self.decoder_var = [
            # PrintLayer(),
            nn.Linear(self.z_var_size, np.product(self.var_reshape)),
            nn.ReLU(True),
            # PrintLayer(),
            View((-1, *self.var_reshape)),
            # PrintLayer(),
        ]

        if self.var_third_cnn_block:
            self.decoder_var += [
                nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                   padding=self.padding_size),
                nn.ReLU(True),
                # PrintLayer(),
            ]

        if self.var_second_cnn_block:
            self.decoder_var += [
                nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                                   padding=self.padding_size),
                nn.ReLU(True),
                # PrintLayer(),
            ]

        self.decoder_var += [
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2,
                               padding=self.padding_size),
            nn.ReLU(True),
            # PrintLayer(),
            nn.ConvTranspose2d(in_channels=32, out_channels=self.nc, kernel_size=3, stride=1,
                               padding=self.padding_size),
            nn.Sigmoid(),
            # PrintLayer(),
        ]
        # --------------------------------------- end decoder ____________________________________________ ----

        self.encoder_var = nn.Sequential(*self.encoder_var)
        self.decoder_var = nn.Sequential(*self.decoder_var)

        # weights initialization:
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
        # z_var:
        z_var = self.encoder_var(x)
        latent_representation = self._encode(z_var, self.z_var_size)
        z_var_sample = self.reparametrize(latent_representation)

        # reconstruction:
        x_recons = self.decoder_var(z_var_sample)

        return x_recons, latent_representation

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
        log_var = z[:, z_size:]

        latent_dist['mu'] = mu
        latent_dist['log_var'] = log_var

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
        log_var = latent_dist['log_var']

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        sample = mu + (std * eps)

        return sample
