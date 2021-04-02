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
                 var_second_cnn_block=False,
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

        """
        # reshape size compute:
        w1 = ((32 - 3 + (2 * 0)) / 1) + 1 - EPS
        self.width_conv1_size = round(w1)
        w2 = ((self.width_conv1_size - 3 + (2 * 0)) / 1) + 1 - EPS
        self.width_conv2_size = round(w2)
        # computing padding size for have output shape equal to input shape (equivalent to padding="same"):
        self.padding_size_1 = ((self.width_conv2_size-1)*2 - self.width_conv2_size + 5)/2 + EPS
        self.padding_size_1 = round(self.padding_size_1)
        w3 = ((self.width_conv2_size - 5 + (2 * self.padding_size_1)) / 2) + 1 - EPS
        self.width_conv3_size = round(w3)
        
        if self.var_second_cnn_block:
            w4 = ((self.width_conv3_size - 3 + (2 * 0)) / 1) + 1 - EPS
            self.width_conv4_size = round(w4)
            w5 = ((self.width_conv4_size - 3 + (2 * 0)) / 1) + 1 - EPS
            self.width_conv5_size = round(w5)
            # computing padding size for have output shape equal to input shape (equivalent to padding="same"):
            self.padding_size_2 = ((self.width_conv5_size - 1) * 2 - self.width_conv5_size + 5) / 2 + EPS
            self.padding_size_2 = round(self.padding_size_2)
            w6 = ((self.width_conv5_size - 5 + (2 * self.padding_size_2)) / 2) + 1 - EPS
            self.width_conv6_size = round(w6)

        if self.var_third_cnn_block:
            w7 = ((self.width_conv6_size - 4 + (2 * 0)) / 1) + 1 - EPS
            self.width_conv7_size = round(w7)

        if self.var_second_cnn_block and not self.var_third_cnn_block:
            self.var_reshape = (64, self.width_conv6_size, self.width_conv6_size)
        elif self.var_third_cnn_block:
            self.var_reshape = (128, self.width_conv7_size, self.width_conv7_size)
        else:
            self.var_reshape = (32, self.width_conv3_size, self.width_conv3_size)
        """

        # Other architecture:
        # reshape size compute:
        # ((w-1)*s - w + f)/2
        self.padding_size_1 = ((32 - 1) * 1 - 32 + 3) / 2 + EPS
        self.padding_size_1 = round(self.padding_size_1)
        w1 = ((32 - 3 + (2 * self.padding_size_1)) / 1) + 1 - EPS
        self.width_conv1_size = round(w1)

        self.padding_size_2 = ((self.width_conv1_size - 1) * 1 - self.width_conv1_size + 3) / 2 + EPS
        self.padding_size_2 = round(self.padding_size_2)
        w2 = ((self.width_conv1_size - 3 + (2 * self.padding_size_2)) / 2) + 1 - EPS
        self.width_conv2_size = round(w2)

        self.padding_size_3 = ((self.width_conv2_size - 1) * 1 - self.width_conv2_size + 3) / 2 + EPS
        self.padding_size_3 = round(self.padding_size_3)
        w3 = ((self.width_conv2_size - 3 + (2 * self.padding_size_3)) / 1) + 1 - EPS
        self.width_conv3_size = round(w3)

        self.padding_size_4 = ((self.width_conv3_size - 1) * 1 - self.width_conv3_size + 3) / 2 + EPS
        self.padding_size_4 = round(self.padding_size_4)
        w4 = ((self.width_conv3_size - 3 + (2 * self.padding_size_4)) / 1) + 1 - EPS
        self.width_conv4_size = round(w4)

        self.var_reshape = (64, self.width_conv4_size, self.width_conv4_size)

        print(self.var_reshape, self.width_conv1_size, self.width_conv2_size, self.width_conv3_size,
              self.width_conv4_size)

        # -----------_________________ define model: encoder var ____________________________________________--------
        # self.encoder_var = [
        #     nn.Conv2d(in_channels=self.nc, out_channels=32, kernel_size=3, stride=1),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(32),
        #     PrintLayer(),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(32),
        #     PrintLayer(),
        #     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=self.padding_size_1),
        #     nn.ReLU(True),
        #     nn.BatchNorm2d(32),
        #     nn.Dropout(0.4),
        #     PrintLayer(),
        # ]

        self.encoder_var = [
            nn.Conv2d(in_channels=self.nc, out_channels=32, kernel_size=3, stride=1, padding=self.padding_size_1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            # PrintLayer(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=self.padding_size_2),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            # PrintLayer(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=self.padding_size_3),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=self.padding_size_4),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.4),
            # PrintLayer(),
        ]

        if self.var_second_cnn_block:
            self.encoder_var += [
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
                nn.ReLU(True),
                nn.BatchNorm2d(64),
                # PrintLayer(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                nn.ReLU(True),
                nn.BatchNorm2d(64),
                # PrintLayer(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=2, padding=self.padding_size_2),
                nn.ReLU(True),
                nn.BatchNorm2d(64),
                nn.Dropout(0.4),
                # PrintLayer(),
            ]

        if self.var_third_cnn_block:
            self.encoder_var += [
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1),
                nn.ReLU(True),
                nn.BatchNorm2d(128),
                PrintLayer(),
            ]

        self.encoder_var += [
            View((-1, np.product(self.var_reshape))),
            nn.Dropout(0.4),
            # PrintLayer(),
            nn.Linear(np.product(self.var_reshape), 32),
            nn.ReLU(True),
            # PrintLayer(),
            nn.Linear(32, self.z_var_size * 2)
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
                nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=1),
                nn.ReLU(True),
                PrintLayer(),
            ]

        if self.var_second_cnn_block:
            self.decoder_var += [
                nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2,
                                   padding=self.padding_size_2-1),
                nn.ReLU(True),
                PrintLayer(),
                nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
                nn.ReLU(True),
                PrintLayer(),
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=1),
                nn.ReLU(True),
                PrintLayer(),
            ]

        # self.decoder_var += [
        #     nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2,
        #                        padding=self.padding_size_1-1),
        #     nn.ReLU(True),
        #     PrintLayer(),
        #     nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
        #     nn.ReLU(True),
        #     PrintLayer(),
        #     nn.ConvTranspose2d(in_channels=32, out_channels=self.nc, kernel_size=3, stride=1),
        #     nn.ReLU(True),
        #     PrintLayer(),
        #     nn.Sigmoid()
        # ]

        self.decoder_var += [
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                               padding=self.padding_size_4),
            nn.ReLU(True),
            # PrintLayer(),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                               padding=self.padding_size_3),
            nn.ReLU(True),
            # PrintLayer(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=self.padding_size_2),
            nn.ReLU(True),
            # PrintLayer(),
            nn.ConvTranspose2d(in_channels=32, out_channels=self.nc, kernel_size=4, stride=1,
                               padding=self.padding_size_1),
            nn.ReLU(True),
            # PrintLayer(),
            nn.Sigmoid()
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
