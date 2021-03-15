from abc import ABC
import torch.nn as nn
from custom_Layer import Flatten, View, PrintLayer, kaiming_init
import numpy as np


class Encoder_decoder(nn.Module, ABC):

    def __init__(self,
                 z_struct_size=5,
                 big_kernel_size=8,
                 stride_size=1,
                 classif_layer_size=30,
                 add_classification_layer=False,
                 hidden_filters_1=32,
                 hidden_filters_2=64,
                 hidden_filters_3=64,
                 BK_in_first_layer=True,
                 two_conv_layer=False,
                 three_conv_layer=False,
                 BK_in_second_layer=False,
                 BK_in_third_layer=False,
                 Binary_z=False,
                 add_linear_after_GMP=True,
                 before_GMP_shape=None,
                 other_architecture=False):
        """
        Class which defines model and forward pass.
        """
        super(Encoder_decoder, self).__init__()
        self.other_architecture = other_architecture
        self.before_GMP_shape = before_GMP_shape
        self.add_linear_after_GMP = add_linear_after_GMP
        self.Binary_z = Binary_z
        # parameters
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
        self.classif_layer_size = classif_layer_size
        self.add_classification_layer = add_classification_layer
        self.last_linear_layer_size = self.z_struct_size
        self.BK_in_first_layer = BK_in_first_layer
        self.BK_in_second_layer = BK_in_second_layer
        self.BK_in_third_layer = BK_in_third_layer
        self.two_conv_layer = two_conv_layer
        self.three_conv_layer = three_conv_layer

        # decoder:
        self.hidden_dim = 36
        self.width_first_conv_decoder = int(np.sqrt(self.hidden_dim))
        self.reshape = (1, self.width_first_conv_decoder, self.width_first_conv_decoder)  # (1, 7, 7)

        if self.BK_in_first_layer:
            self.kernel_size_1 = self.big_kernel_size
        elif self.BK_in_second_layer:
            self.kernel_size_2 = self.big_kernel_size
        elif self.BK_in_third_layer:
            self.kernel_size_3 = self.big_kernel_size

        # -----------_________________ define model: Encoder____________________________________________------------
        # ----------- add conv bloc:
        self.encoder = [
            nn.Conv2d(self.nc, self.hidden_filters_1, self.kernel_size_1, stride=self.stride_size),
            nn.BatchNorm2d(self.hidden_filters_1),
            nn.ReLU(True),
            # PrintLayer(),  # B, 32, 25, 25
        ]

        if self.two_conv_layer:
            self.hidden_filter_GMP = self.hidden_filters_2
            numChannels = self.hidden_filters_2
            self.encoder += [
                nn.Conv2d(self.hidden_filters_1, self.hidden_filters_2, self.kernel_size_2, stride=self.stride_size),
                nn.BatchNorm2d(self.hidden_filters_2),
                nn.ReLU(True),
                # PrintLayer(),  # B, 32, 25, 25
            ]

        if self.three_conv_layer:
            self.hidden_filter_GMP = self.hidden_filters_3
            numChannels = self.hidden_filters_3
            self.encoder += [
                nn.Conv2d(self.hidden_filters_2, self.hidden_filters_3, self.kernel_size_3, stride=self.stride_size),
                nn.BatchNorm2d(self.hidden_filters_3),
                nn.ReLU(True),
                # PrintLayer(),  # B, 32, 25, 25
            ]

        # ----------- add GMP bloc:
        self.encoder += [
            nn.AdaptiveMaxPool2d((1, 1)),  # B, hidden_filters_1, 1, 1
            # PrintLayer(),  # B, hidden_filters_1, 1, 1
            View((-1, self.hidden_filter_GMP)),  # B, hidden_filters_1
            # PrintLayer(),  # B, hidden_filters_1
        ]

        # -----------_________________ end Encoder____________________________________________------------
        # ----------- Define decoder: ------------
        self.z_struct_size = self.hidden_filter_GMP
        self.decoder = [
            nn.Linear(self.z_struct_size, self.hidden_dim),  # B, 36
            nn.ReLU(True),
            # PrintLayer(),
            View((-1, *self.reshape)),  # B, 1, 6, 6
            # PrintLayer(),
            nn.ConvTranspose2d(1, 64, 4, stride=2),  # B, 64, 14, 14
            # nn.BatchNorm2d(64),
            nn.ReLU(True),
            # PrintLayer(),
            nn.ConvTranspose2d(64, 32, 3, stride=2),  # B, 32, 29, 29
            # nn.BatchNorm2d(32),
            nn.ReLU(True),
            # PrintLayer(),
            nn.ConvTranspose2d(32, 1, 4, stride=1),  # B, 1, 32, 32
            # PrintLayer(),
            nn.Sigmoid()
        ]
        # --------------------------------------- end decoder -----------------------------------

        # self.net = nn.Sequential(*self.encoder,
        #                          *self.decoder)

        self.encoder = nn.Sequential(*self.encoder)
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

        z_struct = self.encoder(x)
        x_recons = self.decoder(z_struct)

        return x_recons, z_struct
