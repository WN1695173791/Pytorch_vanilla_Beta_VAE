from abc import ABC
import torch.nn as nn
from custom_Layer import Flatten, View, PrintLayer, kaiming_init
import numpy as np

EPS = 1e-12


class Encoder_decoder(nn.Module, ABC):

    def __init__(self,
                 z_struct_size=5,
                 big_kernel_size=8,
                 stride_size=1,
                 classif_layer_size=30,
                 add_classification_layer=False,
                 hidden_filters_1=32,
                 hidden_filters_2=32,
                 hidden_filters_3=32,
                 BK_in_first_layer=True,
                 two_conv_layer=True,
                 three_conv_layer=False,
                 BK_in_second_layer=False,
                 BK_in_third_layer=False,
                 Binary_z=False,
                 add_linear_after_GMP=True,
                 before_GMP_shape=None,
                 other_architecture=False,
                 decoder_first_dense=36,
                 decoder_n_filter_1=32,
                 decoder_n_filter_2=32,
                 decoder_n_filter_3=32,
                 decoder_kernel_size_1=4,
                 decoder_kernel_size_2=3,
                 decoder_kernel_size_3=4,
                 decoder_stride_1=2,
                 decoder_stride_2=2,
                 decoder_stride_3=1,
                 struct_hidden_filters_1=32,
                 struct_hidden_filters_2=32,
                 struct_hidden_filters_3=32,
                 struct_kernel_size_1=3,
                 struct_kernel_size_2=3,
                 struct_kernel_size_3=3,
                 struct_stride_size_1=1,
                 struct_stride_size_2=1,
                 struct_stride_size_3=1,
                 struct_hidden_dim=256,
                 struct_three_conv_layer=False):
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
        self.n_classes = 10
        self.hidden_filters_1 = hidden_filters_1
        self.hidden_filters_2 = hidden_filters_2
        self.hidden_filters_3 = hidden_filters_3

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
        self.decoder_first_dense = decoder_first_dense  # 36
        self.decoder_n_filter_1 = decoder_n_filter_1  # 32
        self.decoder_n_filter_2 = decoder_n_filter_2  # 32
        self.decoder_n_filter_3 = decoder_n_filter_3  # 32
        self.decoder_kernel_size_1 = decoder_kernel_size_1  # 4
        self.decoder_kernel_size_2 = decoder_kernel_size_2  # 3
        self.decoder_kernel_size_3 = decoder_kernel_size_3  # 4
        self.decoder_stride_1 = decoder_stride_1  # 2
        self.decoder_stride_2 = decoder_stride_2  # 2
        self.decoder_stride_3 = decoder_stride_3  # 1

        self.width_first_conv_decoder = int(np.sqrt(self.decoder_first_dense))
        self.reshape = (1, self.width_first_conv_decoder, self.width_first_conv_decoder)  # (1, 7, 7)

        # Other decoder architecture:
        self.padding = 0
        self.struct_hidden_filters_1 = struct_hidden_filters_1
        self.struct_hidden_filters_2 = struct_hidden_filters_2
        self.struct_hidden_filters_3 = struct_hidden_filters_3
        self.struct_kernel_size_1 = struct_kernel_size_1
        self.struct_kernel_size_2 = struct_kernel_size_2
        self.struct_kernel_size_3 = struct_kernel_size_3
        self.struct_stride_size_1 = struct_stride_size_1
        self.struct_stride_size_2 = struct_stride_size_2
        self.struct_stride_size_3 = struct_stride_size_3
        self.struct_hidden_dim = struct_hidden_dim

        self.struct_hidden_filter_GMP = self.z_struct_size
        if self.two_conv_layer:
            self.hidden_filters_2 = self.z_struct_size
        elif self.three_conv_layer:
            self.hidden_filters_3 = self.z_struct_size

        # reshape size compute:
        w1 = ((32 - self.struct_kernel_size_1 + (2 * self.padding)) / self.struct_stride_size_1) + 1 - EPS
        self.width_conv1_size = round(w1)
        w2 = ((self.width_conv1_size - self.struct_kernel_size_2 + (2 * self.padding)) / self.struct_stride_size_2) + 1 - EPS
        self.width_conv2_size = round(w2)
        w3 = ((self.width_conv2_size - self.struct_kernel_size_3 + (2 * self.padding)) / self.struct_stride_size_3) + 1 - EPS
        self.width_conv3_size = round(w3)

        if self.three_conv_layer:
            self.struct_reshape = (struct_hidden_filters_3, self.width_conv3_size, self.width_conv3_size)
        else:
            self.struct_reshape = (struct_hidden_filters_2, self.width_conv2_size, self.width_conv2_size)

        if self.BK_in_first_layer:
            self.struct_kernel_size_1 = self.big_kernel_size
        elif self.BK_in_second_layer:
            self.struct_kernel_size_2 = self.big_kernel_size
        elif self.BK_in_third_layer:
            self.struct_kernel_size_3 = self.big_kernel_size

        # -----------_________________ define model: encoder struct_______________________________________------------
        # ----------- add conv bloc:

        self.encoder = [
            nn.Conv2d(self.nc, self.hidden_filters_1, self.struct_kernel_size_1, stride=self.stride_size),
            nn.BatchNorm2d(self.hidden_filters_1),
            nn.ReLU(True),
            # PrintLayer(),
        ]

        if self.two_conv_layer:
            self.encoder += [
                nn.Conv2d(self.hidden_filters_1, self.hidden_filters_2, self.struct_kernel_size_2, stride=self.stride_size),
                nn.BatchNorm2d(self.hidden_filters_2),
                nn.ReLU(True),
                # PrintLayer(),
            ]
        if self.three_conv_layer:
            self.encoder += [
                nn.Conv2d(self.hidden_filters_2, self.hidden_filters_3, self.struct_kernel_size_3,
                          stride=self.stride_size),
                nn.BatchNorm2d(self.hidden_filters_3),
                nn.ReLU(True),
                # PrintLayer(),
            ]
        # ----------- add GMP bloc:
        self.encoder += [
            nn.AdaptiveMaxPool2d((1, 1)),  # B, hidden_filters_1, 1, 1
            # PrintLayer(),  # B, hidden_filters_1, 1, 1
            View((-1, self.struct_hidden_filter_GMP)),  # B, hidden_filters_1
            # PrintLayer(),  # B, hidden_filters_1
        ]
        # -----------_________________ end encoder____________________________________________------------

        if self.other_architecture:
            self.decoder = [
                # PrintLayer(),
                nn.Linear(self.z_struct_size, self.struct_hidden_dim),
                nn.ReLU(True),
                # PrintLayer(),
                nn.Linear(self.struct_hidden_dim, np.product(self.struct_reshape)),
                nn.ReLU(True),
                # PrintLayer(),
                View((-1, *self.struct_reshape)),
                # PrintLayer(),
            ]

            if self.three_conv_layer:
                self.decoder += [
                    nn.ConvTranspose2d(self.decoder_n_filter_3,
                                       self.decoder_n_filter_2,
                                       self.decoder_kernel_size_3,
                                       stride=self.decoder_stride_3),
                    nn.ReLU(True),
                    # PrintLayer(),
                ]

            self.decoder += [
                nn.ConvTranspose2d(self.decoder_n_filter_2,
                                   self.decoder_n_filter_1,
                                   self.decoder_kernel_size_2,
                                   stride=self.decoder_stride_2),
                nn.ReLU(True),
                # PrintLayer(),
                nn.ConvTranspose2d(self.decoder_n_filter_1,
                                   self.nc,
                                   self.decoder_kernel_size_1,
                                   stride=self.decoder_stride_1),
                # PrintLayer(),
                nn.Sigmoid()
            ]
        else:
            # ----------- Define decoder: ------------
            self.decoder = [
                nn.Linear(self.z_struct_size, self.struct_hidden_dim),  # B, 36
                nn.ReLU(True),
                # PrintLayer(),
                View((-1, *self.struct_reshape)),  # B, 1, 6, 6
                # PrintLayer(),
                nn.ConvTranspose2d(1,
                                   self.decoder_n_filter_1,
                                   self.decoder_kernel_size_1,
                                   stride=self.decoder_stride_1),  # B, 64, 14, 14
                # nn.BatchNorm2d(64),
                nn.ReLU(True),
                # PrintLayer(),
                nn.ConvTranspose2d(self.decoder_n_filter_1,
                                   self.decoder_n_filter_2,
                                   self.decoder_kernel_size_2,
                                   stride=self.decoder_stride_2),  # B, 32, 29, 29
                # nn.BatchNorm2d(32),
                nn.ReLU(True),
                # PrintLayer(),
                nn.ConvTranspose2d(self.decoder_n_filter_2,
                                   self.nc,
                                   self.decoder_kernel_size_3,
                                   stride=self.decoder_stride_3),  # B, 1, 32, 32
                # PrintLayer(),
                nn.Sigmoid()
            ]
        # --------------------------------------- end decoder -----------------------------------

        self.encoder_struct = nn.Sequential(*self.encoder)
        self.decoder_struct = nn.Sequential(*self.decoder)

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

        z_struct = self.encoder_struct(x)
        x_recons = self.decoder_struct(z_struct)

        return x_recons, z_struct
