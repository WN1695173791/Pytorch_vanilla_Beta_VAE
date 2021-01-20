from abc import ABC
import torch.nn as nn
from custom_Layer import Flatten, View, PrintLayer, kaiming_init


class Custom_CNN_BK(nn.Module, ABC):
    """Model proposed in Kaggle for mnist recognition."""

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
                 BK_in_third_layer=False):
        """
        Class which defines model and forward pass.
        """
        super(Custom_CNN_BK, self).__init__()

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

        if self.BK_in_first_layer:
            self.kernel_size_1 = self.big_kernel_size
        elif self.BK_in_second_layer:
            self.kernel_size_2 = self.big_kernel_size
        elif self.BK_in_third_layer:
            self.kernel_size_3 = self.big_kernel_size

        # ----------- define model: ------------
        # ----------- add conv bloc:
        self.model = [
            nn.Conv2d(self.nc, self.hidden_filters_1, self.kernel_size_1, stride=self.stride_size),
            nn.BatchNorm2d(self.hidden_filters_1),
            nn.ReLU(True),
            # PrintLayer(),  # B, 32, 25, 25
        ]
        if self.two_conv_layer:
            self.hidden_filter_GMP = self.hidden_filters_2
            self.model += [
                nn.Conv2d(self.hidden_filters_1, self.hidden_filters_2, self.kernel_size_2, stride=self.stride_size),
                nn.BatchNorm2d(self.hidden_filters_2),
                nn.ReLU(True),
                # PrintLayer(),  # B, 32, 25, 25
            ]
        if self.three_conv_layer:
            self.hidden_filter_GMP = self.hidden_filters_3
            self.model += [
                nn.Conv2d(self.hidden_filters_2, self.hidden_filters_3, self.kernel_size_3, stride=self.stride_size),
                nn.BatchNorm2d(self.hidden_filters_3),
                nn.ReLU(True),
                # PrintLayer(),  # B, 32, 25, 25
            ]
        # ----------- add GMP bloc:
        self.model += [
            nn.AdaptiveMaxPool2d((1, 1)),  # B, hidden_filters_1, 1, 1
            # PrintLayer(),  # B, hidden_filters_1, 1, 1
            View((-1, self.hidden_filter_GMP)),  # B, hidden_filters_1
            # PrintLayer(),  # B, hidden_filters_1
            nn.Linear(self.hidden_filter_GMP, self.z_struct_size),  # shape: (Batch, z_struct_size)
            nn.BatchNorm1d(self.z_struct_size),
            nn.ReLU(True),
            nn.Dropout(0.4),
            # PrintLayer(),  # B, z_struct_size
        ]
        if self.add_classification_layer:
            self.last_linear_layer_size = self.classif_layer_size
            # ------------ add classification bloc:
            self.model += [
                nn.Linear(self.z_struct_size, self.classif_layer_size),  # B, classif_layer_size
                nn.BatchNorm1d(self.classif_layer_size),
                nn.ReLU(True),
                nn.Dropout(0.4),
                # PrintLayer(),  # B,
            ]
        # ---------- final layer for classification:
        self.model += [
            nn.Linear(self.last_linear_layer_size, self.n_classes),  # B, nb_class
            nn.Softmax()
        ]

        self.net = nn.Sequential(*self.model)
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        """
        Forward pass of model.
        """
        prediction = self.net(x)

        return prediction
