from abc import ABC
import torch.nn as nn
from custom_Layer import Flatten, View, PrintLayer, kaiming_init


class DefaultCNN(nn.Module, ABC):
    """Model proposed in Kaggle for mnist recognition."""

    def __init__(self,
                 add_z_struct_bottleneck=False,
                 add_classification_layer=False,
                 z_struct_size=5,
                 classif_layer_size=30):
        """
        Class which defines model and forward pass.
        """
        super(DefaultCNN, self).__init__()

        # parameters
        self.nc = 1  # number of channels
        self.hidden_filters_1 = 32
        self.hidden_filters_2 = 64
        self.kernel_size_1 = 3
        self.kernel_size_2 = 5
        self.stride_size_1 = 1
        self.stride_size_2 = 2
        self.flatten_dim = 64
        self.linear_dim = 32
        self.n_classes = 10
        self.last_linear_layer_size = self.linear_dim

        # custom parameters:
        self.z_struct_size = z_struct_size
        self.add_z_struct_bottleneck = add_z_struct_bottleneck
        self.add_classification_layer = add_classification_layer
        self.classif_layer_size = classif_layer_size

        # ----------- define model: ------------
        self.model = [
            # --------------- first conv bloc:
            nn.Conv2d(self.nc, self.hidden_filters_1, self.kernel_size_1, stride=self.stride_size_1),
            nn.BatchNorm2d(self.hidden_filters_1),
            nn.ReLU(True),
            # PrintLayer(),  # B, 32, 30, 30
            nn.Conv2d(self.hidden_filters_1, self.hidden_filters_1, self.kernel_size_1, stride=self.stride_size_1),
            nn.BatchNorm2d(self.hidden_filters_1),
            nn.ReLU(True),
            # PrintLayer(),  # B, 32, 28, 28
            nn.Conv2d(self.hidden_filters_1, self.hidden_filters_1, self.kernel_size_2, stride=self.stride_size_2),
            nn.BatchNorm2d(self.hidden_filters_1),
            nn.ReLU(True),
            # PrintLayer(),  # B, 32, 12, 12
            nn.Dropout(0.4),
            # --------------- second conv bloc:
            nn.Conv2d(self.hidden_filters_1, self.hidden_filters_2, self.kernel_size_1, stride=self.stride_size_1),
            nn.BatchNorm2d(self.hidden_filters_2),
            nn.ReLU(True),
            # PrintLayer(),  # B, 64, 10, 10
            nn.Conv2d(self.hidden_filters_2, self.hidden_filters_2, self.kernel_size_2, stride=self.stride_size_1),
            nn.BatchNorm2d(self.hidden_filters_2),
            nn.ReLU(True),
            # PrintLayer(),  # B, 64, 6, 6
            nn.Conv2d(self.hidden_filters_2, self.hidden_filters_2, self.kernel_size_2, stride=self.stride_size_2),
            nn.BatchNorm2d(self.hidden_filters_2),
            nn.ReLU(True),
            # PrintLayer(),  # B, 64, 1, 1
            nn.Dropout(0.4),
            # --------------- Classifier bloc:
            Flatten(),
            # PrintLayer(),  # B, 64
            nn.Linear(self.flatten_dim, self.linear_dim),  # B, 32
            nn.BatchNorm1d(self.linear_dim),
            nn.ReLU(True),
            nn.Dropout(0.4),
        ]

        if self.add_z_struct_bottleneck:
            self.last_linear_layer_size = self.z_struct_size
            self.model += [
                # --------- add z_struct bottleneck :
                nn.Linear(self.linear_dim, self.z_struct_size),  # B, z_struct_size
                nn.BatchNorm1d(self.z_struct_size),
                nn.ReLU(True),
                nn.Dropout(0.4),
            ]

        if self.add_classification_layer:
            self.last_linear_layer_size = self.classif_layer_size
            self.model += [
                # --------- add classification layer :
                nn.Linear(self.z_struct_size, self.classif_layer_size),  # B, classif_layer_size
                nn.BatchNorm1d(self.classif_layer_size),
                nn.ReLU(True),
                nn.Dropout(0.4),
            ]

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
