from abc import ABC
import torch.nn as nn
from custom_Layer import Flatten, View, PrintLayer, kaiming_init
from binary_tools.activations import DeterministicBinaryActivation
import numpy as np
from models.weight_init import weight_init
import torch
import torch.nn.functional as F

EPS = 1e-12


def compute_var_distance_class_test(batch_z_struct, labels_batch, nb_class):
    """
    compute ratio of one batch:
    ratio: to minimize, variance_inter_class / var_intra_class

    :return:
    """

    batch_z_struct = torch.tensor(batch_z_struct)
    labels_batch = torch.tensor(labels_batch)

    first = True
    for class_id in range(nb_class):
        z_struct_class = batch_z_struct[torch.where(labels_batch == class_id)]
        mean_class_iter = torch.mean(z_struct_class, axis=0).squeeze(axis=-1).squeeze(axis=-1).unsqueeze(axis=0)
        if first:
            mean_class = mean_class_iter
        else:
            mean_class = torch.cat((mean_class, mean_class_iter), dim=0)
        first = False

    first = True
    # add distance between all classes:
    for i in range(nb_class):
        for j in range(nb_class):
            if i == j:
                continue
            dist = torch.norm(mean_class[i] - mean_class[j]).unsqueeze(dim=0)
            if first:
                distance_inter_class = dist
            else:
                distance_inter_class = torch.cat((distance_inter_class, dist), dim=0)
            first = False

    std_class_distance = torch.std(distance_inter_class, axis=0)
    variance_intra_class_distance = std_class_distance * std_class_distance
    variance_intra_class_distance_mean = torch.mean(variance_intra_class_distance, axis=0)

    return variance_intra_class_distance_mean, torch.mean(distance_inter_class)


def compute_ratio_batch_test(batch_z_struct, labels_batch, nb_class, other_ratio=False):
    """
    compute ratio of one batch:
    ratio: to minimize, variance_inter_class / var_intra_class

    :return:
    """
    if 'torch.Tensor' in str(type(batch_z_struct)):
        batch_z_struct = batch_z_struct.cpu().detach()
        labels_batch = labels_batch.cpu().detach()

    representation_z_struct_class = []
    for class_id in range(nb_class):
        z_struct_class = batch_z_struct[np.where(labels_batch == class_id)]
        representation_z_struct_class.append(z_struct_class)

    representation_z_struct_class = np.array(representation_z_struct_class)

    z_struct_mean_global_per_class = []
    z_struct_std_global_per_class = []
    for class_id in range(nb_class):
        z_struct_mean_global_per_class.append(np.mean(representation_z_struct_class[class_id], axis=0))
        z_struct_std_global_per_class.append(np.std(representation_z_struct_class[class_id], axis=0))

    z_struct_mean_global_per_class = np.array(z_struct_mean_global_per_class)
    z_struct_std_global_per_class = np.array(z_struct_std_global_per_class)

    variance_intra_class = np.square(z_struct_std_global_per_class)  # shape: (nb_class, len(z_struct))
    variance_intra_class_mean_components = np.mean(variance_intra_class, axis=0)
    variance_inter_class = np.square(np.std(z_struct_mean_global_per_class, axis=0))

    if other_ratio:
        ratio = variance_inter_class / (variance_intra_class_mean_components + EPS)  # shape: (len(z_struct))
    else:
        ratio = variance_intra_class_mean_components / (variance_inter_class + EPS)
    ratio_variance_mean = np.mean(ratio)

    return ratio_variance_mean


class Custom_CNN_BK(nn.Module, ABC):

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
                 binary_chain=False,
                 add_linear_after_GMP=True):
        """
        Class which defines model and forward pass.
        """
        super(Custom_CNN_BK, self).__init__()

        self.add_linear_after_GMP = add_linear_after_GMP
        self.Binary_z = Binary_z
        self.binary_chain = binary_chain
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

        if self.two_conv_layer:
            self.hidden_filters_2 = self.z_struct_size
        elif self.three_conv_layer:
            self.hidden_filters_3 = self.z_struct_size

        # ----------- define model: ------------
        # ----------- add conv bloc:
        self.model = [
            nn.Conv2d(self.nc, self.hidden_filters_1, self.kernel_size_1, stride=self.stride_size),
            nn.BatchNorm2d(self.hidden_filters_1),
            # PrintLayer(),  # B, 32, 25, 25
        ]
        if self.Binary_z and not self.two_conv_layer:
            self.model += [
                # DeterministicBinaryActivation(estimator='ST')
                # nn.ReLU(True),
                # nn.Hardtanh(inplace=True),
                nn.Sigmoid(),
            ]
        else:
            self.model += [
                nn.ReLU(True),
            ]
        if self.two_conv_layer:
            self.hidden_filter_GMP = self.hidden_filters_2
            numChannels = self.hidden_filters_2
            self.model += [
                nn.Conv2d(self.hidden_filters_1, self.hidden_filters_2, self.kernel_size_2, stride=self.stride_size),
                nn.BatchNorm2d(self.hidden_filters_2),
                # PrintLayer(),  # B, 32, 25, 25
            ]
        if self.Binary_z and self.two_conv_layer and not self.three_conv_layer:
            self.model += [
                # DeterministicBinaryActivation(estimator='ST')
                # nn.ReLU(True),
                nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=True),
                # nn.Hardsigmoid(True),
                # nn.Hardtanh(inplace=True),
                # nn.Sigmoid(),
            ]
        elif self.two_conv_layer:
            self.model += [
                nn.ReLU(True),
            ]
        if self.three_conv_layer:
            self.hidden_filter_GMP = self.hidden_filters_3
            numChannels = self.hidden_filters_3
            self.model += [
                nn.Conv2d(self.hidden_filters_2, self.hidden_filters_3, self.kernel_size_3, stride=self.stride_size),
                nn.BatchNorm2d(self.hidden_filters_3),
                # PrintLayer(),  # B, 32, 25, 25
            ]
        if self.Binary_z and self.three_conv_layer:
            self.model += [
                DeterministicBinaryActivation(estimator='ST')
                # nn.ReLU(True),
                # nn.Hardtanh(min_val=0., max_val=1.0, inplace=True),
                # nn.Hardsigmoid(True),
                # nn.Hardtanh(inplace=True),
                # nn.Sigmoid(),
            ]
        elif self.three_conv_layer:
            self.model += [
                nn.ReLU(True),
            ]

        if self.Binary_z:
            numWeights = 16
            depth = 1
            sparsity = 0.5
            self.model_2 = []
            if self.binary_chain:
                self.model_2 += [BlockLBP(numChannels, numWeights, sparsity) for i in range(depth)]
            # else:
            #     self.chain = [
            #         DeterministicBinaryActivation(estimator='ST')
            #     ]

            # ----------- add GMP bloc:
            self.model_2 += [
                nn.AdaptiveMaxPool2d((1, 1)),  # B, hidden_filters_1, 1, 1
                # PrintLayer(),  # B, hidden_filters_1, 1, 1
                View((-1, self.hidden_filter_GMP)),  # B, hidden_filters_1
                # PrintLayer(),  # B, hidden_filters_1
            ]
            if self.add_linear_after_GMP:
                self.model_2 += [
                    nn.Linear(self.hidden_filter_GMP, self.z_struct_size),  # shape: (Batch, z_struct_size)
                    nn.BatchNorm1d(self.z_struct_size),
                    nn.ReLU(True),
                    nn.Dropout(0.4),
                    # PrintLayer(),  # B, z_struct_size
                ]
            else:
                self.last_linear_layer_size = self.hidden_filter_GMP

            if self.add_classification_layer:
                self.last_linear_layer_size = self.classif_layer_size
                # ------------ add classification bloc:
                self.model_2 += [
                    nn.Linear(self.z_struct_size, self.classif_layer_size),  # B, classify_layer_size
                    nn.BatchNorm1d(self.classif_layer_size),
                    nn.ReLU(True),
                    nn.Dropout(0.4),
                    # PrintLayer(),  # B,
                ]
            # ---------- final layer for classification:
            self.model_2 += [
                nn.Linear(self.last_linear_layer_size, self.n_classes)  # B, nb_class
            ]
            self.net = nn.Sequential(*self.model,
                                     *self.model_2)
        else:
            # ----------- add GMP bloc:
            self.model += [
                nn.AdaptiveMaxPool2d((1, 1)),  # B, hidden_filters_1, 1, 1
                # PrintLayer(),  # B, hidden_filters_1, 1, 1
                View((-1, self.hidden_filter_GMP)),  # B, hidden_filters_1
                # PrintLayer(),  # B, hidden_filters_1
            ]
            if self.add_linear_after_GMP:
                self.model += [
                    nn.Linear(self.hidden_filter_GMP, self.z_struct_size),  # shape: (Batch, z_struct_size)
                    nn.BatchNorm1d(self.z_struct_size),
                    nn.ReLU(True),
                    nn.Dropout(0.4),
                    # PrintLayer(),  # B, z_struct_size
                ]
            else:
                self.last_linear_layer_size = self.hidden_filter_GMP

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
                # weight_init(m)
                kaiming_init(m)

    def forward(self, x, labels=None, nb_class=None, use_ratio=False, z_struct_out=False, z_struct_prediction=False,
                z_struct_layer_num=None, other_ratio=False, loss_min_distance_cl=False):
        """
        Forward pass of model.
        """
        if use_ratio:
            assert z_struct_out is True, "to compute ratio we need to extract z_struct"
        if z_struct_out or z_struct_prediction:
            assert z_struct_layer_num is not None, "if we use z_struct we need the extraction layer num"

        # initialization:
        z_struct = None
        ratio = 0
        variance_distance_iter_class = 0
        variance_intra = 0
        mean_distance_intra_class = 0
        variance_inter = 0

        # parameters:
        seuil_min_image = 10  # nombre minimal d'images pour calculer le ratio (ou variance)

        if z_struct_out:
            z_struct = self.net[:z_struct_layer_num](x)

        if z_struct_prediction:
            if self.Binary_z:
                out = self.net[z_struct_layer_num:](x)
                prediction = F.log_softmax(out, dim=1)
            else:
                prediction = self.net[z_struct_layer_num:](x)
        else:
            if self.Binary_z:
                out = self.net(x)
                prediction = F.log_softmax(out, dim=1)
            else:
                prediction = self.net(x)

        if use_ratio:
            ratio, variance_intra, variance_inter = self.compute_ratio_batch(z_struct,
                                                                             labels,
                                                                             nb_class,
                                                                             other_ratio=other_ratio,
                                                                             seuil_min_image=seuil_min_image)

        if loss_min_distance_cl:
            variance_distance_iter_class, mean_distance_intra_class = self.compute_var_distance_class(z_struct,
                                                                                                      labels,
                                                                                                      nb_class,
                                                                                                      seuil_min_image=seuil_min_image)

        return prediction, z_struct, ratio, variance_distance_iter_class, variance_intra, mean_distance_intra_class, \
               variance_inter

    def compute_ratio_batch(self, batch_z_struct, labels_batch, nb_class, other_ratio=False, seuil_min_image=2):
        """
        compute ratio of one batch:
        ratio: to minimize, variance_inter_class / var_intra_class

        :return:
        """

        first = True
        for class_id in range(nb_class):
            z_struct_class = batch_z_struct[torch.where(labels_batch == class_id)]
            if len(z_struct_class) < seuil_min_image:
                print("Ratio: In this batch for class {}, they are less images than {}. Don't use it to compute "
                      "ratio!!!!".format(class_id, seuil_min_image))
                continue
            else:
                mean_class_iter = torch.mean(z_struct_class, axis=0).squeeze(axis=-1).squeeze(axis=-1).unsqueeze(axis=0)
                std_class_iter = torch.std(z_struct_class, axis=0).squeeze(axis=-1).squeeze(axis=-1).unsqueeze(axis=0)
                if first:
                    mean_class = mean_class_iter
                    std_class = std_class_iter
                else:
                    mean_class = torch.cat((mean_class, mean_class_iter), dim=0)
                    std_class = torch.cat((std_class, std_class_iter), dim=0)
                first = False

        # variance_intra_class = torch.square(std_class)  # shape: (nb_class, len(z_struct))
        variance_intra_class = std_class * std_class
        variance_intra_class_mean_components = torch.mean(variance_intra_class, axis=0)
        # variance_inter_class = torch.square(torch.std(mean_class, axis=0))
        variance_inter_class = torch.std(mean_class, axis=0) * torch.std(mean_class, axis=0)

        if other_ratio:
            ratio = variance_inter_class / (variance_intra_class_mean_components + EPS)  # shape: (len(z_struct))
        else:
            ratio = variance_intra_class_mean_components / (variance_inter_class + EPS)

        ratio_variance_mean = torch.mean(ratio)

        return ratio_variance_mean, torch.mean(variance_intra_class_mean_components), torch.mean(variance_inter_class)

    def compute_var_distance_class(self, batch_z_struct, labels_batch, nb_class, seuil_min_image=2):
        """
        compute ratio of one batch:
        ratio: to minimize, variance_inter_class / var_intra_class

        :return:
        """

        first = True
        class_tensor = torch.arange(nb_class)
        new_class_tensor = torch.arange(nb_class)
        for class_id in class_tensor:
            z_struct_class = batch_z_struct[torch.where(labels_batch == class_id)]
            if len(z_struct_class) < seuil_min_image:
                print("Variance: In this batch for class {}, they are less images than {}. Don't use it to compute "
                      "ratio!!!!".format(class_id, seuil_min_image))
                new_class_tensor = new_class_tensor[new_class_tensor != class_id]
                continue
            else:
                mean_class_iter = torch.mean(z_struct_class, axis=0).squeeze(axis=-1).squeeze(axis=-1).unsqueeze(axis=0)
                if first:
                    mean_class = mean_class_iter
                else:
                    mean_class = torch.cat((mean_class, mean_class_iter), dim=0)
                first = False

        first = True
        # add distance between all classes:
        for i in range(len(new_class_tensor)):
            for j in range(len(new_class_tensor)):
                if i == j:
                    continue
                dist = torch.norm(mean_class[i] - mean_class[j]).unsqueeze(dim=0)
                if first:
                    distance_inter_class = dist
                else:
                    distance_inter_class = torch.cat((distance_inter_class, dist), dim=0)
                first = False

        std_class_distance = torch.std(distance_inter_class, axis=0)
        variance_intra_class_distance = std_class_distance * std_class_distance
        variance_intra_class_distance_mean = torch.mean(variance_intra_class_distance, axis=0)

        return variance_intra_class_distance_mean, torch.mean(distance_inter_class)


class BlockLBP(nn.Module):

    def __init__(self, numChannels, numWeights, sparsity=0.5):
        super().__init__()
        self.batch_norm = nn.BatchNorm2d(numChannels)
        self.conv_lbp = ConvLBP(numChannels, numWeights, kernel_size=3, sparsity=sparsity)
        self.conv_1x1 = nn.Conv2d(numWeights, numChannels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.batch_norm(x)
        x = F.relu(self.conv_lbp(x))
        x = self.conv_1x1(x)
        x.add_(residual)
        return x


class ConvLBP(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, sparsity=0.5):
        super().__init__(in_channels, out_channels, kernel_size, padding=1, bias=False)
        weights = next(self.parameters())
        matrix_proba = torch.FloatTensor(weights.data.shape).fill_(0.5)
        binary_weights = torch.bernoulli(matrix_proba) * 2 - 1
        mask_inactive = torch.rand(matrix_proba.shape) > sparsity
        binary_weights.masked_fill_(mask_inactive, 0)
        weights.data = binary_weights
        weights.requires_grad_(False)
