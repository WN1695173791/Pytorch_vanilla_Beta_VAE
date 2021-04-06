from abc import ABC
import torch.nn as nn
from custom_Layer import View, PrintLayer, kaiming_init
from binary_tools.activations import DeterministicBinaryActivation
import torch
import torch.nn.functional as F

EPS = 1e-12


class Encoder_struct(nn.Module, ABC):

    def __init__(self,
                 z_struct_size=5,
                 big_kernel_size=8,
                 stride_size=1,
                 kernel_size_1=4,
                 kernel_size_2=4,
                 kernel_size_3=4,
                 hidden_filters_1=32,
                 hidden_filters_2=32,
                 hidden_filters_3=32,
                 BK_in_first_layer=True,
                 BK_in_second_layer=False,
                 BK_in_third_layer=False,
                 two_conv_layer=True,
                 three_conv_layer=False,
                 Binary_z=False,
                 binary_first_conv=False,
                 binary_second_conv=False,
                 binary_third_conv=False):
        """
        Class which defines model and forward pass.
        """
        super(Encoder_struct, self).__init__()

        self.Binary_z = Binary_z
        # parameters
        self.nc = 1  # number of channels
        self.hidden_filters_1 = hidden_filters_1
        self.hidden_filters_2 = hidden_filters_2
        self.hidden_filters_3 = hidden_filters_3
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2
        self.kernel_size_3 = kernel_size_3
        self.n_classes = 10

        # custom parameters:
        self.z_struct_size = z_struct_size
        self.big_kernel_size = big_kernel_size
        self.stride_size = stride_size
        self.BK_in_first_layer = BK_in_first_layer
        self.BK_in_second_layer = BK_in_second_layer
        self.BK_in_third_layer = BK_in_third_layer
        self.two_conv_layer = two_conv_layer
        self.three_conv_layer = three_conv_layer

        # binary:
        self.binary_first_conv = binary_first_conv
        self.binary_second_conv = binary_second_conv
        self.binary_third_conv = binary_third_conv

        if self.BK_in_first_layer:
            self.kernel_size_1 = self.big_kernel_size
        elif self.BK_in_second_layer:
            self.kernel_size_2 = self.big_kernel_size
        elif self.BK_in_third_layer:
            self.kernel_size_3 = self.big_kernel_size

        if self.two_conv_layer and not self.three_conv_layer:
            self.hidden_filters_2 = self.z_struct_size
        elif self.three_conv_layer:
            self.hidden_filters_3 = self.z_struct_size
        else:
            self.hidden_filters_1 = self.z_struct_size

        # _________________________---------------- define encoder_struct: ------------_____________________________
        self.encoder_struct = [
            nn.Conv2d(self.nc, self.hidden_filters_1, self.kernel_size_1, stride=self.stride_size),
            nn.BatchNorm2d(self.hidden_filters_1),
            # PrintLayer(),  # B, 32, 25, 25
        ]
        if self.Binary_z and self.binary_first_conv:
            self.encoder_struct += [
                DeterministicBinaryActivation(estimator='ST')
            ]
        else:
            self.encoder_struct += [
                nn.ReLU(True),
            ]
        if self.two_conv_layer:
            self.encoder_struct += [
                nn.Conv2d(self.hidden_filters_1, self.hidden_filters_2, self.kernel_size_2, stride=self.stride_size),
                nn.BatchNorm2d(self.hidden_filters_2),
                # PrintLayer(),  # B, 32, 25, 25
            ]
        if self.Binary_z and self.two_conv_layer and self.binary_second_conv:
            self.encoder_struct += [
                DeterministicBinaryActivation(estimator='ST')
            ]
        elif self.two_conv_layer:
            self.encoder_struct += [
                nn.ReLU(True),
            ]
        if self.three_conv_layer:
            self.encoder_struct += [
                nn.Conv2d(self.hidden_filters_2, self.hidden_filters_3, self.kernel_size_3, stride=self.stride_size),
                nn.BatchNorm2d(self.hidden_filters_3),
                # PrintLayer(),  # B, 32, 25, 25
            ]
        if self.Binary_z and self.three_conv_layer and self.binary_third_conv:
            self.encoder_struct += [
                DeterministicBinaryActivation(estimator='ST')
            ]
        elif self.three_conv_layer:
            self.encoder_struct += [
                nn.ReLU(True),
            ]

        # ---------- GMP and final layer for classification:
        self.encoder_struct += [
            nn.AdaptiveMaxPool2d((1, 1)),  # B, z_struct_size
            View((-1, self.z_struct_size)),  # B, z_struct_size
            nn.Linear(self.z_struct_size, self.n_classes)  # B, nb_class
            # nn.Softmax()
        ]
        # _________________________---------------- end encoder_struct: ------------_____________________________

        self.encoder_struct = nn.Sequential(*self.encoder_struct)
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                # weight_init(m)
                kaiming_init(m)

    def forward(self, x, labels=None, nb_class=None, use_ratio=False, z_struct_out=False, z_struct_prediction=False,
                z_struct_layer_num=None, loss_min_distance_cl=False, Hmg_dst_loss=False, uniq_bin_code_target=False,
                target_code=None):
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
        global_avg_Hmg_dst = 0
        avg_dst_classes = 0
        uniq_target_dist_loss = 0

        if z_struct_out:
            z_struct = self.encoder_struct[:z_struct_layer_num](x)

        if z_struct_prediction:
            if self.Binary_z:
                out = self.encoder_struct[z_struct_layer_num:](x)
                prediction = F.log_softmax(out, dim=1)
            else:
                prediction = self.encoder_struct[z_struct_layer_num:](x)
        else:
            out = self.encoder_struct(x)
            if self.Binary_z:
                prediction = F.log_softmax(out, dim=1)
            else:
                prediction = F.softmax(out, dim=1)

        if use_ratio:
            ratio, variance_intra, variance_inter = self.compute_ratio_batch(z_struct,
                                                                             labels,
                                                                             nb_class)

        if loss_min_distance_cl:
            variance_distance_iter_class, mean_distance_intra_class = self.compute_var_distance_class(z_struct,
                                                                                                      labels,
                                                                                                      nb_class)

        if Hmg_dst_loss:
            global_avg_Hmg_dst, avg_dst_classes = self.hamming_distance_loss(z_struct, nb_class, labels)

        if uniq_bin_code_target:
            uniq_target_dist_loss = self.distance_target_uniq_code(z_struct, nb_class, labels, target_code)

        return prediction, z_struct, ratio, variance_distance_iter_class, variance_intra, mean_distance_intra_class, \
               variance_inter, global_avg_Hmg_dst, avg_dst_classes, uniq_target_dist_loss

    def compute_ratio_batch(self, batch_z_struct, labels_batch, nb_class):
        """
        compute ratio of one batch:
        ratio: to minimize, variance_inter_class / var_intra_class
        :return:
        """

        first = True
        for class_id in range(nb_class):
            z_struct_class = batch_z_struct[torch.where(labels_batch == class_id)]
            mean_class_iter = torch.mean(z_struct_class, axis=0).squeeze(axis=-1).squeeze(axis=-1).unsqueeze(axis=0)
            std_class_iter = torch.std(z_struct_class, axis=0).squeeze(axis=-1).squeeze(axis=-1).unsqueeze(axis=0)
            if first:
                mean_class = mean_class_iter
                std_class = std_class_iter
                first = False
            else:
                mean_class = torch.cat((mean_class, mean_class_iter), dim=0)
                std_class = torch.cat((std_class, std_class_iter), dim=0)

        # variance_intra_class = torch.square(std_class)  # shape: (nb_class, len(z_struct))
        variance_intra_class = std_class * std_class
        variance_intra_class_mean_components = torch.mean(variance_intra_class, axis=0)
        # variance_inter_class = torch.square(torch.std(mean_class, axis=0))
        variance_inter_class = torch.std(mean_class, axis=0) * torch.std(mean_class, axis=0)

        ratio = variance_intra_class_mean_components / (variance_inter_class + EPS)
        ratio_variance_mean = torch.mean(ratio)

        return ratio_variance_mean, torch.mean(variance_intra_class_mean_components), torch.mean(variance_inter_class)

    def compute_var_distance_class(self, batch_z_struct, labels_batch, nb_class):
        """
        compute ratio of one batch:
        ratio: to minimize, variance_inter_class / var_intra_class
        :return:
        """
        first = True
        for class_id in range(nb_class):
            z_struct_class = batch_z_struct[torch.where(labels_batch == class_id)]
            mean_class_iter = torch.mean(z_struct_class, axis=0).squeeze(axis=-1).squeeze(axis=-1).unsqueeze(axis=0)
            if first:
                mean_class = mean_class_iter
                first = False
            else:
                mean_class = torch.cat((mean_class, mean_class_iter), dim=0)

        first = True
        # add distance between all classes:
        for i in range(nb_class):
            for j in range(nb_class):
                if i == j:
                    # don't compute distance between same class
                    continue
                dist = torch.norm(mean_class[i] - mean_class[j]).unsqueeze(dim=0)
                if first:
                    distance_inter_class = dist
                    first = False
                else:
                    distance_inter_class = torch.cat((distance_inter_class, dist), dim=0)

        std_class_distance = torch.std(distance_inter_class, axis=0)
        variance_intra_class_distance = std_class_distance * std_class_distance
        variance_intra_class_distance_mean = torch.mean(variance_intra_class_distance, axis=0)

        return variance_intra_class_distance_mean, torch.mean(distance_inter_class)

    def hamming_distance_loss(self, batch_z_struct, nb_class, labels_batch):
        """
        computing hamming distance between z_struct in the same class
        :param z_struct:
        :return:
        """
        assert batch_z_struct is not None, "z_struct mustn't be None to compute Hamming distance"
        assert labels_batch is not None, "labels_batch mustn't be None to compute Hamming distance"

        global_first = True
        for class_id in range(nb_class):
            first = True
            z_struct_class_iter = batch_z_struct[torch.where(labels_batch == class_id)]
            # computing distance:
            for i in range(len(z_struct_class_iter)):
                for j in range(len(z_struct_class_iter)):
                    if i == j:
                        pass
                    else:
                        # Hmg_dist = torch.mean(torch.Tensor.float(z_struct_class_iter[i] != z_struct_class_iter[j]))
                        Hmg_dist = torch.norm(z_struct_class_iter[i] - z_struct_class_iter[j])
                        Hmg_dist = torch.unsqueeze(Hmg_dist, 0)
                        if first:
                            avg_dst_class_id = Hmg_dist
                            first = False
                        else:
                            avg_dst_class_id = torch.cat((avg_dst_class_id, Hmg_dist), dim=0)
            if global_first:
                avg_dst_classes = torch.mean(avg_dst_class_id)
            else:
                avg_dst_classes = torch.cat((avg_dst_classes, torch.mean(avg_dst_class_id)), dim=0)

        global_avg_Hmg_dst = torch.mean(avg_dst_classes)

        return global_avg_Hmg_dst, avg_dst_classes

    def distance_target_uniq_code(self, batch_z_struct, nb_class, labels_batch, target_code):
        """
        computing distance between target class code and z_struct
        :param z_struct:
        :return:
        """

        global_first = True
        for class_id in range(nb_class):
            first = True
            z_struct_class_iter = batch_z_struct[torch.where(labels_batch == class_id)]
            # computing distance:
            for i in range(len(z_struct_class_iter)):
                for j in range(len(z_struct_class_iter)):
                    if i == j:
                        pass
                    else:
                        # Hmg_dist = torch.mean(torch.Tensor.float(z_struct_class_iter[i] != z_struct_class_iter[j]))
                        dst_target = torch.norm(z_struct_class_iter[i] - target_code[class_id])
                        dst_target = torch.unsqueeze(dst_target, 0)
                        if first:
                            avg_dst_class_id = dst_target
                            first = False
                        else:
                            avg_dst_class_id = torch.cat((avg_dst_class_id, dst_target), dim=0)
            if global_first:
                avg_dst_classes = torch.mean(avg_dst_class_id)
            else:
                avg_dst_classes = torch.cat((avg_dst_classes, torch.mean(avg_dst_class_id)), dim=0)

        global_avg_dst = torch.mean(avg_dst_classes)

        return global_avg_dst
