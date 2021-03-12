import logging
import os

import torch
import torch.nn.functional as F
import torch.optim as optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm

# import losses
from dataset import sampler
from dataset.dataset_2 import get_dataloaders, get_mnist_dataset
from models.custom_CNN import Custom_CNN
from models.custom_CNN_BK import Custom_CNN_BK
from models.custom_CNN_BK import compute_ratio_batch_test, compute_var_distance_class_test
from models.default_CNN import DefaultCNN
from pytorchtools import EarlyStopping
from scores_classifier import compute_scores
from solver import gpu_config
from visualizer_CNN import get_layer_zstruct_num
import numpy as np
from dataset.sampler import BalancedBatchSampler
import random
from models.Encoder_decoder import Encoder_decoder
import torch.nn as nn

EPS = 1e-12
worker_id = 0


def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    To use for DataLoader reproducibility.
    :param worker_id:
    :return:
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_z_struct_representation(loader, net, z_struct_layer_num):
    z_struct_representation = []
    labels_list = []
    for data, labels in loader:

        with torch.no_grad():
            input_data = data
        if torch.cuda.is_available():
            input_data = input_data.cuda()

        _, z_struct, _, _, _, _, _ = net(input_data,
                                      z_struct_out=True,
                                      z_struct_layer_num=z_struct_layer_num)

        z_struct_batch = z_struct.squeeze().cpu().detach().numpy()
        z_struct_representation.extend(z_struct_batch)
        labels_list.extend(labels.cpu().detach().numpy())

    return np.array(z_struct_representation), np.array(labels_list)


def compute_scores_and_loss(net, train_loader, test_loader, device, train_loader_size, test_loader_size,
                            net_type, nb_class, ratio_reg, other_ratio, loss_min_distance_cl, z_struct_layer_num,
                            loss_distance_mean):
    score_train, loss_train = compute_scores(net, train_loader, device, train_loader_size)
    score_test, loss_test = compute_scores(net, test_loader, device, test_loader_size)

    if ratio_reg or loss_min_distance_cl or loss_distance_mean:
        # compute ratio on all test set:
        z_struct_representation_test, labels_batch_test = get_z_struct_representation(test_loader,
                                                                                      net,
                                                                                      z_struct_layer_num)
        # compute ratio on all train set:
        z_struct_representation_train, labels_batch_train = get_z_struct_representation(train_loader,
                                                                                        net,
                                                                                        z_struct_layer_num)

    if ratio_reg:
        ratio_test = compute_ratio_batch_test(z_struct_representation_test,
                                              labels_batch_test,
                                              nb_class,
                                              other_ratio=other_ratio)
        ratio_train = compute_ratio_batch_test(z_struct_representation_train,
                                               labels_batch_train,
                                               nb_class,
                                               other_ratio=other_ratio)
    else:
        ratio_test = 0
        ratio_train = 0

    if loss_min_distance_cl:
        var_distance_classes_train, _ = compute_var_distance_class_test(z_struct_representation_train,
                                                                        labels_batch_train,
                                                                        nb_class)
        var_distance_classes_test, _ = compute_var_distance_class_test(z_struct_representation_test,
                                                                       labels_batch_test,
                                                                       nb_class)
    else:
        var_distance_classes_train = 0
        var_distance_classes_test = 0

    if loss_distance_mean:
        _, mean_distance_intra_class_train = compute_var_distance_class_test(z_struct_representation_train,
                                                                             labels_batch_train,
                                                                             nb_class)
        _, mean_distance_intra_class_test = compute_var_distance_class_test(z_struct_representation_test,
                                                                            labels_batch_test,
                                                                            nb_class)
    else:
        mean_distance_intra_class_train = 0
        mean_distance_intra_class_test = 0

    scores = {'train': score_train, 'test': score_test}
    losses = {'train_class': loss_train,
              'test_class': loss_test,
              'ratio_test_loss': ratio_test,
              'ratio_train_loss': ratio_train,
              'var_distance_classes_train': var_distance_classes_train,
              'var_distance_classes_test': var_distance_classes_test,
              'mean_distance_intra_class_train': mean_distance_intra_class_train,
              'mean_distance_intra_class_test': mean_distance_intra_class_test,
              'total_loss_train': loss_train + ratio_train,
              'total_loss_test': loss_test + ratio_test}

    return scores, losses


class SolverClassifier(object):
    def __init__(self, args):

        # parameters:
        self.is_default_model = args.is_default_model
        self.is_custom_model = args.is_custom_model
        self.is_custom_model_BK = args.is_custom_model_BK
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.max_iter = args.max_iter
        self.use_early_stopping = args.use_early_stopping

        # Custom CNN parameters:
        self.lambda_classification = args.lambda_class
        self.add_z_struct_bottleneck = args.add_z_struct_bottleneck
        self.add_classification_layer = args.add_classification_layer
        self.z_struct_size = args.z_struct_size
        self.classif_layer_size = args.classif_layer_size
        if self.is_custom_model_BK:
            self.big_kernel_size = args.big_kernel_size[0]
        self.stride_size = args.stride_size
        self.hidden_filters_1 = args.hidden_filters_layer1
        self.hidden_filters_2 = args.hidden_filters_layer2
        self.hidden_filters_3 = args.hidden_filters_layer3
        self.BK_in_first_layer = args.BK_in_first_layer
        self.two_conv_layer = args.two_conv_layer
        self.three_conv_layer = args.three_conv_layer
        self.BK_in_second_layer = args.BK_in_second_layer
        self.BK_in_third_layer = args.BK_in_third_layer
        self.use_scheduler = args.use_scheduler
        self.add_linear_after_GMP = args.add_linear_after_GMP
        self.without_acc = args.without_acc
        # binary parameters:
        self.binary_z = args.binary_z
        # ratio regularization:
        self.ratio_reg = args.ratio_reg
        self.lambda_ratio_reg = args.lambda_ratio_reg
        self.other_ratio = args.other_ratio
        self.lambda_var_intra = args.lambda_var_intra
        self.lambda_var_inter = args.lambda_var_inter
        # Contrastive loss parameters:
        self.IPC = args.IPC
        self.num_workers = args.num_workers
        self.loss = args.loss
        self.sz_embedding = args.sz_embedding
        self.mrg = args.mrg
        self.dataset = args.dataset
        self.model = args.model
        self.loss = args.loss
        self.alpha = args.alpha
        self.mrg = args.mrg
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.remark = args.remark
        self.contrastive_loss = args.contrastive_loss
        self.lambda_contrastive_loss = args.lambda_contrastive_loss
        # add loss min var distance mean class:
        self.loss_min_distance_cl = args.loss_min_distance_cl
        self.lambda_var_distance = args.lambda_var_distance
        # intra class variance loss:
        self.intra_class_variance_loss = args.intra_class_variance_loss
        self.lambda_intra_class_var = args.lambda_intra_class_var
        # intra class max distance loss:
        self.lambda_distance_mean = args.lambda_distance_mean
        self.loss_distance_mean = args.loss_distance_mean
        self.dataset_balanced = args.dataset_balanced
        self.value_target_distance_mean = args.value_target_distance_mean
        # decoder:
        self.use_decoder = args.use_decoder
        self.test_all_learn = args.test_all_learn

        # logger
        formatter = logging.Formatter('%(asc_time)s %(level_name)s - %(funcName)s: %(message)s', "%H:%M:%S")
        logger = logging.getLogger(__name__)
        logger.setLevel(args.log_level.upper())
        stream = logging.StreamHandler()
        stream.setLevel(args.log_level.upper())
        stream.setFormatter(formatter)
        logger.addHandler(stream)

        # load dataset:
        # dataset parameters:
        if args.dataset.lower() == 'mnist':
            self.img_size = (1, 32, 32)
            self.nb_class = 10
            self.nc = 1
        else:
            raise NotImplementedError
        self.nb_pixels = self.img_size[1] * self.img_size[2]

        if args.dataset == 'mnist' and not self.dataset_balanced:
            self.valid_loader = 0
            self.train_loader, self.test_loader = get_mnist_dataset(batch_size=self.batch_size)
        else:
            self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(args.dataset,
                                                                                     batch_size=self.batch_size,
                                                                                     logger=logger)

        if self.dataset_balanced and args.dataset == 'mnist':
            self.train_loader_bf, self.test_loader_bf = get_mnist_dataset(batch_size=self.batch_size,
                                                                          return_Dataloader=False)

            self.train_loader = torch.utils.data.DataLoader(self.train_loader_bf,
                                                            sampler=BalancedBatchSampler(self.train_loader_bf),
                                                            batch_size=self.batch_size)

            _, self.test_loader = get_mnist_dataset(batch_size=self.batch_size)
            print('Balanced samples per class')

        self.train_loader_size = len(self.train_loader.dataset)
        if self.valid_loader == 0:
            self.valid_loader_size = 0
        else:
            self.valid_loader_size = len(self.valid_loader.dataset)
        self.test_loader_size = len(self.test_loader.dataset)

        if self.contrastive_loss:
            self.train_loader_bf, _ = get_mnist_dataset(batch_size=self.batch_size,
                                                        return_Dataloader=False)
            if self.IPC:
                balanced_sampler = sampler.BalancedSampler(self.train_loader_bf,
                                                           batch_size=self.batch_size,
                                                           images_per_class=3)
                batch_sampler = BatchSampler(balanced_sampler,
                                             batch_size=self.batch_size,
                                             drop_last=True)
                self.train_loader = torch.utils.data.DataLoader(
                    self.train_loader_bf,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    batch_sampler=batch_sampler
                )
                print('Balanced Sampling')
            else:
                self.train_loader = torch.utils.data.DataLoader(
                    self.train_loader_bf,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    drop_last=True,
                    pin_memory=True
                )
                print('Random Sampling')

        logger.info("Train {} with {} train samples, {} valid samples and {}"
                    " test samples".format(args.dataset,
                                           self.train_loader_size,
                                           self.valid_loader_size,
                                           self.test_loader_size))

        # create model:
        if self.is_default_model:
            self.net_type = 'default'
            net = DefaultCNN(add_z_struct_bottleneck=self.add_z_struct_bottleneck,
                             add_classification_layer=self.add_classification_layer,
                             z_struct_size=self.z_struct_size,
                             classif_layer_size=self.classif_layer_size)
        elif self.is_custom_model_BK:
            self.net_type = 'Custom_CNN_BK'
            net = Custom_CNN_BK(z_struct_size=self.z_struct_size,
                                big_kernel_size=self.big_kernel_size,
                                stride_size=self.stride_size,
                                classif_layer_size=self.classif_layer_size,
                                add_classification_layer=self.add_classification_layer,
                                hidden_filters_1=self.hidden_filters_1,
                                hidden_filters_2=self.hidden_filters_2,
                                hidden_filters_3=self.hidden_filters_3,
                                BK_in_first_layer=self.BK_in_first_layer,
                                two_conv_layer=self.two_conv_layer,
                                three_conv_layer=self.three_conv_layer,
                                BK_in_second_layer=self.BK_in_second_layer,
                                BK_in_third_layer=self.BK_in_third_layer,
                                Binary_z=self.binary_z,
                                add_linear_after_GMP=self.add_linear_after_GMP)
        elif self.is_custom_model:
            self.net_type = 'Custom_CNN'
            net = Custom_CNN(z_struct_size=self.z_struct_size,
                             stride_size=self.stride_size,
                             classif_layer_size=self.classif_layer_size,
                             add_classification_layer=self.add_classification_layer,
                             hidden_filters_1=self.hidden_filters_1,
                             hidden_filters_2=self.hidden_filters_2,
                             hidden_filters_3=self.hidden_filters_3,
                             two_conv_layer=self.two_conv_layer,
                             three_conv_layer=self.three_conv_layer)

        # get layer num to extract z_struct:
        self.z_struct_out = True
        self.z_struct_layer_num = get_layer_zstruct_num(net)

        if self.contrastive_loss:
            # DML Losses
            if self.loss == 'Proxy_Anchor':
                if torch.cuda.is_available():
                    self.criterion = losses.Proxy_Anchor(nb_classes=self.nb_class, sz_embed=self.sz_embedding,
                                                         mrg=self.mrg,
                                                         alpha=args.alpha).cuda()
                else:
                    self.criterion = losses.Proxy_Anchor(nb_classes=self.nb_class, sz_embed=self.sz_embedding,
                                                         mrg=self.mrg,
                                                         alpha=args.alpha)
            elif self.loss == 'Proxy_NCA':
                if torch.cuda.is_available():
                    self.criterion = losses.Proxy_NCA(nb_classes=self.nb_class, sz_embed=self.sz_embedding).cuda()
                else:
                    self.criterion = losses.Proxy_NCA(nb_classes=self.nb_class, sz_embed=self.sz_embedding)
            elif self.loss == 'MS':
                if torch.cuda.is_available():
                    self.criterion = losses.MultiSimilarityLoss().cuda()
                else:
                    self.criterion = losses.MultiSimilarityLoss()
            elif self.loss == 'Contrastive':
                if torch.cuda.is_available():
                    self.criterion = losses.ContrastiveLoss().cuda()
                else:
                    self.criterion = losses.ContrastiveLoss()
            elif self.loss == 'Triplet':
                if torch.cuda.is_available():
                    self.criterion = losses.TripletLoss().cuda()
                else:
                    self.criterion = losses.TripletLoss()
            elif self.loss == 'NPair':
                if torch.cuda.is_available():
                    self.criterion = losses.NPairLoss().cuda()
                else:
                    self.criterion = losses.NPairLoss()

        # experience name:
        if self.use_decoder:
            self.checkpoint_dir = os.path.join(args.ckpt_dir, args.exp_name.split('_decoder')[0])
        else:
            self.checkpoint_dir = os.path.join(args.ckpt_dir, args.exp_name)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.checkpoint_dir_scores = os.path.join(args.ckpt_dir_scores, args.exp_name)
        if not os.path.exists(self.checkpoint_dir_scores):
            os.makedirs(self.checkpoint_dir_scores, exist_ok=True)

        # checkpoint save:
        self.file_path_checkpoint_scores = os.path.join(self.checkpoint_dir_scores, 'last')
        if not os.path.exists(self.file_path_checkpoint_scores):
            if self.use_decoder:
                self.checkpoint_scores = {'iter': [],
                                          'epochs': [],
                                          'MSE_decoder': []}
            else:
                self.checkpoint_scores = {'iter': [],
                                          'epochs': [],
                                          'train_score': [],
                                          'train_loss_class': [],
                                          'test_score': [],
                                          'test_loss_class': [],
                                          'ratio_train_loss': [],
                                          'ratio_test_loss': [],
                                          'var_distance_classes_train': [],
                                          'var_distance_classes_test': [],
                                          'mean_distance_intra_class_train': [],
                                          'mean_distance_intra_class_test': [],
                                          'total_loss_train': [],
                                          'total_loss_test': []}
            with open(self.file_path_checkpoint_scores, mode='wb+') as f:
                torch.save(self.checkpoint_scores, f)

        # config gpu:
        self.net, self.device = gpu_config(net)

        # load checkpoints:
        self.load_checkpoint_scores('last')
        self.load_checkpoint('last')

        # create encoder + decoder decoder:
        if self.use_decoder:
            pre_trained_model = nn.Sequential(*[self.net.model[i] for i in range(self.z_struct_layer_num + 1)])
            input_test = torch.rand(*self.img_size).to(self.device)
            before_GMP_shape = self.net.net[:self.z_struct_layer_num-2](input_test.unsqueeze(0)).data.shape

            net = Encoder_decoder(z_struct_size=self.z_struct_size,
                                  big_kernel_size=self.big_kernel_size,
                                  stride_size=self.stride_size,
                                  classif_layer_size=self.classif_layer_size,
                                  add_classification_layer=self.add_classification_layer,
                                  hidden_filters_1=self.hidden_filters_1,
                                  hidden_filters_2=self.hidden_filters_2,
                                  hidden_filters_3=self.hidden_filters_3,
                                  BK_in_first_layer=self.BK_in_first_layer,
                                  two_conv_layer=self.two_conv_layer,
                                  three_conv_layer=self.three_conv_layer,
                                  BK_in_second_layer=self.BK_in_second_layer,
                                  BK_in_third_layer=self.BK_in_third_layer,
                                  Binary_z=self.binary_z,
                                  add_linear_after_GMP=self.add_linear_after_GMP,
                                  before_GMP_shape=before_GMP_shape)

            self.checkpoint_dir = os.path.join(args.ckpt_dir, args.exp_name)
            file_path = os.path.join(self.checkpoint_dir, 'last')
            self.checkpoint_dir_encoder = os.path.join(args.ckpt_dir, args.exp_name.split('_decoder')[0])
            file_path_encoder = os.path.join(self.checkpoint_dir_encoder, 'last')
            if os.path.isfile(file_path):
                print("encoder decoder already exist load it !")
                # config gpu:
                self.net, self.device = gpu_config(net)
                self.load_checkpoint('last')
            elif os.path.isfile(file_path_encoder):
                print("decoder doesn't exist load encoder weighs !")
                # checkpoint save:
                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir, exist_ok=True)
                pretrained_dict = pre_trained_model.state_dict()
                model_dict = net.encoder.state_dict()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                net.encoder.load_state_dict(model_dict)
                self.net, self.device = gpu_config(net)
            else:
                print("encoder doesn't exist, create encoder and decoder")
                self.net, self.device = gpu_config(net)

        # print model characteristics:
        print(self.net)
        num_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('The number of parameters of model is', num_params)

        # for data parallel
        if 'parallel' in str(type(self.net)):
            self.net = self.net.module
        else:
            self.net = self.net

        # TODO: add optimizer choice
        self.optimizer = optimizer.Adam(self.net.parameters(), lr=self.lr)

        # Decays the learning rate of each parameter group by gamma every step_size epochs.
        # Notice that such decay can happen simultaneously with other changes to the learning rate
        # from outside this scheduler. When last_epoch=-1, sets initial lr as lr.
        """
        mode:  lr will be reduced when the quantity monitored has stopped decreasing.
        factor: Factor by which the learning rate will be reduced.
        patience: Number of epochs with no improvement after which learning rate will be reduced.
        """
        if self.use_scheduler:
            self.scheduler = ReduceLROnPlateau(self.optimizer,
                                               mode='min',
                                               factor=0.2,
                                               patience=7,
                                               min_lr=1e-6,
                                               verbose=True)

        # initialize the early_stopping object
        # early stopping patience; how long to wait after last time validation loss improved.
        if self.use_early_stopping:
            self.patience = 10
            self.early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        # other parameters for train:
        self.global_iter = 0
        self.epochs = 0
        if self.contrastive_loss:
            self.pbar = tqdm(enumerate(self.train_loader))
        self.losses_list = []
        self.best_recall = [0]
        self.best_epoch = 0

    def train(self):
        self.net_mode(train=True)

        out = False

        print_bar = tqdm(total=self.max_iter)
        print_bar.update(self.epochs)
        while not out:
            for batch_idx, (data, labels) in enumerate(self.train_loader):

                losses_per_epoch = []

                self.global_iter += 1
                self.epochs = self.global_iter / len(self.train_loader)
                print_bar.update(1)

                data = data.to(self.device)  # Variable(data.to(self.device))
                labels = labels   # Variable(labels.to(self.device))

                if self.use_decoder:
                    x_recons, z_struct = self.net(data)
                    self.mse_loss = F.mse_loss(x_recons, data)
                    loss = self.mse_loss
                else:
                    self.mse_loss = 0
                    prediction, embedding, ratio, \
                    variance_distance_iter_class, \
                    variance_intra, mean_distance_intra_class, \
                    variance_inter = self.net(data,
                                              labels=labels,
                                              nb_class=self.nb_class,
                                              use_ratio=self.ratio_reg,
                                              z_struct_out=self.z_struct_out,
                                              z_struct_layer_num=self.z_struct_layer_num,
                                              other_ratio=self.other_ratio,
                                              loss_min_distance_cl=self.loss_min_distance_cl)

                    loss = 0
                    # compute losses:
                    if not self.without_acc:
                        # averaged over each loss element in the batch
                        Classification_loss = F.nll_loss(prediction, labels)
                        Classification_loss = Classification_loss * self.lambda_classification
                        loss += Classification_loss

                    if self.intra_class_variance_loss:
                        intra_class_loss = variance_intra * self.lambda_intra_class_var
                        loss += intra_class_loss

                    if self.contrastive_loss:
                        # loss take embedding, not prediction
                        embedding = embedding.squeeze(axis=-1).squeeze(axis=-1)
                        contrastive_loss = self.criterion(embedding, labels)
                        contrastive_loss = contrastive_loss * self.lambda_contrastive_loss
                        loss += contrastive_loss

                    if self.ratio_reg:
                        # ratio loss:
                        # if self.other_ratio:
                        #     ratio = -(ratio * self.lambda_ratio_reg)
                        # else:
                        #     ratio = ratio * self.lambda_ratio_reg
                        loss_ratio = (self.lambda_var_intra * variance_intra) - (self.lambda_var_inter * variance_inter)
                        # print(loss_ratio)
                        loss += loss_ratio

                    if self.loss_min_distance_cl:
                        loss_distance_cl = variance_distance_iter_class * self.lambda_var_distance
                        loss += loss_distance_cl

                    if self.loss_distance_mean:
                        # to avoid distance mean be too hight we want distance closest to target_mean value
                        target_mean = torch.tensor(self.value_target_distance_mean)
                        target_mean = target_mean.to(self.device)
                        loss_distance_mean = -(
                            torch.abs(1 / (target_mean - mean_distance_intra_class + EPS))) * self.lambda_distance_mean
                        loss += loss_distance_mean

                # freeze encoder if train decoder:
                if self.use_decoder and not self.test_all_learn:
                    for params in self.net.encoder.parameters():
                        params.requires_grad = False

                    # # passing only those parameters that explicitly requires grad
                    # self.optimizer = optimizer.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                    #                                 lr=self.lr)

                # print('-----------::::::::::::Before:::::::-----------------:')
                # print(self.net.encoder[3].weight[0][0])
                # print(self.net.decoder[3].weight[0])
                # print(list(self.net.parameters())[0])

                # backpropagation loss
                self.optimizer.zero_grad()
                loss.backward()
                # print('loss', loss)
                self.optimizer.step()

                # unfreeze encoder if train decoder:
                if self.use_decoder and not self.test_all_learn:
                    for params in self.net.encoder.parameters():
                        params.requires_grad = True
                    # self.optimizer.add_param_group({'params': self.net.encoder.parameters()})

                # print('-----------::::::::::::After:::::::-----------------:')
                # print(self.net.encoder[3].weight[0][0])
                # print(self.net.decoder[3].weight[0])

                if self.contrastive_loss:
                    torch.nn.utils.clip_grad_value_(self.net.parameters(), 10)
                    if self.loss == 'Proxy_Anchor':
                        torch.nn.utils.clip_grad_value_(self.criterion.parameters(), 10)
                    if torch.cuda.is_available():
                        losses_per_epoch.append(contrastive_loss.data.cpu().numpy())
                    else:
                        losses_per_epoch.append(contrastive_loss.data.numpy())
                    self.pbar.set_description(
                        'Train Epoch: {} [{}/{} ({:.0f}%)] Contrastive Loss: {:.6f}'.format(
                            self.epochs, batch_idx + 1, len(self.train_loader),
                                         100. * batch_idx / len(self.train_loader),
                            contrastive_loss.item()))

            # save step
            self.save_checkpoint('last')
            self.net_mode(train=False)

            if not self.use_decoder:
                self.scores, self.losses = compute_scores_and_loss(self.net,
                                                                   self.train_loader,
                                                                   self.test_loader,
                                                                   self.device,
                                                                   self.train_loader_size,
                                                                   self.test_loader_size,
                                                                   self.net_type,
                                                                   self.nb_class,
                                                                   self.ratio_reg,
                                                                   self.other_ratio,
                                                                   self.loss_min_distance_cl,
                                                                   self.z_struct_layer_num,
                                                                   self.loss_distance_mean)

            self.save_checkpoint_scores_loss()
            self.net_mode(train=True)

            if self.contrastive_loss:
                self.losses_list.append(np.mean(losses_per_epoch))

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            if self.use_early_stopping:
                self.early_stopping(loss, self.net)

            if self.use_decoder:
                print_bar.write('[Save Checkpoint] epoch: [{:.1f}], Train MSE:{:.5f},'.format(self.epochs, self.mse_loss))
            else:
                print_bar.write('[Save Checkpoint] epoch: [{:.1f}], Train score:{:.5f}, Test score:{:.5f}, '
                                'train loss:{:.5f}, test loss:{:.5f}, ratio_train_loss:{:.5f},'
                                'ratio_test_loss:{:.5f}, total_loss_train:{:.5f},'
                                'total_loss_test:{:.5f}, var distance inter class train:{:.5f},'
                                'var distance inter class test:{:.5f}'.format(self.epochs,
                                                                              self.scores['train'],
                                                                              self.scores['test'],
                                                                              self.losses['train_class'],
                                                                              self.losses['test_class'],
                                                                              self.losses['ratio_train_loss'],
                                                                              self.losses['ratio_test_loss'],
                                                                              self.losses['total_loss_train'],
                                                                              self.losses['total_loss_test'],
                                                                              self.losses['var_distance_classes_train'],
                                                                              self.losses['var_distance_classes_test']))

            if self.use_early_stopping:
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    out = True
                    break
            if self.epochs >= self.max_iter:
                out = True
                break

        print_bar.write("[Training Finished]")
        print_bar.close()

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise Exception('Only bool type is supported. True or False')
        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint_scores_loss(self, silent=False):
        """
        save all sample_scores and loss values
        """
        self.checkpoint_scores['iter'].append(self.global_iter)
        self.checkpoint_scores['epochs'].append(self.epochs)
        if not self.use_decoder:
            # sample_scores
            self.checkpoint_scores['train_score'].append(self.scores['train'])
            self.checkpoint_scores['test_score'].append(self.scores['test'])
            # losses
            self.checkpoint_scores['train_loss_class'].append(self.losses['train_class'])
            self.checkpoint_scores['test_loss_class'].append(self.losses['test_class'])
            self.checkpoint_scores['ratio_train_loss'].append(self.losses['ratio_train_loss'])
            self.checkpoint_scores['ratio_test_loss'].append(self.losses['ratio_test_loss'])
            self.checkpoint_scores['var_distance_classes_train'].append(self.losses['var_distance_classes_train'])
            self.checkpoint_scores['var_distance_classes_test'].append(self.losses['var_distance_classes_test'])
            self.checkpoint_scores['total_loss_train'].append(self.losses['total_loss_train'])
            self.checkpoint_scores['total_loss_test'].append(self.losses['total_loss_test'])
            self.checkpoint_scores['mean_distance_intra_class_train'].append(self.losses['mean_distance_intra_class_train'])
            self.checkpoint_scores['mean_distance_intra_class_test'].append(self.losses['mean_distance_intra_class_test'])
        else:
            self.checkpoint_scores['MSE_decoder'].append(self.mse_loss)

        with open(self.file_path_checkpoint_scores, mode='wb+') as f:
            torch.save(self.checkpoint_scores, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {}, epoch {})".format(self.file_path_checkpoint_scores,
                                                                        self.global_iter, self.epochs))

    def save_checkpoint(self, filename, silent=False):
        model = {'net': self.net.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'iter': self.global_iter,
                 'epochs': self.epochs
                 }
        file_path = os.path.join(self.checkpoint_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(model, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {}, epoch {})".format(file_path, self.global_iter, self.epochs))

    def load_checkpoint_scores(self, filename):
        file_path = os.path.join(self.checkpoint_dir_scores, filename)
        if os.path.isfile(file_path):
            if os.path.getsize(file_path) > 0:  # check that the file is not empty
                with open(file_path, "rb"):
                    if not torch.cuda.is_available():
                        self.checkpoint_scores = torch.load(file_path, map_location=torch.device('cpu'))
                    else:
                        self.checkpoint_scores = torch.load(file_path)
                    print("=> loaded sample_scores and losses")
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.checkpoint_dir, filename)
        if os.path.isfile(file_path):
            if not torch.cuda.is_available():
                checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.epochs = checkpoint['epochs']
            self.net.load_state_dict(checkpoint['net'])
            # self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
