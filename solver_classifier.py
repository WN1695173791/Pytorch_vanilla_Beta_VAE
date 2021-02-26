import logging
import os

import torch
import torch.nn.functional as F
import torch.optim as optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm

import losses
from dataset import sampler
from dataset.dataset_2 import get_dataloaders, get_mnist_dataset
from models.custom_CNN import Custom_CNN
from models.custom_CNN_BK import Custom_CNN_BK
from models.custom_CNN_BK import compute_ratio_batch
from models.default_CNN import DefaultCNN
from pytorchtools import EarlyStopping
from scores_classifier import compute_scores
from solver import gpu_config
from visualizer_CNN import get_layer_zstruct_num, compute_z_struct
import numpy as np
from torch.autograd import Variable


def compute_scores_and_loss(net, train_loader, test_loader, device, train_loader_size, test_loader_size,
                            net_type, nb_class, ratio_reg, other_ratio):
    score_train, loss_train = compute_scores(net,
                                             train_loader,
                                             device,
                                             train_loader_size)
    score_test, loss_test = compute_scores(net,
                                           test_loader,
                                           device,
                                           test_loader_size)
    if ratio_reg:
        # compute ratio on all test set:
        z_struct_representation_test, labels_batch_test = compute_z_struct(net,
                                                                           'exp_name',
                                                                           test_loader,
                                                                           train_test='None',
                                                                           net_type=net_type,
                                                                           return_results=True)
        ratio_test = compute_ratio_batch(z_struct_representation_test, labels_batch_test, nb_class)
        # compute ratio on all train set:
        z_struct_representation_train, labels_batch_train = compute_z_struct(net,
                                                                             'exp_name',
                                                                             train_loader,
                                                                             train_test='None',
                                                                             net_type=net_type,
                                                                             return_results=True)
        ratio_train = compute_ratio_batch(z_struct_representation_train,
                                          labels_batch_train,
                                          nb_class,
                                          other_ratio=other_ratio)
    else:
        ratio_test = 0
        ratio_train = 0

    scores = {'train': score_train, 'test': score_test}
    losses = {'train_class': loss_train,
              'test_class': loss_test,
              'ratio_test_loss': ratio_test,
              'ratio_train_loss': ratio_train,
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

        # wandb parameters:
        self.use_wandb = False
        if self.use_wandb:
            import wandb

        # Directory for Log
        if self.use_wandb:
            LOG_DIR = args.LOG_DIR + '/logs_{}/{}_{}_embedding_{}_alpha{}_mrg{}_{}_lr{}_batch{}{}'.format(self.dataset,
                                                                                                          self.model,
                                                                                                          self.loss,
                                                                                                          self.sz_embedding,
                                                                                                          self.alpha,
                                                                                                          self.mrg,
                                                                                                          self.optimizer,
                                                                                                          self.lr,
                                                                                                          self.batch_size,
                                                                                                          self.remark)

        # Wandb Initialization
        if self.use_wandb:
            wandb.init(project=args.dataset + '_ProxyAnchor', notes=LOG_DIR)
            wandb.config.update(args)

        # dataset parameters:
        if args.dataset.lower() == 'mnist':
            self.img_size = (1, 32, 32)
            self.nb_class = 10
            self.nc = 1
        else:
            raise NotImplementedError
        self.nb_pixels = self.img_size[1] * self.img_size[2]

        # initialize the early_stopping object
        # early stopping patience; how long to wait after last time validation loss improved.
        if self.use_early_stopping:
            self.patience = 20
            self.early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        # logger
        formatter = logging.Formatter('%(asc_time)s %(level_name)s - %(funcName)s: %(message)s', "%H:%M:%S")
        logger = logging.getLogger(__name__)
        logger.setLevel(args.log_level.upper())
        stream = logging.StreamHandler()
        stream.setLevel(args.log_level.upper())
        stream.setFormatter(formatter)
        logger.addHandler(stream)

        # load dataset:
        if args.dataset == 'mnist':
            self.valid_loader = 0
            self.train_loader, self.test_loader = get_mnist_dataset(batch_size=self.batch_size)
        else:
            self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(args.dataset,
                                                                                     batch_size=self.batch_size,
                                                                                     logger=logger)

        self.train_loader_size = len(self.train_loader.dataset)
        if self.valid_loader == 0:
            self.valid_loader_size = 0
        else:
            self.valid_loader_size = len(self.valid_loader.dataset)
        self.test_loader_size = len(self.test_loader.dataset)

        if self.contrastive_loss:
            if self.IPC:
                balanced_sampler = sampler.BalancedSampler(self.train_loader,
                                                           batch_size=self.batch_size,
                                                           images_per_class=self.IPC)
                batch_sampler = BatchSampler(balanced_sampler,
                                             batch_size=self.batch_size,
                                             drop_last=True)
                self.dl_tr = torch.utils.data.DataLoader(
                    self.train_loader,
                    num_workers=self.num_workers,
                    pin_memory=True,
                    batch_sampler=batch_sampler
                )
                print('Balanced Sampling')
            else:
                self.dl_tr = torch.utils.data.DataLoader(
                    self.train_loader,
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

        # create model
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

        # print model characteristics:
        print(net)
        num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print('The number of parameters of model is', num_params)

        # get layer num to extract z_struct:
        self.z_struct_out = True
        self.z_struct_layer_num = get_layer_zstruct_num(net, self.net_type)

        # config gpu:
        self.net, self.device = gpu_config(net)

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
                                               patience=5,
                                               min_lr=1e-6,
                                               verbose=True)

        if self.contrastive_loss:
            # DML Losses
            if self.loss == 'Proxy_Anchor':
                if torch.cuda.is_available():
                    self.criterion = losses.Proxy_Anchor(nb_classes=self.nb_class, sz_embed=self.sz_embedding, mrg=self.mrg,
                                                         alpha=args.alpha).cuda()
                else:
                    self.criterion = losses.Proxy_Anchor(nb_classes=self.nb_class, sz_embed=self.sz_embedding, mrg=self.mrg,
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

        if 'parallel' in str(type(self.net)):
            self.net = self.net.module
        else:
            self.net = self.net

        # experience name
        self.checkpoint_dir = os.path.join(args.ckpt_dir, args.exp_name)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.checkpoint_dir_scores = os.path.join(args.ckpt_dir_scores, args.exp_name)
        if not os.path.exists(self.checkpoint_dir_scores):
            os.makedirs(self.checkpoint_dir_scores, exist_ok=True)

        # checkpoint save
        self.file_path_checkpoint_scores = os.path.join(self.checkpoint_dir_scores, 'last')
        if not os.path.exists(self.file_path_checkpoint_scores):
            self.checkpoint_scores = {'iter': [],
                                      'epochs': [],
                                      'train_score': [],
                                      'train_loss_class': [],
                                      'test_score': [],
                                      'test_loss_class': [],
                                      'ratio_train_loss': [],
                                      'ratio_test_loss': [],
                                      'total_loss_train': [],
                                      'total_loss_test': []}
            with open(self.file_path_checkpoint_scores, mode='wb+') as f:
                torch.save(self.checkpoint_scores, f)

        # load checkpoints:
        self.load_checkpoint_scores('last')
        self.load_checkpoint('last')

        # other parameters for train:
        self.global_iter = 0
        self.epochs = 0
        self.Classification_loss = 0
        self.Total_loss = 0
        self.scores = 0
        self.losses = 0
        self.ratio = 0

        if self.contrastive_loss:
            self.pbar = tqdm(enumerate(self.dl_tr))
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
                labels = labels.to(self.device)  # Variable(labels.to(self.device))

                prediction, embedding, ratio = self.net(data,
                                                         labels=labels,
                                                         nb_class=self.nb_class,
                                                         use_ratio=self.ratio_reg,
                                                         z_struct_out=self.z_struct_out,
                                                         z_struct_layer_num=self.z_struct_layer_num,
                                                         other_ratio=self.other_ratio)

                # classification loss
                # averaged over each loss element in the batch
                self.Classification_loss = F.nll_loss(prediction, labels)
                self.Classification_loss = self.lambda_classification * self.Classification_loss

                if self.contrastive_loss:
                    # loss take emnedding, not prediciton
                    embedding = embedding.squeeze(axis=-1).squeeze(axis=-1)
                    loss = self.criterion(embedding, labels)
                    loss = Variable(loss.data, requires_grad=True)

                if self.ratio_reg:
                    # ratio loss:
                    if self.other_ratio:
                        self.ratio = -(ratio * self.lambda_ratio_reg)
                    else:
                        self.ratio = ratio * self.lambda_ratio_reg
                    self.ratio = Variable(self.ratio.data, requires_grad=True)

                # total loss:
                if self.without_acc:
                    if self.contrastive_loss:
                        self.total_loss = loss + self.ratio
                    else:
                        self.total_loss = self.ratio
                else:
                    if self.ratio_reg:
                        if self.contrastive_loss:
                            self.total_loss = self.Classification_loss + loss + self.ratio
                        else:
                            self.total_loss = self.Classification_loss + self.ratio
                    else:
                        if self.contrastive_loss:
                            self.total_loss = self.Classification_loss + loss
                        else:
                            self.total_loss = self.Classification_loss

                # backpropagation loss
                self.optimizer.zero_grad()
                self.total_loss.backward()

                if self.contrastive_loss:
                    torch.nn.utils.clip_grad_value_(self.net.parameters(), 10)
                    if self.loss == 'Proxy_Anchor':
                        torch.nn.utils.clip_grad_value_(self.criterion.parameters(), 10)

                    if torch.cuda.is_available():
                        losses_per_epoch.append(loss.data.cpu().numpy())
                    else:
                        losses_per_epoch.append(loss.data.numpy())

                    self.pbar.set_description(
                        'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                            self.epochs, batch_idx + 1, len(self.dl_tr),
                                         100. * batch_idx / len(self.dl_tr),
                            loss.item()))

                self.optimizer.step()

            # save step
            self.save_checkpoint('last')
            self.net_mode(train=False)

            self.scores, self.losses = compute_scores_and_loss(self.net,
                                                               self.train_loader,
                                                               self.test_loader,
                                                               self.device,
                                                               self.train_loader_size,
                                                               self.test_loader_size,
                                                               self.net_type,
                                                               self.nb_class,
                                                               self.ratio_reg,
                                                               self.other_ratio)

            if self.contrastive_loss:
                self.losses_list.append(np.mean(losses_per_epoch))
            self.epochs = int(self.epochs)

            if self.use_wandb:
                if self.contrastive_loss:
                    wandb.log({'contrastive loss': self.losses_list[-1]}, step=self.epochs)
                wandb.log({'ratio': self.losses['ratio_test_loss']}, step=self.epochs)
                wandb.log({'acc': self.scores['test']}, step=self.epochs)
                wandb.log({'Total loss': self.total_loss}, step=self.epochs)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            if self.use_early_stopping:
                self.early_stopping(self.losses['total_loss_test'], self.net)

            self.save_checkpoint_scores_loss()
            self.net_mode(train=True)

            print_bar.write('[Save Checkpoint] epoch: [{:.1f}], Train score:{:.5f}, Test score:{:.5f}, '
                            'train loss:{:.5f}, test loss:{:.5f}, ratio_train_loss:{:.5f},'
                            'ratio_test_loss:{:.5f}, total_loss_train:{:.5f},'
                            'total_loss_test:{:.5f}, '.format(self.epochs,
                                                              self.scores['train'],
                                                              self.scores['test'],
                                                              self.losses['train_class'],
                                                              self.losses['test_class'],
                                                              self.losses['ratio_train_loss'],
                                                              self.losses['ratio_test_loss'],
                                                              self.losses['total_loss_train'],
                                                              self.losses['total_loss_test']))
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
        # sample_scores
        self.checkpoint_scores['train_score'].append(self.scores['train'])
        self.checkpoint_scores['test_score'].append(self.scores['test'])
        # losses
        self.checkpoint_scores['train_loss_class'].append(self.losses['train_class'])
        self.checkpoint_scores['test_loss_class'].append(self.losses['test_class'])
        self.checkpoint_scores['ratio_train_loss'].append(self.losses['ratio_train_loss'])
        self.checkpoint_scores['ratio_test_loss'].append(self.losses['ratio_test_loss'])
        self.checkpoint_scores['total_loss_train'].append(self.losses['total_loss_train'])
        self.checkpoint_scores['total_loss_test'].append(self.losses['total_loss_test'])

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
