import torch
import logging
import torch.optim as optimizer
import torch.nn.functional as F
import os
from torchvision import transforms
import torchvision.datasets as datasets

from dataset.dataset_2 import get_dataloaders
from models.default_CNN import DefaultCNN
from models.custom_CNN_BK import Custom_CNN_BK
from models.custom_CNN import Custom_CNN
from solver import gpu_config
from tqdm import tqdm
from scores_classifier import compute_scores
from visualizer_CNN import get_layer_zstruct_num


def compute_scores_and_loss(net, train_loader, test_loader, device, train_loader_size, test_loader_size):
    score_train, loss_train = compute_scores(net,
                                             train_loader,
                                             device,
                                             train_loader_size)
    score_test, loss_test = compute_scores(net,
                                           test_loader,
                                           device,
                                           test_loader_size)

    scores = {'train': score_train, 'test': score_test}
    losses = {'train': loss_train, 'test': loss_test}

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

        # Custom CNN parameters:
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
        # binary parameters:
        self.binary_z = args.binary_z

        # dataset parameters:
        if args.dataset.lower() == 'mnist':
            self.img_size = (1, 32, 32)
            self.nb_class = 10
            self.nc = 1
        else:
            raise NotImplementedError
        self.nb_pixels = self.img_size[1] * self.img_size[2]

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
            mnist_trainset = datasets.MNIST(root='../data/mnist',
                                            train=True,
                                            download=True,
                                            transform=transforms.Compose([transforms.Resize(32),
                                                                          transforms.ToTensor()]))
            mnist_testset = datasets.MNIST(root='../data/mnist',
                                           train=False,
                                           download=True,
                                           transform=transforms.Compose([transforms.Resize(32),
                                                                         transforms.ToTensor()]))

            self.train_loader = torch.utils.data.DataLoader(dataset=mnist_trainset,
                                                       batch_size=self.batch_size,
                                                       shuffle=True)
            self.test_loader = torch.utils.data.DataLoader(dataset=mnist_testset,
                                                      batch_size=self.batch_size,
                                                      shuffle=False)
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
                                Binary_z=self.binary_z)
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

        # config gpu:
        self.net, self.device = gpu_config(net)
        self.optimizer = optimizer.Adam(self.net.parameters(), lr=self.lr)

        if 'parallel' in str(type(self.net)):
            self.net = self.net.module
        else:
            self.net = self.net

        # experience name
        self.checkpoint_dir = os.path.join(args.ckpt_dir, args.expe_name)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.checkpoint_dir_scores = os.path.join(args.ckpt_dir_scores, args.expe_name)
        if not os.path.exists(self.checkpoint_dir_scores):
            os.makedirs(self.checkpoint_dir_scores, exist_ok=True)

        # checkpoint save
        self.file_path_checkpoint_scores = os.path.join(self.checkpoint_dir_scores, 'last')
        if not os.path.exists(self.file_path_checkpoint_scores):
            self.checkpoint_scores = {'iter': [],
                                      'epochs': [],
                                      'train_score': [],
                                      'train_loss': [],
                                      'test_score': [],
                                      'test_loss': []
                                      }
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

        # z_struct_layer_num:
        self.z_struct_layer_num = get_layer_zstruct_num(self.net, self.net_type)

    def train(self):
        self.net_mode(train=True)

        out = False

        print_bar = tqdm(total=self.max_iter)
        print_bar.update(self.epochs)
        while not out:
            for data, labels in self.train_loader:
                self.global_iter += 1
                self.epochs = self.global_iter / len(self.train_loader)
                print_bar.update(1)

                data = data.to(self.device)  # Variable(data.to(self.device))
                labels = labels.to(self.device)  # Variable(labels.to(self.device))

                prediction, _ = self.net(data)

                # classification loss
                # averaged over each loss element in the batch
                self.Classification_loss = F.nll_loss(prediction, labels)

                self.Total_loss = self.Classification_loss
                # backpropagation loss
                self.optimizer.zero_grad()
                self.Total_loss.backward()
                self.optimizer.step()

            # save step
            self.save_checkpoint('last')
            self.net_mode(train=False)

            self.scores, self.losses = compute_scores_and_loss(self.net,
                                                               self.train_loader,
                                                               self.test_loader,
                                                               self.device,
                                                               self.train_loader_size,
                                                               self.test_loader_size)

            self.save_checkpoint_scores_loss()
            self.net_mode(train=True)

            print_bar.write('[Save Checkpoint] epoch: [{:.1f}], Train score:{:.5f}, Test score:{:.5f}, '
                            'train loss:{:.5f}, test loss:{:.5f}'.format(self.epochs,
                                                                         self.scores['train'],
                                                                         self.scores['test'],
                                                                         self.losses['train'],
                                                                         self.losses['test']))

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
        self.checkpoint_scores['train_loss'].append(self.losses['train'])
        self.checkpoint_scores['test_loss'].append(self.losses['test'])

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
