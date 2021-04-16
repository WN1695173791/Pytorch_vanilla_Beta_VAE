import logging

import torch.nn.functional as F
import torch.optim as optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm

from dataset.dataset_2 import get_dataloaders, get_mnist_dataset
from models.vae_var import VAE_var
from models.encoder_struct import Encoder_struct
from pytorchtools import EarlyStopping
from scores_classifier import compute_scores, compute_scores_VAE
from solver import gpu_config
from visualizer_CNN import get_layer_zstruct_num, score_uniq_code
from dataset.sampler import BalancedBatchSampler
import random
from models.VAE import VAE
import torch.nn as nn

from visualizer import *

EPS = 1e-12
worker_id = 0


def seed_all(seed):
    if not seed:
        seed = 10

    print("[Using Seed: {}]".format(seed))

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
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_lambda_uniq_code_dst(epoch, nb_epoch, lambda_loss):
    """
    get lambda value dependent of epoch number.
    nb_epoch is the epoch number from which we stop arise lambda value
    :param epoch:
    :param nb_epoch:
    :param n:
    :return:
    """
    if epoch > nb_epoch:
        x = float(nb_epoch)
    else:
        x = float(epoch)
    return (torch.exp(torch.tensor(x)) / torch.exp(torch.tensor(nb_epoch-2))) * lambda_loss


def compute_scores_and_loss(net, train_loader, test_loader, device, train_loader_size, test_loader_size, nb_class,
                            ratio_reg, other_ratio, loss_min_distance_cl, z_struct_layer_num, contrastive_criterion,
                            without_acc, lambda_classification,
                            lambda_contrastive, lambda_ratio_reg, diff_var, lambda_var_intra, lambda_var_inter,
                            lambda_var_distance, lambda_distance_mean, z_struct_out, Hmg_dst_loss, lambda_hmg_dst):
    score_train, classification_loss_train, total_loss_iter_train, ratio_loss_train, contrastive_loss_train, \
    diff_var_loss_train, variance_intra_train, \
    variance_inter_train, loss_distance_cl_train, loss_distance_mean_train, total_loss_train = compute_scores(net,
                                                                                                              train_loader,
                                                                                                              device,
                                                                                                              train_loader_size,
                                                                                                              nb_class,
                                                                                                              ratio_reg,
                                                                                                              z_struct_layer_num,
                                                                                                              other_ratio,
                                                                                                              loss_min_distance_cl,
                                                                                                              contrastive_criterion,
                                                                                                              without_acc,
                                                                                                              lambda_classification,
                                                                                                              lambda_contrastive,
                                                                                                              lambda_ratio_reg,
                                                                                                              diff_var,
                                                                                                              lambda_var_intra,
                                                                                                              lambda_var_inter,
                                                                                                              lambda_var_distance,
                                                                                                              lambda_distance_mean,
                                                                                                              z_struct_out,
                                                                                                              Hmg_dst_loss,
                                                                                                              lambda_hmg_dst)
    score_test, classification_loss_test, total_loss_iter_test, ratio_loss_test, contrastive_loss_test, \
    diff_var_loss_test, variance_intra_test, \
    variance_inter_test, loss_distance_cl_test, loss_distance_mean_test, total_loss_test = compute_scores(net,
                                                                                                          test_loader,
                                                                                                          device,
                                                                                                          test_loader_size,
                                                                                                          nb_class,
                                                                                                          ratio_reg,
                                                                                                          z_struct_layer_num,
                                                                                                          other_ratio,
                                                                                                          loss_min_distance_cl,
                                                                                                          contrastive_criterion,
                                                                                                          without_acc,
                                                                                                          lambda_classification,
                                                                                                          lambda_contrastive,
                                                                                                          lambda_ratio_reg,
                                                                                                          diff_var,
                                                                                                          lambda_var_intra,
                                                                                                          lambda_var_inter,
                                                                                                          lambda_var_distance,
                                                                                                          lambda_distance_mean,
                                                                                                          z_struct_out,
                                                                                                          Hmg_dst_loss,
                                                                                                          lambda_hmg_dst)

    scores = {'train': score_train, 'test': score_test}
    losses = {'total_train': total_loss_train,
              'total_test': total_loss_test,
              'ratio_test_loss': ratio_loss_test,
              'ratio_train_loss': ratio_loss_train,
              'var_distance_classes_train': loss_distance_cl_train,
              'var_distance_classes_test': loss_distance_cl_test,
              'mean_distance_intra_class_train': loss_distance_mean_train,
              'mean_distance_intra_class_test': loss_distance_mean_test,
              'classification_test': classification_loss_test,
              'classification_train': classification_loss_train,
              'contrastive_train': contrastive_loss_train,
              'contrastive_test': contrastive_loss_test,
              'diff_var_train': diff_var_loss_train,
              'diff_var_test': diff_var_loss_test,
              'intra_var_train': variance_intra_test,
              'intra_var_test': variance_intra_train,
              'inter_var_train': variance_inter_train,
              'inter_var_test': variance_inter_test}

    return scores, losses


def compute_scores_and_loss_VAE(net, train_loader, test_loader, train_loader_size, test_loader_size, device,
                                lambda_BCE, beta, is_vae_var, ES_reconstruction, EV_classifier):
    Total_loss_train, BCE_train, KLD_train = compute_scores_VAE(net,
                                                                train_loader,
                                                                train_loader_size,
                                                                device,
                                                                lambda_BCE,
                                                                beta,
                                                                is_vae_var,
                                                                ES_reconstruction,
                                                                EV_classifier)
    Total_loss_test, BCE_test, KLD_test = compute_scores_VAE(net,
                                                             test_loader,
                                                             test_loader_size,
                                                             device,
                                                             lambda_BCE,
                                                             beta,
                                                             is_vae_var,
                                                             ES_reconstruction,
                                                             EV_classifier)
    losses = {'Total_loss_train': Total_loss_train,
              'BCE_train': BCE_train,
              'KLD_train': KLD_train,
              'Total_loss_test': Total_loss_test,
              'BCE_test': BCE_test,
              'KLD_test': KLD_test}

    return losses


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
        self.big_kernel_size = args.big_kernel_size
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
        self.binary_chain = args.binary_chain
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
        self.lambda_intra_class_var = args.lambda_intra_class_var
        # intra class max distance loss:
        self.lambda_distance_mean = args.lambda_distance_mean
        self.loss_distance_mean = args.loss_distance_mean
        self.dataset_balanced = args.dataset_balanced
        self.value_target_distance_mean = args.value_target_distance_mean
        # decoder:
        self.other_architecture = args.other_architecture
        self.freeze_Encoder = args.freeze_Encoder
        self.diff_var = args.diff_var
        self.add_linear_after_GMP = args.add_linear_after_GMP
        self.decoder_first_dense = args.decoder_first_dense
        self.decoder_n_filter_1 = args.decoder_n_filter_1
        self.decoder_n_filter_2 = args.decoder_n_filter_2
        self.decoder_n_filter_3 = args.decoder_n_filter_3
        self.decoder_kernel_size_1 = args.decoder_kernel_size_1
        self.decoder_kernel_size_2 = args.decoder_kernel_size_2
        self.decoder_kernel_size_3 = args.decoder_kernel_size_3
        self.decoder_stride_1 = args.decoder_stride_1
        self.decoder_stride_2 = args.decoder_stride_2
        self.decoder_stride_3 = args.decoder_stride_3
        # for reproductibility:
        self.randomness = args.randomness
        self.random_seed = args.random_seed
        # encoder struct:
        self.is_encoder_struct = args.is_encoder_struct
        self.kernel_size_1 = args.kernel_size_1
        self.kernel_size_2 = args.kernel_size_2
        self.kernel_size_3 = args.kernel_size_3
        self.binary_first_conv = args.binary_first_conv
        self.binary_second_conv = args.binary_second_conv
        self.binary_third_conv = args.binary_third_conv
        self.Hmg_dst_loss = args.Hmg_dst_loss
        self.lambda_hmg_dst = args.lambda_hmg_dst
        self.uniq_code_dst_loss = args.uniq_code_dst_loss
        self.lambda_uniq_code_dst = args.lambda_uniq_code_dst
        self.max_epoch_use_uniq_code_target = args.max_epoch_use_uniq_code_target
        self.add_dl_class = args.add_dl_class
        self.hidden_dim = args.hidden_dim
        self.bin_after_GMP = args.bin_after_GMP
        self.L2_dst_loss = args.L2_dst_loss
        self.lambda_L2_dst = args.lambda_L2_dst
        # VAE var:
        self.is_VAE_var = args.is_VAE_var
        self.var_second_cnn_block = args.var_second_cnn_block
        self.var_third_cnn_block = args.var_third_cnn_block
        self.other_architecture = args.other_architecture
        self.z_var_size = args.z_var_size
        self.EV_classifier = args.EV_classifier
        self.grad_inv = args.grad_inv
        self.PT_model = args.PT_model
        self.PT_model_EV_classifier = args.PT_model_EV_classifier
        # VAE parameters:
        self.is_VAE = args.is_VAE
        self.lambda_BCE = args.lambda_BCE
        self.beta = args.beta
        self.encoder_var_name = args.encoder_var_name
        self.encoder_struct_name = args.encoder_struct_name
        self.use_small_lr_encoder_var = args.use_small_lr_encoder_var
        self.both_decoders_freeze = args.both_decoders_freeze
        self.ES_reconstruction = args.ES_reconstruction

        self.contrastive_criterion = False
        if self.is_encoder_struct:
            self.use_VAE = False
        elif self.is_VAE or self.is_VAE_var:
            self.use_VAE = True

        # For reproducibility:
        if self.randomness:
            seed_all(self.random_seed)

        # logger
        formatter = logging.Formatter('%(asc_time)s %(level_name)s - %(funcName)s: %(message)s', "%H:%M:%S")
        logger = logging.getLogger(__name__)
        logger.setLevel(args.log_level.upper())
        stream = logging.StreamHandler()
        stream.setLevel(args.log_level.upper())
        stream.setFormatter(formatter)
        logger.addHandler(stream)

        # ___________________________________________ load dataset:___________________________________________________
        # dataset parameters:
        if args.dataset.lower() == 'mnist':
            self.img_size = (1, 32, 32)
            self.nb_class = 10
            self.nc = 1
        else:
            raise NotImplementedError
        self.nb_pixels = self.img_size[1] * self.img_size[2]

        if args.dataset == 'mnist':
            if self.dataset_balanced:
                # Load balanced mnist dataset:
                self.train_loader_bf, _ = get_mnist_dataset(batch_size=self.batch_size, return_Dataloader=False)

                self.train_loader = torch.utils.data.DataLoader(self.train_loader_bf,
                                                                sampler=BalancedBatchSampler(self.train_loader_bf),
                                                                batch_size=self.batch_size)
                _, self.test_loader = get_mnist_dataset(batch_size=self.batch_size)
                print('Balanced dataset loaded')
            else:
                # load mnist dataset without balanced data
                self.train_loader, self.test_loader = get_mnist_dataset(batch_size=self.batch_size)

        self.train_loader_size = len(self.train_loader.dataset)
        self.test_loader_size = len(self.test_loader.dataset)

        # ___________________________________________ end dataset:___________________________________________________

        logger.info("Dataset {}: train with {} samples and {} test samples".format(args.dataset,
                                                                                   self.train_loader_size,
                                                                                   self.test_loader_size))

        # ___________________________________________create model:_________________________________________________
        if self.is_encoder_struct:
            self.net_type = 'encoder_struct'
            net = Encoder_struct(z_struct_size=self.z_struct_size,
                                 big_kernel_size=self.big_kernel_size,
                                 stride_size=self.stride_size,
                                 kernel_size_1=self.kernel_size_1,
                                 kernel_size_2=self.kernel_size_2,
                                 kernel_size_3=self.kernel_size_3,
                                 hidden_filters_1=self.hidden_filters_1,
                                 hidden_filters_2=self.hidden_filters_2,
                                 hidden_filters_3=self.hidden_filters_3,
                                 BK_in_first_layer=self.BK_in_first_layer,
                                 BK_in_second_layer=self.BK_in_second_layer,
                                 BK_in_third_layer=self.BK_in_third_layer,
                                 two_conv_layer=self.two_conv_layer,
                                 three_conv_layer=self.three_conv_layer,
                                 Binary_z=self.binary_z,
                                 binary_first_conv=self.binary_first_conv,
                                 binary_second_conv=self.binary_second_conv,
                                 binary_third_conv=self.binary_third_conv,
                                 add_dl_class=self.add_dl_class,
                                 hidden_dim=self.hidden_dim,
                                 bin_after_GMP=self.bin_after_GMP)
        elif self.is_VAE_var:
            self.net_type = 'vae_var'
            net = VAE_var(z_var_size=self.z_var_size,
                          var_second_cnn_block=self.var_second_cnn_block,
                          var_third_cnn_block=self.var_third_cnn_block,
                          other_architecture=self.other_architecture,
                          EV_classifier=self.EV_classifier,
                          n_classes=self.nb_class,
                          grad_inv=self.grad_inv)
        elif self.is_VAE:
            self.net_type = 'VAE'
            net = VAE(z_var_size=self.z_var_size,
                      var_second_cnn_block=self.var_second_cnn_block,
                      var_third_cnn_block=self.var_third_cnn_block,
                      other_architecture=self.other_architecture,
                      z_struct_size=self.z_struct_size,
                      big_kernel_size=self.big_kernel_size,
                      stride_size=self.stride_size,
                      kernel_size_1=self.kernel_size_1,
                      kernel_size_2=self.kernel_size_2,
                      kernel_size_3=self.kernel_size_3,
                      hidden_filters_1=self.hidden_filters_1,
                      hidden_filters_2=self.hidden_filters_2,
                      hidden_filters_3=self.hidden_filters_3,
                      BK_in_first_layer=self.BK_in_first_layer,
                      BK_in_second_layer=self.BK_in_second_layer,
                      BK_in_third_layer=self.BK_in_third_layer,
                      two_conv_layer=self.two_conv_layer,
                      three_conv_layer=self.three_conv_layer,
                      Binary_z=self.binary_z,
                      binary_first_conv=self.binary_first_conv,
                      binary_second_conv=self.binary_second_conv,
                      binary_third_conv=self.binary_third_conv,
                      ES_reconstruction=self.ES_reconstruction)

        # get layer num to extract z_struct:
        self.z_struct_out = True

        if self.is_encoder_struct:
            # z_struct_layer_num depend to architecture model:
            if self.bin_after_GMP and self.L2_dst_loss:
                # in this case we want to compute dst in continue z_struct so before binarization:
                add_layer = 1
            elif self.bin_after_GMP:
                # We want use binary z_struct:
                add_layer = 3
            else:
                add_layer = 1
            self.z_struct_layer_num = get_layer_zstruct_num(net, add_layer=add_layer)

        # experience name:
        self.checkpoint_dir = os.path.join(args.ckpt_dir, args.exp_name)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.checkpoint_dir_scores = os.path.join(args.ckpt_dir_scores, args.exp_name)
        if not os.path.exists(self.checkpoint_dir_scores):
            os.makedirs(self.checkpoint_dir_scores, exist_ok=True)

        # checkpoint save:
        self.file_path_checkpoint_scores = os.path.join(self.checkpoint_dir_scores, 'last')
        if not os.path.exists(self.file_path_checkpoint_scores):
            if self.use_VAE:
                self.checkpoint_scores = {'iter': [],
                                          'epochs': [],
                                          'BCE_train': [],
                                          'KLD_train': [],
                                          'Total_loss_train': [],
                                          'BCE_test': [],
                                          'KLD_test': [],
                                          'Total_loss_test': []}
            else:
                self.checkpoint_scores = {'iter': [],
                                          'epochs': [],
                                          'train_score': [],
                                          'test_score': [],
                                          'total_train': [],
                                          'total_test': [],
                                          'ratio_train_loss': [],
                                          'ratio_test_loss': [],
                                          'var_distance_classes_train': [],
                                          'var_distance_classes_test': [],
                                          'mean_distance_intra_class_train': [],
                                          'mean_distance_intra_class_test': [],
                                          'intra_var_train': [],
                                          'intra_var_test': [],
                                          'inter_var_train': [],
                                          'inter_var_test': [],
                                          'diff_var_train': [],
                                          'diff_var_test': [],
                                          'contrastive_train': [],
                                          'contrastive_test': [],
                                          'classification_test': [],
                                          'classification_train': []}
            with open(self.file_path_checkpoint_scores, mode='wb+') as f:
                torch.save(self.checkpoint_scores, f)

        # Load weights for models:
        if self.is_encoder_struct or self.is_VAE_var:
            if (self.Hmg_dst_loss or self.uniq_code_dst_loss) and self.is_encoder_struct:
                # We load pre trained weights if in exp name they are "_PT":
                # model name before "_PT" must correspond to encoder struct pre trained name.
                print('USe encoder struct with pre trained weighs encoder, load it !')
                self.checkpoint_dir = os.path.join(args.ckpt_dir, args.exp_name.split('_PT')[0])
                self.net, self.device = gpu_config(net)
                self.load_checkpoint('last')
                # after load weights we replace real exp name for checkpoint directory.
                self.checkpoint_dir = os.path.join(args.ckpt_dir, args.exp_name)
            else:
                if self.EV_classifier or self.PT_model:
                    print("Train classifier from encoder var pre-trained !")
                    checkpoint_dir_encoder_var = os.path.join(args.ckpt_dir, self.encoder_var_name)
                    file_path_encoder_var = os.path.join(checkpoint_dir_encoder_var, 'last')
                    if os.path.isfile(file_path_encoder_var):
                        print("Load encoder var weights !")
                        # we create and load encoder struct with model name associate:
                        # EV_classifier parameter for PT model:

                        pre_trained_var_model = VAE_var(z_var_size=self.z_var_size,
                                                        var_second_cnn_block=self.var_second_cnn_block,
                                                        var_third_cnn_block=self.var_third_cnn_block,
                                                        other_architecture=self.other_architecture,
                                                        EV_classifier=self.PT_model_EV_classifier,
                                                        n_classes=self.nb_class,
                                                        grad_inv=self.grad_inv)
                        # load weighs:
                        pre_trained_var_model = self.load_pre_trained_checkpoint('last',
                                                                                 checkpoint_dir_encoder_var,
                                                                                 pre_trained_var_model)
                        # get weighs dict of pre trained model:
                        pretrained_dict = pre_trained_var_model.encoder_var.state_dict()
                        # get dict of weighs for model VAE create (with init weights)
                        model_dict = net.encoder_var.state_dict()
                        # copy weighs pre trained in current model for same name layer:
                        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                        model_dict.update(pretrained_dict)
                        # Load pre trained weighs in net model:
                        net.encoder_var.load_state_dict(model_dict)
                        print("Weighs loaded for var encoder !")
                        # to device:
                        self.net, self.device = gpu_config(net)
                    else:
                        self.net, self.device = gpu_config(net)
                        self.load_checkpoint('last')
                else:
                    self.net, self.device = gpu_config(net)
                    self.load_checkpoint('last')
        elif self.is_VAE:
                # real checkpoint dir name:
                self.checkpoint_dir = os.path.join(args.ckpt_dir, args.exp_name)
                file_path = os.path.join(self.checkpoint_dir, 'last')
                if os.path.isfile(file_path):
                    print("VAE already exist load it !")
                    # config gpu:
                    self.net, self.device = gpu_config(net)
                    self.load_checkpoint('last')
                else:
                    # If path encoder name exist we load associate weighs:
                    checkpoint_dir_encoder_var = os.path.join(args.ckpt_dir, self.encoder_var_name)
                    file_path_encoder_var = os.path.join(checkpoint_dir_encoder_var, 'last')
                    checkpoint_dir_encoder_struct = os.path.join(args.ckpt_dir, self.encoder_struct_name)
                    file_path_encoder_struct = os.path.join(checkpoint_dir_encoder_struct, 'last')

                    if os.path.isfile(file_path_encoder_var):
                        print("VAE doesn't exist but encoder var yes ! load var encoder weighs !")
                        # we create and load encoder struct with model name associate:
                        pre_trained_var_model = VAE_var(z_var_size=self.z_var_size,
                                                        var_second_cnn_block=self.var_second_cnn_block,
                                                        var_third_cnn_block=self.var_third_cnn_block,
                                                        other_architecture=self.other_architecture)

                        # load weighs:
                        pre_trained_var_model = self.load_pre_trained_checkpoint('last',
                                                                                 checkpoint_dir_encoder_var,
                                                                                 pre_trained_var_model)

                        # get weighs dict of pre trained model:
                        pretrained_dict = pre_trained_var_model.encoder_var.state_dict()
                        # get dict of weighs for model VAE create (with init weights)
                        model_dict = net.encoder_var.state_dict()
                        # copy weighs pre trained in current model for same name layer:
                        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                        model_dict.update(pretrained_dict)
                        # Load pre trained weighs in net model:
                        net.encoder_var.load_state_dict(model_dict)

                        print("Weighs loaded for var encoder !")
                    if os.path.isfile(file_path_encoder_struct):
                        print("VAE doesn't exist but encoder struct yes ! load struct encoder weighs !")
                        # we create and load encoder struct with model name associate:
                        pre_trained_struct_model = Encoder_struct(z_struct_size=self.z_struct_size,
                                                                  big_kernel_size=self.big_kernel_size,
                                                                  stride_size=self.stride_size,
                                                                  kernel_size_1=self.kernel_size_1,
                                                                  kernel_size_2=self.kernel_size_2,
                                                                  kernel_size_3=self.kernel_size_3,
                                                                  hidden_filters_1=self.hidden_filters_1,
                                                                  hidden_filters_2=self.hidden_filters_2,
                                                                  hidden_filters_3=self.hidden_filters_3,
                                                                  BK_in_first_layer=self.BK_in_first_layer,
                                                                  BK_in_second_layer=self.BK_in_second_layer,
                                                                  BK_in_third_layer=self.BK_in_third_layer,
                                                                  two_conv_layer=self.two_conv_layer,
                                                                  three_conv_layer=self.three_conv_layer,
                                                                  Binary_z=self.binary_z,
                                                                  binary_first_conv=self.binary_first_conv,
                                                                  binary_second_conv=self.binary_second_conv,
                                                                  binary_third_conv=self.binary_third_conv)

                        # load weighs:
                        pre_trained_struct_model = self.load_pre_trained_checkpoint('last',
                                                                                    checkpoint_dir_encoder_struct,
                                                                                    pre_trained_struct_model)

                        # get weighs dict of pre trained model:
                        pretrained_dict = pre_trained_struct_model.encoder_struct.state_dict()
                        # get dict of weighs for model VAE create (with init weights)
                        model_dict = net.encoder_struct.state_dict()
                        # copy weighs pre trained in current model for same name layer:
                        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                        model_dict.update(pretrained_dict)
                        # Load pre trained weighs in net model:
                        net.encoder_struct.load_state_dict(model_dict)

                        print("Weighs loaded for struct encoder !")
                    if (not os.path.isfile(file_path_encoder_var)) and (not os.path.isfile(file_path_encoder_struct)):
                        print("VAE doesn't exist, create VAE and train it from scratch ! good luck !")
                    # to device:
                    self.net, self.device = gpu_config(net)
        else:
            print("Error, not net type exist !")

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
        if self.use_small_lr_encoder_var:
            print('use smaller lr for encoder var !')
            self.optimizer = optimizer.Adam([{'params': self.net.decoder.parameters()},
                                            {'params': self.net.encoder_var.parameters(), 'lr': 1e-5}
                                             ], lr=self.lr)
        else:
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
                                               patience=4,
                                               min_lr=1e-6,
                                               verbose=True)

        # initialize the early_stopping object
        # early stopping patience; how long to wait after last time validation loss improved.
        if self.use_early_stopping:
            self.patience = 25
            self.early_stopping = EarlyStopping(patience=self.patience, verbose=True)

        # other parameters for train:
        self.global_iter = 0
        self.epochs = 0
        self.losses_list = []
        self.best_recall = [0]
        self.best_epoch = 0
        self.use_uniq_bin_code_target = False
        self.target_code = None

        # init lambda uniq code target:
        self.lambda_uc = 0

    def train(self):
        self.net_mode(train=True)

        out = False

        print_bar = tqdm(total=self.max_iter)
        print_bar.update(self.epochs)

        while not out:
            for batch_idx, (data, labels) in enumerate(self.train_loader):

                self.global_iter += 1
                self.epochs = self.global_iter / len(self.train_loader)
                print_bar.update(1)

                data = data.to(self.device)  # Variable(data.to(self.device))
                labels = labels.to(self.device)  # Variable(labels.to(self.device))

                if self.use_VAE:
                    if self.is_VAE:
                        x_recons, z_struct, z_var, z_var_sample, latent_representation, z = self.net(data)
                    else:
                        x_recons, latent_representation, prediction_var = self.net(data)

                    if self.EV_classifier:
                        classification_loss = F.nll_loss(prediction_var, labels)
                        loss = classification_loss
                    else:
                        if self.ES_reconstruction:
                            BCE_loss = F.mse_loss(x_recons, data, size_average=False)
                            loss = BCE_loss
                        else:
                            mu = latent_representation['mu']
                            log_var = latent_representation['log_var']

                            # BCE tries to make our reconstruction as accurate as possible:
                            # BCE_loss = F.binary_cross_entropy(x_recons, data, size_average=False)
                            BCE_loss = F.mse_loss(x_recons, data, size_average=False)

                            # KLD tries to push the distributions as close as possible to unit Gaussian:
                            KLD_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

                            loss = (self.lambda_BCE * BCE_loss) + (self.beta * KLD_loss)
                else:
                    self.mse_loss = 0
                    loss = 0
                    prediction, embedding, ratio, \
                    variance_distance_iter_class, \
                    variance_intra, mean_distance_intra_class, \
                    variance_inter, global_avg_Hmg_dst, \
                    avg_dst_classes, uniq_target_dist_loss, \
                    global_avg_L2_dst, avg_L2_dst_classes = self.net(data,
                                                                     labels=labels,
                                                                     nb_class=self.nb_class,
                                                                     use_ratio=self.ratio_reg,
                                                                     z_struct_out=self.z_struct_out,
                                                                     z_struct_layer_num=self.z_struct_layer_num,
                                                                     loss_min_distance_cl=self.loss_min_distance_cl,
                                                                     Hmg_dst_loss=self.Hmg_dst_loss,
                                                                     uniq_bin_code_target=self.use_uniq_bin_code_target,
                                                                     target_code=self.target_code,
                                                                     L2_dst_loss=self.L2_dst_loss)

                    # compute losses:
                    if not self.without_acc:
                        # averaged over each loss element in the batch
                        Classification_loss = F.nll_loss(prediction, labels)
                        Classification_loss = Classification_loss * self.lambda_classification
                        loss += Classification_loss

                    if self.ratio_reg:
                        # ratio loss:
                        if self.other_ratio:
                            ratio = -(ratio * self.lambda_ratio_reg)
                        else:
                            ratio = ratio * self.lambda_ratio_reg
                        loss += ratio

                    if self.diff_var:
                        loss_diff_var = (self.lambda_var_intra * variance_intra) - \
                                        (self.lambda_var_inter * variance_inter)
                        loss += loss_diff_var

                    if self.loss_min_distance_cl:
                        # variance of distances between each class with other
                        loss_distance_cl = variance_distance_iter_class * self.lambda_var_distance
                        loss += loss_distance_cl

                    if self.loss_distance_mean:
                        # to avoid distance mean be too high we want distance closest to target_mean value
                        target_mean = torch.tensor(self.value_target_distance_mean)
                        target_mean = target_mean.to(self.device)
                        loss_distance_mean = -(
                            torch.abs(1 / (target_mean - mean_distance_intra_class + EPS))) * self.lambda_distance_mean
                        loss += loss_distance_mean

                    if self.Hmg_dst_loss:
                        # Hamming distance is not differentiable
                        # global_avg_Hmg_dst = Variable(global_avg_Hmg_dst, requires_grad=True)
                        # global_avg_Hmg_dst.requires_grad_(True)
                        loss += global_avg_Hmg_dst * self.lambda_hmg_dst

                    if self.L2_dst_loss:
                        # Hamming distance is not differentiable
                        # global_avg_Hmg_dst = Variable(global_avg_Hmg_dst, requires_grad=True)
                        # global_avg_Hmg_dst.requires_grad_(True)
                        loss += global_avg_L2_dst * self.lambda_L2_dst

                    if self.uniq_code_dst_loss and self.use_uniq_bin_code_target:
                        dst_target_loss = uniq_target_dist_loss * self.lambda_uc
                        loss += dst_target_loss

                # freeze encoder_struct if train decoder:
                if self.use_VAE and self.freeze_Encoder:
                    if self.is_VAE_var:
                        for params in self.net.encoder_var.parameters():
                            params.requires_grad = False
                    else:
                        if self.both_decoders_freeze:
                            for params in self.net.encoder_var.parameters():
                                params.requires_grad = False
                        for params in self.net.encoder_struct.parameters():
                            params.requires_grad = False

                # print('-----------::::::::::::Before:::::::-----------------:')
                # print(self.net.encoder_struct[3].weight[24][12])
                # print(self.net.encoder_var[3].weight[24][12])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # unfreeze encoder_struct if train decoder:
                if self.use_VAE and self.freeze_Encoder:
                    if self.is_VAE_var:
                        for params in self.net.encoder_var.parameters():
                            params.requires_grad = False
                    else:
                        if self.both_decoders_freeze:
                            for params in self.net.encoder_var.parameters():
                                params.requires_grad = False
                        for params in self.net.encoder_struct.parameters():
                            params.requires_grad = False

                # print('-----------::::::::::::After:::::::-----------------:')
                # print(self.net.encoder_struct[3].weight[24][12])
                # print(self.net.encoder_var[3].weight[24][12])

            if self.uniq_code_dst_loss:
                print(self.epochs, " Compute uniq code target:")
                self.use_uniq_bin_code_target = True
                # test if we can use uniq code for target latent code:
                score, self.target_code = score_uniq_code(self.net,
                                                          self.test_loader,
                                                          self.device,
                                                          self.z_struct_layer_num,
                                                          self.nb_class,
                                                          self.z_struct_size,
                                                          self.test_loader_size)
                # updte lambda value:
                print(self.max_epoch_use_uniq_code_target)
                self.lambda_uc = get_lambda_uniq_code_dst(self.epochs,
                                                          self.max_epoch_use_uniq_code_target,
                                                          self.lambda_uniq_code_dst)
                # if score == 100.:
                #     self.use_uniq_bin_code_target = True
                print('Uniq code computing: lambda: {}, score: {}, uniq codes: {}'.format(self.lambda_uc,
                                                                                          score,
                                                                                          self.target_code))

            # backpropagation loss
            if self.use_scheduler:
                self.scheduler.step(loss)
                # print('Epoch:', self.epochs, 'LR:', self.scheduler.get_lr())

            # save step
            self.save_checkpoint('last')
            self.net_mode(train=False)

            if self.use_VAE:
                self.losses = compute_scores_and_loss_VAE(self.net,
                                                          self.train_loader,
                                                          self.test_loader,
                                                          self.train_loader_size,
                                                          self.test_loader_size,
                                                          self.device,
                                                          self.lambda_BCE,
                                                          self.beta,
                                                          self.is_VAE_var,
                                                          self.ES_reconstruction,
                                                          self.EV_classifier)
            elif self.is_encoder_struct:
                self.scores, self.losses = compute_scores_and_loss(self.net,
                                                                   self.train_loader,
                                                                   self.test_loader,
                                                                   self.device,
                                                                   self.train_loader_size,
                                                                   self.test_loader_size,
                                                                   self.nb_class,
                                                                   self.ratio_reg,
                                                                   self.other_ratio,
                                                                   self.loss_min_distance_cl,
                                                                   self.z_struct_layer_num,
                                                                   self.contrastive_criterion,
                                                                   self.without_acc,
                                                                   self.lambda_classification,
                                                                   self.lambda_contrastive_loss,
                                                                   self.lambda_ratio_reg,
                                                                   self.diff_var,
                                                                   self.lambda_var_intra,
                                                                   self.lambda_var_inter,
                                                                   self.lambda_var_distance,
                                                                   self.lambda_distance_mean,
                                                                   self.z_struct_out,
                                                                   self.Hmg_dst_loss,
                                                                   self.lambda_hmg_dst)

            self.save_checkpoint_scores_loss()
            self.net_mode(train=True)

            # early_stopping needs the validation loss to check if it has decresed,
            # and if it has, it will make a checkpoint of the current model
            if self.use_early_stopping:
                self.early_stopping(loss, self.net)

            if self.use_VAE:
                print_bar.write(
                    '[Save Checkpoint] epoch: [{:.1f}], Train: total:{:.5f}, BCE:{:.5f}, KLD:{:.5f},'
                    'Test: total:{:.5f}, BCE:{:.5f}, KLD:{:.5f}'.format(self.epochs,
                                                                        self.losses['Total_loss_train'],
                                                                        self.losses['BCE_train'],
                                                                        self.losses['KLD_train'],
                                                                        self.losses['Total_loss_test'],
                                                                        self.losses['BCE_test'],
                                                                        self.losses['KLD_test']))
            else:
                print_bar.write('[Save Checkpoint] epoch: [{:.1f}], Train score:{:.5f}, Test score:{:.5f}, '
                                'train loss:{:.5f}, test loss:{:.5f}, ratio_train_loss:{:.5f},'
                                'ratio_test_loss:{:.5f}, var distance inter class train:{:.5f},'
                                'var distance inter class test:{:.5f}'.format(self.epochs,
                                                                              self.scores['train'],
                                                                              self.scores['test'],
                                                                              self.losses['total_train'],
                                                                              self.losses['total_test'],
                                                                              self.losses['ratio_train_loss'],
                                                                              self.losses['ratio_test_loss'],
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
        if self.is_encoder_struct:
            # sample_scores
            self.checkpoint_scores['train_score'].append(self.scores['train'])
            self.checkpoint_scores['test_score'].append(self.scores['test'])
            # losses
            self.checkpoint_scores['total_train'].append(self.losses['total_train'])
            self.checkpoint_scores['total_test'].append(self.losses['total_test'])
            self.checkpoint_scores['ratio_train_loss'].append(self.losses['ratio_train_loss'])
            self.checkpoint_scores['ratio_test_loss'].append(self.losses['ratio_test_loss'])
            self.checkpoint_scores['var_distance_classes_train'].append(self.losses['var_distance_classes_train'])
            self.checkpoint_scores['var_distance_classes_test'].append(self.losses['var_distance_classes_test'])
            self.checkpoint_scores['mean_distance_intra_class_train'].append(
                self.losses['mean_distance_intra_class_train'])
            self.checkpoint_scores['mean_distance_intra_class_test'].append(
                self.losses['mean_distance_intra_class_test'])
            self.checkpoint_scores['intra_var_train'].append(self.losses['intra_var_train'])
            self.checkpoint_scores['intra_var_test'].append(self.losses['intra_var_test'])
            self.checkpoint_scores['inter_var_train'].append(self.losses['inter_var_train'])
            self.checkpoint_scores['inter_var_test'].append(self.losses['inter_var_test'])
            self.checkpoint_scores['diff_var_train'].append(self.losses['diff_var_train'])
            self.checkpoint_scores['diff_var_test'].append(self.losses['diff_var_test'])
            self.checkpoint_scores['contrastive_train'].append(self.losses['contrastive_train'])
            self.checkpoint_scores['contrastive_test'].append(self.losses['contrastive_test'])
            self.checkpoint_scores['classification_test'].append(self.losses['classification_test'])
            self.checkpoint_scores['classification_train'].append(self.losses['classification_train'])
        elif self.use_VAE:
            self.checkpoint_scores['Total_loss_train'].append(self.losses['Total_loss_train'])
            self.checkpoint_scores['BCE_train'].append(self.losses['BCE_train'])
            self.checkpoint_scores['KLD_train'].append(self.losses['KLD_train'])
            self.checkpoint_scores['Total_loss_test'].append(self.losses['Total_loss_test'])
            self.checkpoint_scores['BCE_test'].append(self.losses['BCE_test'])
            self.checkpoint_scores['KLD_test'].append(self.losses['KLD_test'])

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

    def load_pre_trained_checkpoint(self, filename, checkpoint_dir, model):
        file_path = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(file_path):
            if not torch.cuda.is_available():
                checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(file_path)
            global_iter = checkpoint['iter']
            model.load_state_dict(checkpoint['net'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, global_iter))

            return model
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
