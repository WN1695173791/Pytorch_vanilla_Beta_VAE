from models.custom_CNN_BK import Custom_CNN_BK
from models.custom_CNN import Custom_CNN
from dataset.dataset_2 import get_dataloaders, get_mnist_dataset

from visualizer import *
from visualizer_CNN import *
from viz.viz_regions import *
from scores_classifier import compute_scores
from models.Encoder_decoder import Encoder_decoder
from models.encoder_struct import Encoder_struct
from models.vae_var import VAE_var

from models.VAE import VAE
from visualizer import viz_reconstruction
from viz.visualize import Visualizer
import torch.nn.functional as F

import csv


def str2bool(string):
    return string.lower() in ("yes", "true", "t", "1")


# ___________________________________ begin extrraction parameters models _____________________________________________
def run_exp_extraction_and_visualization_custom_BK(list_model, is_ratio=False, is_decoder=False, is_VAE=False,
                                                   is_custom=False, is_encoder_struct=False):
    for exp_name in list_model:

        # load csv parameters:
        exp_csv_name = 'args_parser/' + exp_name + '.csv'
        with open(exp_csv_name, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                parameters_dict = row

        # model:
        model_name = parameters_dict['exp_name']
        Binary_z = str2bool(parameters_dict['binary_z'])
        binary_chain = str2bool(parameters_dict['binary_chain'])

        # Parameters:
        stride_size = int(parameters_dict['stride_size'])

        # losses:
        diff_var = str2bool(parameters_dict['diff_var'])
        loss_distance_mean = str2bool(parameters_dict['loss_distance_mean'])
        loss_min_distance_cl = str2bool(parameters_dict['loss_min_distance_cl'])

        # VAE parameters:
        lambda_BCE = float(parameters_dict['lambda_BCE'])
        beta = float(parameters_dict['beta'])
        z_struct_size = int(parameters_dict['z_struct_size'])
        decoder_first_dense = int(parameters_dict['decoder_first_dense'])
        decoder_n_filter_1 = int(parameters_dict['decoder_n_filter_1'])
        decoder_n_filter_2 = int(parameters_dict['decoder_n_filter_2'])
        decoder_kernel_size_1 = int(parameters_dict['decoder_kernel_size_1'])
        decoder_kernel_size_2 = int(parameters_dict['decoder_kernel_size_2'])
        decoder_kernel_size_3 = int(parameters_dict['decoder_kernel_size_3'])
        decoder_stride_1 = int(parameters_dict['decoder_stride_1'])
        decoder_stride_2 = int(parameters_dict['decoder_stride_2'])
        decoder_stride_3 = int(parameters_dict['decoder_stride_3'])
        other_architecture = parameters_dict['other_architecture']

        # Encoder struct parameters:
        big_kernel_size = int(parameters_dict['big_kernel_size'])
        BK_in_first_layer = str2bool(parameters_dict['BK_in_first_layer'])
        BK_in_second_layer = str2bool(parameters_dict['BK_in_second_layer'])
        BK_in_third_layer = str2bool(parameters_dict['BK_in_third_layer'])
        two_conv_layer = str2bool(parameters_dict['two_conv_layer'])
        three_conv_layer = str2bool(parameters_dict['three_conv_layer'])

        classif_layer_size = int(parameters_dict['classif_layer_size'])
        add_classification_layer = str2bool(parameters_dict['add_classification_layer'])
        add_linear_after_GMP = str2bool(parameters_dict['add_linear_after_GMP'])

        # Encoder var parameters:
        z_var_size = int(parameters_dict['z_var_size'])
        var_hidden_filters_1 = int(parameters_dict['var_hidden_filters_1'])
        var_hidden_filters_2 = int(parameters_dict['var_hidden_filters_2'])
        var_hidden_filters_3 = int(parameters_dict['var_hidden_filters_3'])
        var_kernel_size_1 = int(parameters_dict['var_kernel_size_1'])
        var_kernel_size_2 = int(parameters_dict['var_kernel_size_2'])
        var_kernel_size_3 = int(parameters_dict['var_kernel_size_3'])
        var_stride_size_1 = int(parameters_dict['var_stride_size_1'])
        var_stride_size_2 = int(parameters_dict['var_stride_size_2'])
        var_stride_size_3 = int(parameters_dict['var_stride_size_3'])
        var_hidden_dim = int(parameters_dict['var_hidden_dim'])
        var_three_conv_layer = str2bool(parameters_dict['var_three_conv_layer'])

        use_structural_encoder = str2bool(parameters_dict['use_structural_encoder'])

        # encoder struct:
        kernel_size_1 = int(parameters_dict['kernel_size_1'])
        kernel_size_2 = int(parameters_dict['kernel_size_2'])
        kernel_size_3 = int(parameters_dict['kernel_size_3'])
        binary_z = str2bool(parameters_dict['binary_z'])
        binary_first_conv = str2bool(parameters_dict['binary_first_conv'])
        binary_second_conv = str2bool(parameters_dict['binary_second_conv'])
        binary_third_conv = str2bool(parameters_dict['binary_third_conv'])

        # other default parameters:
        hidden_filters_1 = int(parameters_dict['hidden_filters_layer1'])
        hidden_filters_2 = int(parameters_dict['hidden_filters_layer2'])
        hidden_filters_3 = int(parameters_dict['hidden_filters_layer3'])

        # vae var:
        if is_encoder_struct:
            is_VAE_var = False
            var_second_cnn_block = False
            var_third_cnn_block = False
        else:
            is_VAE_var = str2bool(parameters_dict['is_VAE_var'])
            var_second_cnn_block = str2bool(parameters_dict['var_second_cnn_block'])
            var_third_cnn_block = str2bool(parameters_dict['var_third_cnn_block'])

        # print(binary_chain, Binary_z)
        if is_decoder:
            net = Encoder_decoder(z_struct_size=z_struct_size,
                                  big_kernel_size=big_kernel_size,
                                  stride_size=stride_size,
                                  classif_layer_size=classif_layer_size,
                                  add_classification_layer=add_classification_layer,
                                  hidden_filters_1=hidden_filters_1,
                                  hidden_filters_2=hidden_filters_2,
                                  hidden_filters_3=hidden_filters_3,
                                  BK_in_first_layer=BK_in_first_layer,
                                  two_conv_layer=two_conv_layer,
                                  three_conv_layer=three_conv_layer,
                                  BK_in_second_layer=BK_in_second_layer,
                                  BK_in_third_layer=BK_in_third_layer,
                                  Binary_z=Binary_z,
                                  other_architecture=other_architecture,
                                  add_linear_after_GMP=add_linear_after_GMP,
                                  decoder_first_dense=decoder_first_dense,
                                  decoder_n_filter_1=decoder_n_filter_1,
                                  decoder_n_filter_2=decoder_n_filter_2,
                                  decoder_kernel_size_1=decoder_kernel_size_1,
                                  decoder_kernel_size_2=decoder_kernel_size_2,
                                  decoder_kernel_size_3=decoder_kernel_size_3,
                                  decoder_stride_1=decoder_stride_1,
                                  decoder_stride_2=decoder_stride_2,
                                  decoder_stride_3=decoder_stride_3,
                                  struct_hidden_filters_1=var_hidden_filters_1,
                                  struct_hidden_filters_2=var_hidden_filters_2,
                                  struct_hidden_filters_3=var_hidden_filters_3,
                                  struct_kernel_size_1=var_kernel_size_1,
                                  struct_kernel_size_2=var_kernel_size_2,
                                  struct_kernel_size_3=var_kernel_size_3,
                                  struct_stride_size_1=var_stride_size_1,
                                  struct_stride_size_2=var_stride_size_2,
                                  struct_stride_size_3=var_stride_size_3,
                                  struct_hidden_dim=var_hidden_dim,
                                  struct_three_conv_layer=var_three_conv_layer)
        elif is_VAE and not is_VAE_var:
            net = VAE(z_struct_size=z_struct_size,
                      big_kernel_size=big_kernel_size,
                      stride_size=stride_size,
                      hidden_filters_1=hidden_filters_1,
                      hidden_filters_2=hidden_filters_2,
                      hidden_filters_3=hidden_filters_3,
                      BK_in_first_layer=BK_in_first_layer,
                      two_conv_layer=two_conv_layer,
                      three_conv_layer=three_conv_layer,
                      BK_in_second_layer=BK_in_second_layer,
                      BK_in_third_layer=BK_in_third_layer,
                      z_var_size=z_var_size,
                      var_hidden_filters_1=var_hidden_filters_1,
                      var_hidden_filters_2=var_hidden_filters_2,
                      var_hidden_filters_3=var_hidden_filters_3,
                      var_kernel_size_1=var_kernel_size_1,
                      var_kernel_size_2=var_kernel_size_2,
                      var_kernel_size_3=var_kernel_size_3,
                      var_stride_size_1=var_stride_size_1,
                      var_stride_size_2=var_stride_size_2,
                      var_stride_size_3=var_stride_size_3,
                      var_hidden_dim=var_hidden_dim,
                      var_three_conv_layer=var_three_conv_layer)
        elif is_custom:
            net_type = 'Custom_CNN_BK'
            net = Custom_CNN_BK(z_struct_size=z_struct_size,
                                big_kernel_size=big_kernel_size,
                                stride_size=stride_size,
                                classif_layer_size=classif_layer_size,
                                add_classification_layer=add_classification_layer,
                                hidden_filters_1=hidden_filters_1,
                                hidden_filters_2=hidden_filters_2,
                                hidden_filters_3=hidden_filters_3,
                                BK_in_first_layer=BK_in_first_layer,
                                two_conv_layer=two_conv_layer,
                                three_conv_layer=three_conv_layer,
                                BK_in_second_layer=BK_in_second_layer,
                                BK_in_third_layer=BK_in_third_layer,
                                Binary_z=Binary_z,
                                binary_chain=binary_chain,
                                add_linear_after_GMP=add_linear_after_GMP)
        elif is_encoder_struct:
            net_type = 'Encoder_struct'
            net = Encoder_struct(z_struct_size=z_struct_size,
                                 big_kernel_size=big_kernel_size,
                                 stride_size=stride_size,
                                 kernel_size_1=kernel_size_1,
                                 kernel_size_2=kernel_size_2,
                                 kernel_size_3=kernel_size_3,
                                 hidden_filters_1=hidden_filters_1,
                                 hidden_filters_2=hidden_filters_2,
                                 hidden_filters_3=hidden_filters_3,
                                 BK_in_first_layer=BK_in_first_layer,
                                 BK_in_second_layer=BK_in_second_layer,
                                 BK_in_third_layer=BK_in_third_layer,
                                 two_conv_layer=two_conv_layer,
                                 three_conv_layer=three_conv_layer,
                                 Binary_z=binary_z,
                                 binary_first_conv=binary_first_conv,
                                 binary_second_conv=binary_second_conv,
                                 binary_third_conv=binary_third_conv)
        elif is_VAE_var:
            net_type = 'VAE_var'
            net = VAE_var(z_var_size=z_var_size,
                          var_second_cnn_block=var_second_cnn_block,
                          var_third_cnn_block=var_third_cnn_block)
        if is_decoder:
            run_decoder(model_name, net)
        elif is_VAE or is_VAE_var:
            run_VAE(model_name, net, lambda_BCE, beta, z_struct_size, z_var_size, use_structural_encoder, is_VAE_var)
        else:
            run_viz_expes(model_name, net, is_ratio, loss_min_distance_cl, loss_distance_mean, cat='Encoder_struct',
                          net_type=net_type, diff_var_loss=diff_var)
            # visualize_regions_of_interest(model_name, net, net_type='Custom_CNN_BK')


# ______________________________________end extrraction parameters models ______________________________________________


def run_score(exp_name, net):
    print(exp_name)
    path = 'checkpoints_CNN/'
    path_scores = 'checkpoint_scores_CNN'
    net_trained, _, nb_epochs = get_checkpoints(net, path, exp_name)
    # scores and losses:
    plot_scores_and_loss_CNN(net_trained, exp_name, path_scores, save=True)

    return


def build_compare_reconstruction(size, data, input_data, x_recon):
    # reconstructions
    num_images = int(size[0] * size[1] / 2)
    if data.shape[1] == 3:
        originals = input_data[:num_images].cpu()
    else:
        originals = input_data[:num_images].cpu()

    reconstructions = x_recon.view(-1, *img_size)[:num_images].cpu()
    # If there are fewer examples given than spaces available in grid,
    # augment with blank images
    num_examples = originals.size()[0]
    if num_images > num_examples:
        blank_images = torch.zeros((num_images - num_examples,) + originals.size()[1:])
        originals = torch.cat([originals, blank_images])
        reconstructions = torch.cat([reconstructions, blank_images])

    # Concatenate images and reconstructions
    comparison = torch.cat([originals, reconstructions])

    return comparison


def run_decoder(exp_name, net, multi_label=True):
    print(exp_name, 'run decoder')
    path = 'checkpoints_CNN/'
    path_scores = 'checkpoint_scores_CNN'
    net, _, _ = get_checkpoints(net, path, exp_name)

    print(net)
    # print('Weighs encoder struct:', net.encoder_struct[3].weight[7][25])

    loader = test_loader

    viz_decoder_multi_label(net, loader, exp_name, nb_img=10, nb_class=nb_class, save=True)

    # viz_latent_prediction_reconstruction(net_trained, exp_name, img_size, size=8, random=True, batch=batch)

    return


def run_VAE(model_name, net, lambda_BCE, beta, z_struct_size, z_var_size, VAE_struct, is_vae_var):
    print(model_name, 'run VAE')
    print(net)
    path = 'checkpoints_CNN/'
    path_scores = 'checkpoint_scores_CNN'
    net, _, _ = get_checkpoints(net, path, model_name)

    # print('Weighs encoder struct:', net.encoder_struct[3].weight[7][25])

    loader = test_loader
    train_test = 'test'
    batch = batch_test
    embedding_size = z_struct_size + z_var_size

    # losses:
    # plot_loss_results_VAE(path_scores, model_name, beta, lambda_BCE, save=True)

    # viz_latent_prediction_reconstruction(net,
    #                                      model_name,
    #                                      embedding_size,
    #                                      z_struct_size,
    #                                      z_var_size,
    #                                      size=8,
    #                                      random=False,
    #                                      batch=batch)

    # Image reconstruction with real distribution: mu_var, sigma_var, mu_struct, sigma_struct
    mu_var, sigma_var, mu_struct, sigma_struct = real_distribution_model(net,
                                                                         model_name,
                                                                         z_struct_size,
                                                                         z_var_size,
                                                                         loader,
                                                                         'test',
                                                                         plot_gaussian=True,
                                                                         save=True,
                                                                         VAE_struct=VAE_struct,
                                                                         is_vae_var=is_vae_var)

    # z_struct sample:  sample from positive gaussian:
    # z_struct_sample = sigma_struct * np.random.randn(...) + mu_struct

    viz_reconstruction_VAE(net, loader, model_name, z_var_size, z_struct_size, nb_img=10,
                           nb_class=nb_class, save=True, z_reconstruction=True,
                           z_struct_reconstruction=False, z_var_reconstruction=False,
                           return_scores=False, real_distribution=True, mu_var=mu_var, std_var=sigma_var,
                           mu_struct=mu_struct, std_struct=sigma_struct, is_vae_var=is_vae_var)

    # viz switch image:
    # switch_img(net, model_name, loader, z_var_size)

    # viz multi sample with same z_var or z_struct:
    # first: compute z_struct mean per class:
    # compute_z_struct_mean_VAE(net, model_name, loader, train_test='test', return_results=False)

    # second: get average z_struct per classe:
    # _, average_z_struct_class = get_z_struct_per_class_VAE(model_name, train_test='test', nb_class=nb_class)

    # third: plot:
    # plot_prototyoe_z_struct_per_class(1, average_z_struct_class, train_test, model_name,
    #                                   nb_class, None, device, net, z_var_size=z_var_size, save=save)
    # plot_struct_fixe_and_z_var_moove(average_z_struct_class, train_test, net,
    #                                  device, nb_class, None, model_name, z_var_size=z_var_size,
    #                                  embedding_size=embedding_size, save=save, mu_var=mu_var, std_var=sigma_var)
    # plot_var_fixe_and_z_struct_moove(average_z_struct_class, train_test, net,
    #                                  device, nb_class, None, model_name, z_struct_size=z_struct_size,
    #                                  z_var_size=z_var_size,
    #                                  embedding_size=embedding_size, save=save, mu_struct=mu_struct,
    #                                 std_struct=sigma_struct, traversal_latent=False)

    # score sample:
    # images_generation(net, model_name, batch, size=(8, 8), mu_var=mu_var, mu_struct=mu_struct, std_var=sigma_var,
    #                   std_struct=sigma_struct, z_var_size=z_var_size, z_struct_size=z_struct_size, FID=True, IS=True,
    #                   LPIPS=True, real_distribution=True, save=True)

    # Display a 2D manifold of the digits:
    # manifold_digit(net, model_name, device, size=(20, 20), component_var=0, component_struct=0, random=False,
    #                loader=loader, mu_var=mu_var, std_var=sigma_var, mu_struct=mu_struct, std_struct=sigma_struct,
    #                z_var_size=z_var_size, z_struct_size=z_struct_size, img_choice=None)

    return


def run_viz_expes(model_name, net, is_ratio, is_distance_loss, loss_distance_mean, net_type=None, cat=None,
                  ratio_reg=False, diff_var_loss=False, contrastive_loss=False):
    print('________________------------------{}-----------------_____________________'.format(model_name))
    path = 'checkpoints_CNN/'
    path_scores = 'checkpoint_scores_CNN'
    net, _, nb_epochs = get_checkpoints(net, path, model_name)
    print(net)
    net.eval()

    train_test = 'test'
    loader = test_loader
    loader_size = len(loader.dataset)

    # scores and losses:
    # plot_scores_and_loss_CNN(net, model_name, path_scores, is_ratio=ratio_reg, save=True,
    #                          is_distance_loss=is_distance_loss, loss_distance_mean=loss_distance_mean,
    #                          diff_var=diff_var_loss, contrastive_loss=contrastive_loss)
    # score_test, _, _, _, _, _, _, \
    # _, _, _, _ = compute_scores(net, loader, device, loader_size, False, False,
    #                             False, False, False, False, False, False, False, False,
    #                             False, False, False, False, False, False)
    # score_train, _, _, _, _, _, _, \
    # _, _, _, _ = compute_scores(net, train_loader, device, len(train_loader.dataset), False, False,
    #                             False, False, False, False, False, False, False, False,
    #                             False, False, False, False, False, False)
    # print('________________------------------{}-----------------_____________________'.format(model_name))
    # print('score Test acc: {:.3f}%'.format(score_test))

    # compute features:
    # compute_z_struct(net, model_name, loader, train_test=train_test, net_type=net_type)
    # compute_z_struct_representation_noised(net, model_name, train_test=train_test, nb_repeat=10, nb_class=nb_class,
    #                                        net_type=net_type)
    # get_z_struct_per_class(model_name, train_test=train_test, nb_class=nb_class)
    # get_average_z_struct_per_classes(exp_name=model_name, train_test=train_test)
    # get_prediction_per_classes(model_name, train_test=train_test)
    # get_prediction_noised_per_class(model_name, train_test=train_test)
    # compute_all_score_acc(model_name, train_test=train_test)
    # compute_mean_std_prediction(model_name, train_test=train_test)

    # receptive_field = get_receptive_field(net, img_size, net_type=net_type)

    # plot:
    # ratio_variance, variance_intra_class, variance_inter_class = ratio(model_name, train_test=train_test, cat=cat)
    # print('ratio:', ratio_variance)
    # print('Variance intra class:', np.mean(variance_intra_class, axis=1))
    # print('Variance intra class averaged:', np.mean(np.mean(variance_intra_class, axis=1)))

    # score = correlation_filters(net, model_name, train_test=train_test, ch=nc, vis_filters=False, plot_fig=True,
    #                             cat=cat)
    # score_corr_class = dispersion_classes(model_name, train_test=train_test, plot_fig=True, cat=cat)
    # ratio_variance = 'Nan'
    # plot_2d_projection_z_struct(nb_class, model_name, train_test=train_test, ratio=ratio_variance)

    # plot_acc_bit_noised_per_class(model_name,
    #                               train_test=train_test,
    #                               plot_all_classes=False,
    #                               plot_prediction_mean_std=False,
    #                               plot_score_global=False,
    #                               plot_each_class_score=False,
    #                               plot_each_class_prediction_mean_std=False,
    #                               plot_loss=True,
    #                               separate_plot_class=False,
    #                               cat=cat)

    # compute distance matrix er class:

    # _ = distance_matrix(net, model_name, train_test=train_test, plot_fig=True)

    # Plot resume:
    # compute_z_struct(net, model_name, loader, train_test=train_test, net_type=net_type)
    # get_z_struct_per_class(model_name, train_test=train_test, nb_class=nb_class)
    # get_average_z_struct_per_classes(exp_name=model_name, train_test=train_test)
    # plot_resume(net, model_name, is_ratio, is_distance_loss, loss_distance_mean, loader, train_loader,
    #             device, cat=cat, train_test=train_test, path_scores=path_scores, diff_var=diff_var_loss,
    #             contrastive_loss=contrastive_loss)

    # see percentage of same binary code for encoder struct:
    same_binary_code(net, model_name, loader, nb_class)

    return


def visualize_regions_of_interest(exp_name, net, net_type=None):
    print(exp_name, 'visualize regions of interrest')
    path = 'checkpoints_CNN/'
    net_trained, _, nb_epochs = get_checkpoints(net, path, exp_name)
    loader = test_loader
    print(net_trained)

    net_trained.eval()

    visualize_regions(exp_name, net_trained, len_img_h, len_img_w, loader, plot_activation_value=True,
                      plot_correlation_regions=True, percentage=1)

    random_index = False  # select one random index just for see a random regions for a random image
    choice_label = True  # Choose a specific label to see images of this label
    label = 5  # choice of the label to see regions
    same_im = True  # to see first image of any label
    nb_im = 1  # if is an integer > 0 so select nb_im images else if in a float between 0 and 1, select nb_im % images
    best_region = False  # if we choose best region, we select nb_img who active the most each filter independently
    worst_regions = False  # if we choose worst region, we select nb_img who active the less each filter independently
    any_label = False  # if we don't want to choose any specific label
    average_result = False  # if we want to see all regions extracted averaged by filter
    plot_activation_value = True  # if we plot diagram with bar to see activation value in order to see the most active
    plot_correlation_regions = True  # if plot correlation between region extracted to see redundancy

    # viz_region_im(exp_name,
    #               net_trained,
    #               random_index=random_index,
    #               choice_label=choice_label,
    #               label=label,
    #               nb_im=nb_im,
    #               best_region=best_region,
    #               worst_regions=worst_regions,
    #               any_label=any_label,
    #               average_result=average_result,
    #               plot_activation_value=plot_activation_value,
    #               plot_correlation_regions=plot_correlation_regions,
    #               same_im=same_im,
    #               net_type=net_type)
    return


# parameters:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_test = torch.load('data/batch_mnist.pt')
batch_size = 128  # 10000 to run regions visualization (we must have only one so all data set test in one batch)
train_loader, test_loader = get_mnist_dataset(batch_size=batch_size)

# parameters:
img_size = (1, 32, 32)
nb_class = 10
nb_samples = 10
size = 8
nc = 1
four_conv = False
save = True
L3_without_random = False
is_binary_structural_latent = False
stride = True
len_img_h = img_size[-1]
len_img_w = img_size[-2]
images = batch_test
padding = 0
# for traversal real image:
indx_image = 0

path_select_model_analyse_20 = 'list_model_selected/criterion_select_for_analyse_20.npy'
if os.path.exists(path_select_model_analyse_20):
    selected_analyse_20 = np.load(path_select_model_analyse_20)
path_select_model_analyse_50 = 'list_model_selected/criterion_select_for_analyse_50.npy'
if os.path.exists(path_select_model_analyse_50):
    selected_analyse_50 = np.load(path_select_model_analyse_50)

if __name__ == '__main__':
    list_encoder_struct = ['mnist_struct_baseline_scheduler_binary_1_5_test_wt_sfmax']
                           # 'mnist_struct_baseline_scheduler_binary_1_3']
                           # 'mnist_struct_baseline_scheduler_binary_1_5']
                           # 'mnist_struct_baseline_scheduler_binary_1_6',
                           # 'mnist_struct_baseline_scheduler_binary_1_7',
                           # 'mnist_struct_baseline_scheduler_binary_1_8',
                           # 'mnist_struct_baseline_scheduler_binary_1_9',
                           # 'mnist_struct_baseline_scheduler_binary_1_10',
                           # 'mnist_struct_baseline_scheduler_binary_1_11',
                           # 'mnist_struct_baseline_scheduler_binary_1_12',
                           # 'mnist_struct_min_scheduler_binary_1_10',
                           # 'mnist_struct_mean_scheduler_binary_1_3',
                           # 'mnist_struct_mean_scheduler_binary_1_5',
                           # 'mnist_struct_mean_scheduler_binary_1_6',
                           # 'mnist_struct_mean_scheduler_binary_1_8',
                           # 'mnist_struct_mean_scheduler_binary_1_9',
                           # 'mnist_struct_mean_scheduler_binary_1_10',
                           # 'mnist_struct_mean_scheduler_binary_1_11',
                           # 'mnist_struct_mean_scheduler_binary_1_12']

    list_exp_VAE_var = ['mnist_vae_var_oa_1']

    parameters_mnist_classifier_BK_ratio = "parameters_combinations/mnist_classifier_ratio.txt"

    run_exp_extraction_and_visualization_custom_BK(list_encoder_struct,
                                                   is_ratio=False,
                                                   is_decoder=False,
                                                   is_VAE=False,
                                                   is_encoder_struct=True)
