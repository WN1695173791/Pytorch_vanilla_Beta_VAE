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

from solver import gpu_config
import csv


def str2bool(string):
    return string.lower() in ("yes", "true", "t", "1")


# ___________________________________ begin extrraction parameters models _____________________________________________
def run_exp_extraction_and_visualization_custom_BK(list_model, is_ratio=False, is_decoder=False, is_VAE=False,
                                                   is_custom=False, is_encoder_struct=False, other_architecture=False):
    for exp_name in list_model:

        # load csv parameters:
        exp_csv_name = 'args_parser/' + exp_name + '.csv'
        with open(exp_csv_name, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                parameters_dict = row

        # model:
        model_name = parameters_dict['exp_name']

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
        hidden_filters_1 = int(parameters_dict['hidden_filters_layer1'])
        hidden_filters_2 = int(parameters_dict['hidden_filters_layer2'])
        hidden_filters_3 = int(parameters_dict['hidden_filters_layer3'])
        is_VAE = str2bool(parameters_dict['is_VAE'])
        if 'ES_reconstruction' in parameters_dict.keys():
            ES_reconstruction = str2bool(parameters_dict['ES_reconstruction'])
        else:
            ES_reconstruction = False

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
        is_VAE_var = str2bool(parameters_dict['is_VAE_var'])
        var_second_cnn_block = str2bool(parameters_dict['var_second_cnn_block'])
        var_third_cnn_block = str2bool(parameters_dict['var_third_cnn_block'])
        if 'EV_classifier' in parameters_dict.keys():
            EV_classifier = str2bool(parameters_dict['EV_classifier'])
        else:
            EV_classifier = False

        # encoder struct:
        kernel_size_1 = int(parameters_dict['kernel_size_1'])
        kernel_size_2 = int(parameters_dict['kernel_size_2'])
        kernel_size_3 = int(parameters_dict['kernel_size_3'])
        binary_z = str2bool(parameters_dict['binary_z'])
        binary_first_conv = str2bool(parameters_dict['binary_first_conv'])
        binary_second_conv = str2bool(parameters_dict['binary_second_conv'])
        binary_third_conv = str2bool(parameters_dict['binary_third_conv'])

        if is_VAE:
            use_structural_encoder = True
            net = VAE(z_var_size=z_var_size,
                      var_second_cnn_block=var_second_cnn_block,
                      var_third_cnn_block=var_third_cnn_block,
                      other_architecture=other_architecture,
                      z_struct_size=z_struct_size,
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
                      binary_third_conv=binary_third_conv,
                      ES_reconstruction=ES_reconstruction)
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
                          var_third_cnn_block=var_third_cnn_block,
                          other_architecture=other_architecture,
                          EV_classifier=EV_classifier)
        if is_decoder:
            run_decoder(model_name, net)
        elif is_VAE or is_VAE_var:
            run_VAE(model_name, net, lambda_BCE, beta, z_struct_size, z_var_size, use_structural_encoder, is_VAE_var,
                    ES_reconstruction)
        else:
            run_viz_expes(model_name, net, is_ratio, loss_min_distance_cl, loss_distance_mean, cat='Encoder_struct',
                          net_type=net_type, diff_var_loss=diff_var, z_struct_size=z_struct_size)
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


def run_VAE(model_name, net, lambda_BCE, beta, z_struct_size, z_var_size, VAE_struct, is_vae_var, ES_reconstruction):

    path = 'checkpoints_CNN/'
    path_scores = 'checkpoint_scores_CNN'
    net, _, _ = get_checkpoints(net, path, model_name)
    net, device = gpu_config(net)
    print('VAE: _____________------------------{}-----------------_____________________'.format(model_name))
    print(net)

    loader = test_loader
    train_test = 'test'
    batch = batch_test
    embedding_size = z_struct_size + z_var_size
    net.eval()

    # __________-----------Values computing----------_____________
    # save code majoritaire with their percent:
    # same_binary_code(net, model_name, loader, nb_class, train_test=train_test, save=True, Hmg_dist=False, is_VAE=True)
    # z_struct_code_classes(model_name, nb_class, train_test=train_test)

    # Uc bin maj:
    # percent_max = histo_count_uniq_code(model_name, train_test, plot_histo=False, return_percent=True)
    # maj_uc = np.load('binary_encoder_struct_results/uniq_code/uc_maj_class_' + model_name + '_' \
    #                            + train_test + '.npy', allow_pickle=True)

    # first: compute z_struct mean per class:
    # compute_z_struct_mean_VAE(net, model_name, loader, train_test='test', return_results=False)

    # second: get average z_struct per classe:
    # _, average_z_struct_class = get_z_struct_per_class_VAE(model_name, train_test='test', nb_class=nb_class)


    # __________--------------PLOT------------_________________
    # losses:
    # plot_loss_results_VAE(path_scores, model_name, beta, lambda_BCE, save=True)

    # compute and plot real distribution:
    # mu_var, sigma_var, encoder_struct_zeros_proportion = real_distribution_model(net,
    #                                                                      model_name,
    #                                                                      z_struct_size,
    #                                                                      z_var_size,
    #                                                                      loader,
    #                                                                      'test',
    #                                                                      plot_gaussian=False,
    #                                                                      save=True,
    #                                                                      VAE_struct=VAE_struct,
    #                                                                      is_vae_var=is_vae_var)

    # mu_var, sigma_var, encoder_struct_zeros_proportion = 0, 0, 0
    # plot reconstruciton with real distribution sample:
    # viz_reconstruction_VAE(net, loader, model_name, z_var_size, z_struct_size, nb_img=10,
    #                        nb_class=nb_class, save=True, z_reconstruction=True,
    #                        z_struct_reconstruction=False, z_var_reconstruction=False,
    #                        return_scores=False, real_distribution=True, mu_var=mu_var, std_var=sigma_var,
    #                        mu_struct=encoder_struct_zeros_proportion, is_vae_var=is_vae_var)

    # samples with compute scores:
    # images_generation(net,
    #                   model_name,
    #                   batch,
    #                   train_test,
    #                   size=(8, 8),
    #                   mu_var=mu_var,
    #                   es_distribution=encoder_struct_zeros_proportion,
    #                   std_var=sigma_var,
    #                   z_var_size=z_var_size,
    #                   z_struct_size=z_struct_size,
    #                   FID=True,
    #                   IS=True,
    #                   LPIPS=True,
    #                   real_distribution=True,
    #                   save=True,
    #                   use_maj_uc=True)

    # prototype z_struct with z_var traversale latent:
    # for index in range(nb_class):
    # index = 0
    # traversal_latent_prototype(net,
    #                            model_name,
    #                            train_test,
    #                            device,
    #                            z_var_size,
    #                            sigma_var=sigma_var,
    #                            nb_class=nb_class,
    #                            size=8,
    #                            avg_prototype=True,
    #                            maj_uc_prototype=True,
    #                            batch=batch,
    #                            index=index,
    #                            save=True)

    # viz switch image:
    # switch_img(net, model_name, loader, z_var_size)

    # Plot prototype:
    # plot_prototype(net, model_name, train_test, z_var_size, nb_class=nb_class, avg_zstruct=True, maj_uc_prototype=True,
    #                save=True)

    # plot resume VAE:
    # plot_VAE_resume(net, model_name, z_struct_size, z_var_size, loader, VAE_struct, is_vae_var, train_test, save=True,
    #                 nb_class=nb_class, nb_img=8, std_var=sigma_var, mu_var=mu_var,
    #                 mu_struct=encoder_struct_zeros_proportion, index=0)

    # ----------------------------------------------------------------------------------------------------------
    # Test vae var classfiier hypothesis: ----------------------------------------------------------------------
    # replace weigths value:
    # print(net.var_classifier[0].weight)
    # x = net.var_classifier[0].weight
    # y = torch.zeros(net.var_classifier[0].weight.shape)
    # net.var_classifier[0].weight = torch.nn.Parameter(torch.where((torch.abs(net.var_classifier[0].weight) < 5.), x, y))
    # print(net.var_classifier[0].weight)

    # TODO: 1) Check classification accuracy on train and test set:
    score_test = VAE_var_classifier_score(net, loader)
    print(score_test)  # 92%

    # for VAE var with decoder trained:
    mu_var, sigma_var, encoder_struct_zeros_proportion = 0, 0, 0
    viz_reconstruction_VAE(net, loader, model_name, z_var_size, z_struct_size, nb_img=10,
                           nb_class=nb_class, save=True, z_reconstruction=True,
                           z_struct_reconstruction=False, z_var_reconstruction=False,
                           return_scores=False, real_distribution=True, mu_var=mu_var, std_var=sigma_var,
                           mu_struct=encoder_struct_zeros_proportion, is_vae_var=is_vae_var)




    # ----------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------

    net.train()
    return


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def run_viz_expes(model_name, net, is_ratio, is_distance_loss, loss_distance_mean, net_type=None, cat=None,
                  ratio_reg=False, diff_var_loss=False, contrastive_loss=False, z_struct_size=5):

    path = 'checkpoints_CNN/'
    path_scores = 'checkpoint_scores_CNN'
    net, _, nb_epochs = get_checkpoints(net, path, model_name)
    net, device = gpu_config(net)

    print('________________------------------{}-----------------_____________________'.format(model_name))
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
    #                             False, False, False, False, False, False, False, False)
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
    same_binary_code(net, model_name, loader, nb_class, train_test=train_test, save=True, Hmg_dist=True)
    z_struct_code_classes(model_name, nb_class, train_test=train_test)
    compute_z_struct(net, model_name, loader, train_test=train_test, net_type=net_type)
    get_z_struct_per_class(model_name, train_test=train_test, nb_class=nb_class)
    get_average_z_struct_per_classes(exp_name=model_name, train_test=train_test)
    plot_resume(net, model_name, is_ratio, is_distance_loss, loss_distance_mean, loader, train_loader,
                device, cat=cat, train_test=train_test, path_scores=path_scores, diff_var=diff_var_loss,
                contrastive_loss=contrastive_loss, encoder_struct=True, Hmg_dst=True, z_struct_size=z_struct_size)

    # receptive_field = get_receptive_field_size(net, batch_test)
    # _ = score_with_best_code_uniq(net, model_name, train_test, loader, z_struct_size, loader_size)

    # _ = histo_count_uniq_code(model_name, train_test, plot_histo=True, return_percent=True)

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

    list_encoder_struct = ['mnist_struct_baseline_scheduler_binary_10']
                           # 'mnist_struct_baseline_scheduler_binary_15',
                           # 'mnist_struct_baseline_scheduler_binary_20',
                           # 'mnist_struct_baseline_scheduler_binary_25',
                           # 'mnist_struct_baseline_scheduler_binary_30']

    list_encoder_struct_Hmg = ['mnist_struct_baseline_scheduler_binary_10_Hmg_dst_1',
                               'mnist_struct_baseline_scheduler_binary_15_Hmg_dst_1',
                               'mnist_struct_baseline_scheduler_binary_20_Hmg_dst_1',
                               'mnist_struct_baseline_scheduler_binary_25_Hmg_dst_1',
                               'mnist_struct_baseline_scheduler_binary_30_Hmg_dst_1',
                               'mnist_struct_baseline_scheduler_binary_10_target_uc_1',
                               'mnist_struct_baseline_scheduler_binary_15_target_uc_1',
                               'mnist_struct_baseline_scheduler_binary_20_target_uc_1',
                               'mnist_struct_baseline_scheduler_binary_25_target_uc_1',
                               'mnist_struct_baseline_scheduler_binary_30_target_uc_1',
                               'mnist_struct_baseline_scheduler_binary_10_Hmg_dst_2',
                               'mnist_struct_baseline_scheduler_binary_15_Hmg_dst_2',
                               'mnist_struct_baseline_scheduler_binary_20_Hmg_dst_2',
                               'mnist_struct_baseline_scheduler_binary_25_Hmg_dst_2',
                               'mnist_struct_baseline_scheduler_binary_30_Hmg_dst_2',
                               'mnist_struct_baseline_scheduler_binary_10_target_uc_2',
                               'mnist_struct_baseline_scheduler_binary_15_target_uc_2',
                               'mnist_struct_baseline_scheduler_binary_20_target_uc_2',
                               'mnist_struct_baseline_scheduler_binary_25_target_uc_2',
                               'mnist_struct_baseline_scheduler_binary_30_target_uc_2',
                               'mnist_struct_baseline_scheduler_binary_10_Hmg_dst_3',
                               'mnist_struct_baseline_scheduler_binary_15_Hmg_dst_3',
                               'mnist_struct_baseline_scheduler_binary_20_Hmg_dst_3',
                               'mnist_struct_baseline_scheduler_binary_25_Hmg_dst_3',
                               'mnist_struct_baseline_scheduler_binary_30_Hmg_dst_3',
                               'mnist_struct_baseline_scheduler_binary_10_target_uc_3',
                               'mnist_struct_baseline_scheduler_binary_15_target_uc_3',
                               'mnist_struct_baseline_scheduler_binary_20_target_uc_3',
                               'mnist_struct_baseline_scheduler_binary_25_target_uc_3',
                               'mnist_struct_baseline_scheduler_binary_30_target_uc_3']

    list_exp_VAE_var = ['mnist_vae_var_2cb_15_grad_inv_PT',
                        'mnist_vae_var_2cb_15_grad_inv_FS']
                        # 'mnist_vae_var_2cb_5',
                        # 'mnist_vae_var_2cb_10',
                        # 'mnist_vae_var_2cb_15',
                        # 'mnist_vae_var_2cb_20',
                        # 'mnist_vae_var_2cb_25',
                        # 'mnist_vae_var_2cb_30']

    list_exp_VAE = ['mnist_VAE_s10_v5_SE_beta_3',
                    'mnist_VAE_s15_v5_SE_beta_3',
                    'mnist_VAE_s20_v5_SE_beta_3',
                    'mnist_VAE_s25_v5_SE_beta_3',
                    'mnist_VAE_s30_v5_SE_beta_3',
                    'mnist_VAE_s10_v10_SE_beta_3',
                    'mnist_VAE_s15_v10_SE_beta_3',
                    'mnist_VAE_s20_v10_SE_beta_3',
                    'mnist_VAE_s25_v10_SE_beta_3',
                    'mnist_VAE_s30_v10_SE_beta_3',
                    'mnist_VAE_s10_v15_SE_beta_3',
                    'mnist_VAE_s15_v15_SE_beta_3',
                    'mnist_VAE_s20_v15_SE_beta_3',
                    'mnist_VAE_s25_v15_SE_beta_3',
                    'mnist_VAE_s30_v15_SE_beta_3',
                    'mnist_VAE_s10_v20_SE_beta_3',
                    'mnist_VAE_s15_v20_SE_beta_3',
                    'mnist_VAE_s20_v20_SE_beta_3',
                    'mnist_VAE_s25_v20_SE_beta_3',
                    'mnist_VAE_s30_v20_SE_beta_3',
                    'mnist_VAE_s10_v25_SE_beta_3',
                    'mnist_VAE_s15_v25_SE_beta_3',
                    'mnist_VAE_s20_v25_SE_beta_3',
                    'mnist_VAE_s25_v25_SE_beta_3',
                    'mnist_VAE_s30_v25_SE_beta_3',
                    'mnist_VAE_s10_v30_SE_beta_3',
                    'mnist_VAE_s15_v30_SE_beta_3',
                    'mnist_VAE_s20_v30_SE_beta_3',
                    'mnist_VAE_s25_v30_SE_beta_3',
                    'mnist_VAE_s30_v30_SE_beta_3']
                    # 'mnist_VAE_s10_v5_Hmg_dst_SE',
                    # 'mnist_VAE_s15_v5_Hmg_dst_SE',
                    # 'mnist_VAE_s20_v5_Hmg_dst_SE',
                    # 'mnist_VAE_s25_v5_Hmg_dst_SE',
                    # 'mnist_VAE_s30_v5_Hmg_dst_SE',
                    # 'mnist_VAE_s10_v10_Hmg_dst_SE',
                    # 'mnist_VAE_s15_v10_Hmg_dst_SE',
                    # 'mnist_VAE_s20_v10_Hmg_dst_SE',
                    # 'mnist_VAE_s25_v10_Hmg_dst_SE',
                    # 'mnist_VAE_s30_v10_Hmg_dst_SE',
                    # 'mnist_VAE_s10_v15_Hmg_dst_SE',
                    # 'mnist_VAE_s15_v15_Hmg_dst_SE',
                    # 'mnist_VAE_s20_v15_Hmg_dst_SE',
                    # 'mnist_VAE_s25_v15_Hmg_dst_SE',
                    # 'mnist_VAE_s30_v15_Hmg_dst_SE',
                    # 'mnist_VAE_s10_v20_Hmg_dst_SE',
                    # 'mnist_VAE_s15_v20_Hmg_dst_SE',
                    # 'mnist_VAE_s20_v20_Hmg_dst_SE',
                    # 'mnist_VAE_s25_v20_Hmg_dst_SE',
                    # 'mnist_VAE_s30_v20_Hmg_dst_SE',
                    # 'mnist_VAE_s10_v25_Hmg_dst_SE',
                    # 'mnist_VAE_s15_v25_Hmg_dst_SE',
                    # 'mnist_VAE_s20_v25_Hmg_dst_SE',
                    # 'mnist_VAE_s25_v25_Hmg_dst_SE',
                    # 'mnist_VAE_s30_v25_Hmg_dst_SE',
                    # 'mnist_VAE_s10_v30_Hmg_dst_SE',
                    # 'mnist_VAE_s15_v30_Hmg_dst_SE',
                    # 'mnist_VAE_s20_v30_Hmg_dst_SE',
                    # 'mnist_VAE_s25_v30_Hmg_dst_SE',
                    # 'mnist_VAE_s30_v30_Hmg_dst_SE']
                    # 'mnist_VAE_s10_v5_BF',
                    # 'mnist_VAE_s10_v5_PT',
                    # 'mnist_VAE_s10_v5_SE',
                    # 'mnist_VAE_s10_v5_FS',
                    # 'mnist_VAE_s15_v5_BF',
                    # 'mnist_VAE_s20_v5_BF',
                    # 'mnist_VAE_s25_v5_BF',
                    # 'mnist_VAE_s30_v5_BF',
                    # 'mnist_VAE_s10_v10_BF',
                    # 'mnist_VAE_s15_v10_BF',
                    # 'mnist_VAE_s20_v10_BF',
                    # 'mnist_VAE_s25_v10_BF',
                    # 'mnist_VAE_s30_v10_BF',
                    # 'mnist_VAE_s10_v15_BF',
                    # 'mnist_VAE_s15_v15_BF',
                    # 'mnist_VAE_s20_v15_BF',
                    # 'mnist_VAE_s25_v15_BF',
                    # 'mnist_VAE_s30_v15_BF',
                    # 'mnist_VAE_s10_v20_BF',
                    # 'mnist_VAE_s15_v20_BF',
                    # 'mnist_VAE_s20_v20_BF',
                    # 'mnist_VAE_s25_v20_BF',
                    # 'mnist_VAE_s30_v20_BF',
                    # 'mnist_VAE_s10_v25_BF',
                    # 'mnist_VAE_s15_v25_BF',
                    # 'mnist_VAE_s20_v25_BF',
                    # 'mnist_VAE_s25_v25_BF',
                    # 'mnist_VAE_s30_v25_BF',
                    # 'mnist_VAE_s10_v30_BF',
                    # 'mnist_VAE_s15_v30_BF',
                    # 'mnist_VAE_s20_v30_BF',
                    # 'mnist_VAE_s25_v30_BF',
                    # 'mnist_VAE_s30_v30_BF',
                    # 'mnist_VAE_s10_v5_PT',
                    # 'mnist_VAE_s15_v5_PT',
                    # 'mnist_VAE_s20_v5_PT',
                    # 'mnist_VAE_s25_v5_PT',
                    # 'mnist_VAE_s30_v5_PT',
                    # 'mnist_VAE_s10_v10_PT',
                    # 'mnist_VAE_s15_v10_PT',
                    # 'mnist_VAE_s20_v10_PT',
                    # 'mnist_VAE_s25_v10_PT',
                    # 'mnist_VAE_s30_v10_PT',
                    # 'mnist_VAE_s10_v15_PT',
                    # 'mnist_VAE_s15_v15_PT',
                    # 'mnist_VAE_s20_v15_PT',
                    # 'mnist_VAE_s25_v15_PT',
                    # 'mnist_VAE_s30_v15_PT',
                    # 'mnist_VAE_s10_v20_PT',
                    # 'mnist_VAE_s15_v20_PT',
                    # 'mnist_VAE_s20_v20_PT',
                    # 'mnist_VAE_s25_v20_PT',
                    # 'mnist_VAE_s30_v20_PT',
                    # 'mnist_VAE_s10_v25_PT',
                    # 'mnist_VAE_s15_v25_PT',
                    # 'mnist_VAE_s20_v25_PT',
                    # 'mnist_VAE_s25_v25_PT',
                    # 'mnist_VAE_s30_v25_PT',
                    # 'mnist_VAE_s10_v30_PT',
                    # 'mnist_VAE_s15_v30_PT',
                    # 'mnist_VAE_s20_v30_PT',
                    # 'mnist_VAE_s25_v30_PT',
                    # 'mnist_VAE_s30_v30_PT',
                    # 'mnist_VAE_s10_v5_SE',
                    # 'mnist_VAE_s15_v5_SE',
                    # 'mnist_VAE_s20_v5_SE',
                    # 'mnist_VAE_s25_v5_SE',
                    # 'mnist_VAE_s30_v5_SE',
                    # 'mnist_VAE_s10_v10_SE',
                    # 'mnist_VAE_s15_v10_SE',
                    # 'mnist_VAE_s20_v10_SE',
                    # 'mnist_VAE_s25_v10_SE',
                    # 'mnist_VAE_s30_v10_SE',
                    # 'mnist_VAE_s10_v15_SE',
                    # 'mnist_VAE_s15_v15_SE',
                    # 'mnist_VAE_s20_v15_SE',
                    # 'mnist_VAE_s25_v15_SE',
                    # 'mnist_VAE_s30_v15_SE',
                    # 'mnist_VAE_s10_v20_SE',
                    # 'mnist_VAE_s15_v20_SE',
                    # 'mnist_VAE_s20_v20_SE',
                    # 'mnist_VAE_s25_v20_SE',
                    # 'mnist_VAE_s30_v20_SE',
                    # 'mnist_VAE_s10_v25_SE',
                    # 'mnist_VAE_s15_v25_SE',
                    # 'mnist_VAE_s20_v25_SE',
                    # 'mnist_VAE_s25_v25_SE',
                    # 'mnist_VAE_s30_v25_SE',
                    # 'mnist_VAE_s10_v30_SE',
                    # 'mnist_VAE_s15_v30_SE',
                    # 'mnist_VAE_s20_v30_SE',
                    # 'mnist_VAE_s25_v30_SE',
                    # 'mnist_VAE_s30_v30_SE',
                    # 'mnist_VAE_s10_v5_FS',
                    # 'mnist_VAE_s15_v5_FS',
                    # 'mnist_VAE_s20_v5_FS',
                    # 'mnist_VAE_s25_v5_FS',
                    # 'mnist_VAE_s30_v5_FS',
                    # 'mnist_VAE_s10_v10_FS',
                    # 'mnist_VAE_s15_v10_FS',
                    # 'mnist_VAE_s20_v10_FS',
                    # 'mnist_VAE_s25_v10_FS',
                    # 'mnist_VAE_s30_v10_FS',
                    # 'mnist_VAE_s10_v15_FS',
                    # 'mnist_VAE_s15_v15_FS',
                    # 'mnist_VAE_s20_v15_FS',
                    # 'mnist_VAE_s25_v15_FS',
                    # 'mnist_VAE_s30_v15_FS',
                    # 'mnist_VAE_s10_v20_FS',
                    # 'mnist_VAE_s15_v20_FS',
                    # 'mnist_VAE_s20_v20_FS',
                    # 'mnist_VAE_s25_v20_FS',
                    # 'mnist_VAE_s30_v20_FS',
                    # 'mnist_VAE_s10_v25_FS',
                    # 'mnist_VAE_s15_v25_FS',
                    # 'mnist_VAE_s20_v25_FS',
                    # 'mnist_VAE_s25_v25_FS',
                    # 'mnist_VAE_s30_v25_FS',
                    # 'mnist_VAE_s10_v30_FS',
                    # 'mnist_VAE_s15_v30_FS',
                    # 'mnist_VAE_s20_v30_FS',
                    # 'mnist_VAE_s25_v30_FS',
                    # 'mnist_VAE_s30_v30_FS']

    list_ES_reconstruction = ['mnist_encoder_struct_reconstruction_s10_PT_2',
                              'mnist_encoder_struct_reconstruction_s10_FS_2',
                              'mnist_encoder_struct_reconstruction_s15_PT_2',
                              'mnist_encoder_struct_reconstruction_s15_FS_2',
                              'mnist_encoder_struct_reconstruction_s20_PT_2',
                              'mnist_encoder_struct_reconstruction_s20_FS_2',
                              'mnist_encoder_struct_reconstruction_s25_PT_2',
                              'mnist_encoder_struct_reconstruction_s25_FS_2',
                              'mnist_encoder_struct_reconstruction_s30_PT_2',
                              'mnist_encoder_struct_reconstruction_s30_FS_2',
                              'mnist_encoder_struct_reconstruction_s10_Hmg_dst_1_PT',
                              'mnist_encoder_struct_reconstruction_s10_Hmg_dst_2_PT',
                              'mnist_encoder_struct_reconstruction_s10_Hmg_dst_3_PT',
                              'mnist_encoder_struct_reconstruction_s15_Hmg_dst_1_PT',
                              'mnist_encoder_struct_reconstruction_s15_Hmg_dst_2_PT',
                              'mnist_encoder_struct_reconstruction_s15_Hmg_dst_3_PT',
                              'mnist_encoder_struct_reconstruction_s20_Hmg_dst_1_PT',
                              'mnist_encoder_struct_reconstruction_s20_Hmg_dst_2_PT',
                              'mnist_encoder_struct_reconstruction_s20_Hmg_dst_3_PT',
                              'mnist_encoder_struct_reconstruction_s25_Hmg_dst_1_PT',
                              'mnist_encoder_struct_reconstruction_s25_Hmg_dst_2_PT',
                              'mnist_encoder_struct_reconstruction_s25_Hmg_dst_3_PT',
                              'mnist_encoder_struct_reconstruction_s30_Hmg_dst_1_PT',
                              'mnist_encoder_struct_reconstruction_s30_Hmg_dst_2_PT',
                              'mnist_encoder_struct_reconstruction_s30_Hmg_dst_3_PT']

    list_VAE_var_classifier = ['mnist_vae_var_2cb_15_classifier_grad_inv_PT',
                               'mnist_vae_var_2cb_15_classifier_grad_inv_FS']
                               # 'mnist_vae_var_2cb_5_classifier']
                               # 'mnist_vae_var_2cb_10_classifier',
                               # 'mnist_vae_var_2cb_15_classifier',
                               # 'mnist_vae_var_2cb_20_classifier',
                               # 'mnist_vae_var_2cb_25_classifier',
                               # 'mnist_vae_var_2cb_30_classifier']

    parameters_mnist_classifier_BK_ratio = "parameters_combinations/mnist_classifier_ratio.txt"

    # run_exp_extraction_and_visualization_custom_BK(list_encoder_struct,
    #                                                is_ratio=False,
    #                                                is_decoder=False,
    #                                                is_VAE=False,
    #                                                is_encoder_struct=True)

    # run_exp_extraction_and_visualization_custom_BK(list_encoder_struct_Hmg,
    #                                                is_ratio=False,
    #                                                is_decoder=False,
    #                                                is_VAE=False,
    #                                                is_encoder_struct=True)

    # run_exp_extraction_and_visualization_custom_BK(list_exp_VAE,
    #                                                is_ratio=False,
    #                                                is_decoder=False,
    #                                                is_VAE=True,
    #                                                is_encoder_struct=False)

    # run_exp_extraction_and_visualization_custom_BK(list_ES_reconstruction,
    #                                                is_ratio=False,
    #                                                is_decoder=False,
    #                                                is_VAE=True,
    #                                                is_encoder_struct=False)

    run_exp_extraction_and_visualization_custom_BK(list_VAE_var_classifier,
                                                   is_ratio=False,
                                                   is_decoder=False,
                                                   is_VAE=True,
                                                   is_encoder_struct=False)

    run_exp_extraction_and_visualization_custom_BK(list_exp_VAE_var,
                                                   is_ratio=False,
                                                   is_decoder=False,
                                                   is_VAE=False,
                                                   is_encoder_struct=False)


