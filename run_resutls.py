from dataset.dataset_2 import get_mnist_dataset

from viz.visualizer_CNN import *
from viz.viz_regions import *
from models.encoder_struct import Encoder_struct
from models.vae_var import VAE_var

from models.VAE import VAE

from solver import gpu_config


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

    """
    # _________------------------TEst VAE-----------------_______________________
    # test to debug: TODO: remove test.
    z_struct_layer_num = get_layer_zstruct_num(net) + 1

    print(net.encoder_struct[6].weight[8][12])

    print(z_struct_layer_num)
    print(net.encoder_struct[:z_struct_layer_num])

    # test to debug: TODO: remove test.
    for data, labels in loader:
        # evaluation mode:

        with torch.no_grad():
            input_data = data
        if torch.cuda.is_available():
            input_data = input_data.cuda()

        _, z_struct, _, _, _, _, _, _, _, _ = net(input_data, z_struct_out=True,
                                                  z_struct_layer_num=z_struct_layer_num)

        break
    print(z_struct.shape)
    print(z_struct[:20])
    z_struct_distribution = z_struct.detach().numpy()
    zeros_proportion = (np.count_nonzero(z_struct_distribution == 0, axis=0) * 100.) / len(z_struct_distribution)
    print(zeros_proportion)
    # _________------------------End TEst VAE-----------------_______________________
    """

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
    # same_binary_code(net, model_name, loader, nb_class, train_test=train_test, save=True, Hmg_dist=True)
    # z_struct_code_classes(model_name, nb_class, train_test=train_test)
    # compute_z_struct(net, model_name, loader, train_test=train_test, net_type=net_type)
    # get_z_struct_per_class(model_name, train_test=train_test, nb_class=nb_class)
    # get_average_z_struct_per_classes(exp_name=model_name, train_test=train_test)
    # plot_resume(net, model_name, is_ratio, is_distance_loss, loss_distance_mean, loader, train_loader,
    #             device, cat=cat, train_test=train_test, path_scores=path_scores, diff_var=diff_var_loss,
    #             contrastive_loss=contrastive_loss, encoder_struct=True, Hmg_dst=True, z_struct_size=z_struct_size)

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
