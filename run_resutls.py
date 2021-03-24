from models.custom_CNN_BK import Custom_CNN_BK
from models.custom_CNN import Custom_CNN
from dataset.dataset_2 import get_dataloaders, get_mnist_dataset

from visualizer import *
from visualizer_CNN import *
from viz.viz_regions import *
from scores_classifier import compute_scores
from models.Encoder_decoder import Encoder_decoder
from models.VAE import VAE
from visualizer import viz_reconstruction
from viz.visualize import Visualizer
import torch.nn.functional as F


# ___________________________________ begin extrraction parameters models ______________________________________________
def run_exp_extraction_and_visualization_custom(path_parameter, line_begin, line_end, list_model):
    # Read mnist_expes:
    file = open(path_parameter, "r")
    # 24 arguments to recuperate
    arguments_1 = {}
    list_model_selected_Custom_CNN_95_acc_z_struct_5 = []

    path_save_95_acc_zstruct_5 = 'list_model_selected/list_model_selected_Custom_CNN_95_acc_z_struct_5.txt'
    if os.path.exists(path_save_95_acc_zstruct_5):
        with open(path_save_95_acc_zstruct_5, 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                currentPlace = line[:-1]
                # add item to the list
                list_model_selected_Custom_CNN_95_acc_z_struct_5.append(currentPlace)

    index = 0
    key = 0
    for line in file:
        index += 1
        if index < line_begin:
            pass
        elif line[0].isspace() or index > line_end:
            break
        elif line[0] != "-":
            pass
        else:
            key += 1
            arguments_1[key] = []
            current_line = line.split("--")
            for l in range(len(current_line)):
                arg = current_line[l].split(' ')
                arguments_1[key].append(arg)

    for key in arguments_1:

        args = arguments_1[key]
        exp_name = args[23][-1].split('\n')[0]
        z_struct_size = int(args[11][1])
        stride_size = int(args[19][1])
        classif_layer_size = int(args[13][1])
        hidden_filters_1 = int(args[20][1])
        hidden_filters_2 = int(args[21][1])
        hidden_filters_3 = int(args[22][1])
        if args[17][1] == 'True':
            two_conv_layer = True
        elif args[17][1] == 'False':
            two_conv_layer = False
        if args[18][1] == 'True':
            three_conv_layer = True
        elif args[18][1] == 'False':
            three_conv_layer = False
        if args[12][1] == 'True':
            add_classification_layer = True
        elif args[12][1] == 'False':
            add_classification_layer = False

        net = Custom_CNN(z_struct_size=z_struct_size,
                         stride_size=stride_size,
                         classif_layer_size=classif_layer_size,
                         add_classification_layer=add_classification_layer,
                         hidden_filters_1=hidden_filters_1,
                         hidden_filters_2=hidden_filters_2,
                         hidden_filters_3=hidden_filters_3,
                         two_conv_layer=two_conv_layer,
                         three_conv_layer=three_conv_layer)

        # if exp_name in list_model_selected_Custom_CNN_95_acc_z_struct_5:
        #     run_viz_expes(exp_name, net, net_type='Custom_CNN', cat='zstruct_5')
        # if exp_name in list_model_selected_Custom_CNN_95_acc_z_struct_10:
        #     run_viz_expes(exp_name, net, net_type='Custom_CNN', cat='zstruct_10')
        # if exp_name in list_model_selected_Custom_CNN_95_acc_z_struct_15:
        #     run_viz_expes(exp_name, net, net_type='Custom_CNN', cat='zstruct_15')
        # if exp_name in list_model_selected_Custom_CNN_95_acc_z_struct_20:
        #     run_viz_expes(exp_name, net, net_type='Custom_CNN', cat='zstruct_20')
        # if exp_name in list_model_selected_Custom_CNN_95_acc_z_struct_30:
        #     run_viz_expes(exp_name, net, net_type='Custom_CNN', cat='zstruct_30')
        # if exp_name in list_model_selected_Custom_CNN_95_acc_z_struct_50:
        #     run_viz_expes(exp_name, net, net_type='Custom_CNN', cat='zstruct_50')

        # if exp_name in selected_analyse_20:
        #     run_viz_expes(exp_name, net, net_type='Custom_CNN', cat='zstruct_20')
        # if exp_name in selected_analyse_50:
        #     run_viz_expes(exp_name, net, net_type='Custom_CNN', cat='zstruct_50')

        if exp_name in list_model:
            run_viz_expes(exp_name, net, net_type='Custom_CNN', cat='zstruct_20')
            visualize_regions_of_interest(exp_name, net, net_type='Custom_CNN')

        # # save list models:
        # if z_struct_size == 5:
        #     if not os.path.exists(path_save_95_acc_zstruct_5):
        #         if add_classification_layer:
        #             model_name = selection(net, exp_name)
        #             if model_name is not None:
        #                 list_model_selected_Custom_CNN_95_acc_z_struct_5.append(model_name)

    # if not os.path.exists(path_save_95_acc_zstruct_5):
    #     # save list models:
    #     with open(path_save_95_acc_zstruct_5, 'w') as filehandle:
    #         for listitem in list_model_selected_Custom_CNN_95_acc_z_struct_5:
    #             filehandle.write('%s\n' % listitem)

    file.close()


def run_exp_extraction_and_visualization_custom_BK(path_parameter, line_begin, line_end, list_model, is_ratio=False,
                                                   is_decoder=False, is_VAE=False):
    file = open(path_parameter, "r")
    # 28 arguments to recuperate
    arguments_2 = {}
    list_model_selected_Custom_CNN_BK_95_acc_z_struct_5 = []
    path_save = 'list_model_selected/list_model_selected_Custom_CNN_BK_95_acc.txt'

    path_save_BK_95_acc_zstruct_5 = 'list_model_selected/list_model_selected_Custom_CNN_BK_95_acc_z_struct_5.txt'
    if os.path.exists(path_save_BK_95_acc_zstruct_5):
        with open(path_save_BK_95_acc_zstruct_5, 'r') as filehandle:
            for line in filehandle:
                # remove linebreak which is the last character of the string
                currentPlace = line[:-1]
                # add item to the list
                list_model_selected_Custom_CNN_BK_95_acc_z_struct_5.append(currentPlace)

    index = 0
    key = 0
    for line in file:
        index += 1
        if index < line_begin:
            pass
        elif line[0].isspace() or index > line_end:
            break
        elif line[0] != "-":
            pass
        else:
            key += 1
            arguments_2[key] = []
            current_line = line.split("--")
            for l in range(len(current_line)):
                arg = current_line[l].split(' ')
                arguments_2[key].append(arg)

    for key in arguments_2:

        args = arguments_2[key]
        batch_size = args[4]

        Binary_z = False
        binary_chain = False
        exp_name = args[-1][-1].split('\n')[0]

        # VAE:
        if is_VAE:
            # encoder struct:
            z_struct_size = 32
            big_kernel_size = 8
            stride_size = 1
            classif_layer_size = 30
            add_classification_layer = True
            hidden_filters_1 = 32
            hidden_filters_2 = 32
            hidden_filters_3 = 32
            BK_in_first_layer = True
            two_conv_layer = True
            three_conv_layer = False
            BK_in_second_layer = False
            BK_in_third_layer = False
            add_linear_after_GMP = True
            # VAE:
            z_var_size = int(args[-15][1])
            lambda_BCE = float(args[-14][1])
            beta = float(args[-13][1])
            var_hidden_filters_1 = int(args[-12][1])
            var_hidden_filters_2 = int(args[-11][1])
            var_hidden_filters_3 = int(args[-10][1])
            var_kernel_size_1 = int(args[-9][1])
            var_kernel_size_2 = int(args[-8][1])
            var_kernel_size_3 = int(args[-7][1])
            var_stride_size_1 = int(args[-6][1])
            var_stride_size_2 = int(args[-5][1])
            var_stride_size_3 = int(args[-4][1])
            var_hidden_dim = int(args[-3][1])
            if args[-2][1] == 'True':
                var_three_conv_layer = True
            elif args[-2][1] == 'False':
                var_three_conv_layer = False
        else:
            # decoder:
            if is_decoder:
                decoder_first_dense = int(args[-11][1])
                decoder_n_filter_1 = int(args[-10][1])
                decoder_n_filter_2 = int(args[-9][1])
                decoder_kernel_size_1 = int(args[-8][1])
                decoder_kernel_size_2 = int(args[-7][1])
                decoder_kernel_size_3 = int(args[-6][1])
                decoder_stride_1 = int(args[-5][1])
                decoder_stride_2 = int(args[-4][1])
                decoder_stride_3 = int(args[-3][1])

            if args[-2][0] == 'binary_chain':
                if args[-2][1] == 'True':
                    binary_chain = True
                elif args[-2][1] == 'False':
                    binary_chain = False
            if args[-2][0] == 'binary_z':
                if args[-2][1] == 'True':
                    Binary_z = True
                elif args[-2][1] == 'False':
                    Binary_z = False
            else:
                if args[-3][0] == 'binary_z':
                    if args[-3][1] == 'True':
                        Binary_z = True
                    elif args[-3][1] == 'False':
                        Binary_z = False

            # diff var loss:
            if args[-4][0] == 'diff_var':
                if args[-4][1] == 'True':
                    diff_var = True
                elif args[-4][1] == 'False':
                    diff_var = False

            if args[-3][0] == 'loss_distance_mean':
                if args[-3][1] == 'True':
                    loss_distance_mean = True
                elif args[-3][1] == 'False':
                    loss_distance_mean = False
            else:
                loss_distance_mean = False
            if is_ratio:
                ratio_reg = True
                lambda_ratio = args[28][1]
                lambda_class = args[29][1]
                if args[31][0] == 'alpha':
                    alpha = args[31][0]
                    loss = args[32][0]
                    optimizer = args[33][0]
                    mrg = args[34][0]
                    IPC = args[35][0]
                    warm = args[36][0]
                    sz_embedding = args[37][0]
                    add_linear_after_GMP = True
                elif args[30][0] == 'other_ratio':
                    if args[30][1] == 'True':
                        other_ratio = True
                    elif args[30][1] == 'False':
                        other_ratio = False
                    if args[31][0] == 'loss_min_distance_cl':
                        if args[31][1] == 'True':
                            loss_min_distance_cl = True
                        elif args[31][1] == 'False':
                            loss_min_distance_cl = False
                        lambda_var_distance = args[32][1]
                        intra_class_variance_loss = args[33][1]
                        lambda_intra_class_var = args[34][1]

                        if args[-2][0] == 'binary_z':
                            if args[-2][1] == 'True':
                                Binary_z = True
                            elif args[-2][1] == 'False':
                                Binary_z = False
                    add_linear_after_GMP = True
                elif args[30][0] == 'add_linear_after_GMP':
                    if args[30][1] == 'True':
                        add_linear_after_GMP = True
                    elif args[30][1] == 'False':
                        add_linear_after_GMP = False
                elif args[30][0] == 'without_acc':
                    add_linear_after_GMP = True
                else:
                    add_linear_after_GMP = True
            else:
                ratio_reg = False
                add_linear_after_GMP = True

            z_struct_size = int(args[11][1])
            stride_size = int(args[23][1])
            classif_layer_size = int(args[13][1])
            hidden_filters_1 = int(args[24][1])
            hidden_filters_2 = int(args[25][1])
            hidden_filters_3 = int(args[26][1])
            if args[18][1] == 'True':
                two_conv_layer = True
            elif args[18][1] == 'False':
                two_conv_layer = False
            if args[19][1] == 'True':
                three_conv_layer = True
            elif args[19][1] == 'False':
                three_conv_layer = False
            if args[12][1] == 'True':
                add_classification_layer = True
            elif args[12][1] == 'False':
                add_classification_layer = False
            if args[20][1] == 'True':
                BK_in_first_layer = True
            elif args[20][1] == 'False':
                BK_in_first_layer = False
            if args[21][1] == 'True':
                BK_in_second_layer = True
            elif args[21][1] == 'False':
                BK_in_second_layer = False
            if args[22][1] == 'True':
                BK_in_third_layer = True
            elif args[22][1] == 'False':
                BK_in_third_layer = False

            big_kernel_size = int(args[17][1])

            if two_conv_layer:
                zstruct_size = str(hidden_filters_2)
            elif three_conv_layer:
                zstruct_size = str(hidden_filters_3)
            else:
                zstruct_size = str(hidden_filters_1)
            cat = 'zstruct_' + zstruct_size

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
                                  add_linear_after_GMP=add_linear_after_GMP,
                                  decoder_first_dense=decoder_first_dense,
                                  decoder_n_filter_1=decoder_n_filter_1,
                                  decoder_n_filter_2=decoder_n_filter_2,
                                  decoder_kernel_size_1=decoder_kernel_size_1,
                                  decoder_kernel_size_2=decoder_kernel_size_2,
                                  decoder_kernel_size_3=decoder_kernel_size_3,
                                  decoder_stride_1=decoder_stride_1,
                                  decoder_stride_2=decoder_stride_2,
                                  decoder_stride_3=decoder_stride_3)
        elif is_VAE:
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
        else:
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

        if exp_name in list_model:
            if is_decoder:
                run_decoder(exp_name, net)
            elif is_VAE:
                run_VAE(exp_name, net, lambda_BCE, beta, z_struct_size, z_var_size)
            else:
                run_viz_expes(exp_name, net, is_ratio, loss_min_distance_cl, loss_distance_mean,
                              net_type='Custom_CNN_BK',
                              cat=cat, ratio_reg=ratio_reg, diff_var_loss=diff_var)
                visualize_regions_of_interest(exp_name, net, net_type='Custom_CNN_BK')

        # save list models:
        # if z_struct_size == 5:
        #     if not os.path.exists(path_save_BK_95_acc_zstruct_5):
        #         if add_classification_layer:
        #             model_name = selection(net, exp_name)
        #             if model_name is not None:
        #                 list_model_selected_Custom_CNN_BK_95_acc_z_struct_5.append(model_name)

    # if not os.path.exists(path_save_BK_95_acc_zstruct_5):
    #     # save list models:
    #     with open(path_save_BK_95_acc_zstruct_5, 'w') as filehandle:
    #         for listitem in list_model_selected_Custom_CNN_BK_95_acc_z_struct_5:
    #             filehandle.write('%s\n' % listitem)

    file.close()


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
    net_trained, _, _ = get_checkpoints(net, path, exp_name)
    print(net)

    loader = test_loader

    if multi_label:
        # pass
        viz_decoder_multi_label(net, loader, exp_name, nb_img=10, nb_class=nb_class, save=True)
    else:
        # net_trained.eval()

        train_test = 'test'
        loader = test_loader
        loader_size = len(loader.dataset)
        size = (8, 8)

        with torch.no_grad():
            input_data = batch_test
        if torch.cuda.is_available():
            input_data = input_data.cuda()
        x_recon, _ = net_trained(input_data)

        # net_trained.train()

        recon_loss = F.mse_loss(x_recon, input_data).detach().numpy()
        recon_loss_around = np.around(recon_loss * 100, 2)

        comparison = build_compare_reconstruction(size, batch_test, input_data, x_recon)
        reconstructions = make_grid(comparison.data, nrow=size[0])

        # grid with originals data
        recon_grid = reconstructions.permute(1, 2, 0)
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='w', edgecolor='k')
        ax.set(title=('model: {}: reconstruction: MSE: {}'.format(exp_name, recon_loss_around)))

        ax.imshow(recon_grid.numpy())
        ax.axhline(y=size[0] // 2, linewidth=4, color='r')
        plt.show()
        if save:
            fig.savefig("fig_results/reconstructions/fig_reconstructions_z_" + exp_name + ".png")

    # viz_latent_prediction_reconstruction(net_trained, exp_name, img_size, size=8, random=True, batch=batch)

    return


def run_VAE(exp_name, net, lambda_BCE, beta, z_struct_size, z_var_size):
    print(exp_name, 'run VAE')
    path = 'checkpoints_CNN/'
    path_scores = 'checkpoint_scores_CNN'
    net_trained, _, _ = get_checkpoints(net, path, exp_name)

    # print(net)
    loader = test_loader
    batch = batch_test
    embedding_size = z_struct_size + z_var_size

    # losses:
    # plot_loss_results_VAE(path_scores, exp_name, beta, lambda_BCE, save=True)

    # reconstruction:
    # _, _, _, _, _ = viz_reconstructino_VAE(net_trained, loader, exp_name, z_var_size, z_struct_size, nb_img=10, nb_class=nb_class,
    #                                     save=True, z_reconstruction=True, z_struct_reconstruction=True,
    #                                     z_var_reconstruction=True, return_scores=False)

    # viz_latent_prediction_reconstruction(net_trained, exp_name, embedding_size, z_struct_size, z_var_size, size=8, random=False, batch=batch)

    # Image reconstruction with real distribution:

    # mu_var, sigma_var, mu_struct, sigma_struct = real_distribution_model(net_trained, exp_name, z_struct_size, z_var_size,
    #                                                                      loader, 'test', plot_gaussian=False, save=True)

    # z_struct sample:  sample from positive gaussian:
    # z_struct_sample = sigma_struct * np.random.randn(...) + mu_struct

    # _, _, _, _ = viz_reconstructino_VAE(net_trained, loader, exp_name, z_var_size, z_struct_size, nb_img=10,
    # nb_class=nb_class, save=True, z_reconstruction=True, z_struct_reconstruction=True, z_var_reconstruction=True,
    # return_scores=False, real_distribution=True, mu_var=mu_var, std_var=sigma_var, mu_struct=mu_struct,
    # std_struct=sigma_struct)

    switch_img(net, exp_name, loader, z_var_size)

    return


def run_viz_expes(exp_name, net, is_ratio, is_distance_loss, loss_distance_mean, net_type=None, cat=None,
                  ratio_reg=False, diff_var_loss=False, contrastive_loss=False):
    print(exp_name, 'run viz expes')
    path = 'checkpoints_CNN/'
    path_scores = 'checkpoint_scores_CNN'
    net_trained, _, nb_epochs = get_checkpoints(net, path, exp_name)
    # print(net)
    net_trained.eval()

    train_test = 'test'
    loader = test_loader
    loader_size = len(loader.dataset)

    # scores and losses:
    # plot_scores_and_loss_CNN(net_trained, exp_name, path_scores, is_ratio=ratio_reg, save=True,
    #                          is_distance_loss=is_distance_loss, loss_distance_mean=loss_distance_mean,
    #                          diff_var=diff_var_loss, contrastive_loss=contrastive_loss)
    # score_test, _ = compute_scores(net_trained, loader, device, loader_size)
    # score_train, _ = compute_scores(net_trained, train_loader, device, len(train_loader.dataset))
    # print('score Test acc: {:.3f}% and Train set acc: {:.3f}%'.format(score_test, score_train))

    # compute features:
    # compute_z_struct(net_trained, exp_name, loader, train_test=train_test, net_type=net_type)
    # compute_z_struct_representation_noised(net, exp_name, train_test=train_test, nb_repeat=10, nb_class=nb_class,
    #                                        net_type=net_type)
    # get_z_struct_per_class(exp_name, train_test=train_test, nb_class=nb_class)
    # get_average_z_struct_per_classes(exp_name=exp_name, train_test=train_test)
    # get_prediction_per_classes(exp_name, train_test=train_test)
    # get_prediction_noised_per_class(exp_name, train_test=train_test)
    # compute_all_score_acc(exp_name, train_test=train_test)
    # compute_mean_std_prediction(exp_name, train_test=train_test)

    # receptive_field = get_receptive_field(net_trained, img_size, net_type=net_type)

    # plot:
    # ratio_variance, variance_intra_class, variance_inter_class = ratio(exp_name, train_test=train_test, cat=cat)
    # print('ratio:', ratio_variance)
    # print('Variance intra class:', np.mean(variance_intra_class, axis=1))
    # print('Variance intra class averaged:', np.mean(np.mean(variance_intra_class, axis=1)))

    # score = correlation_filters(net_trained, exp_name, train_test=train_test, ch=nc, vis_filters=False, plot_fig=True,
    #                             cat=cat)
    # score_corr_class = dispersion_classes(exp_name, train_test=train_test, plot_fig=True, cat=cat)
    # ratio_variance = 'Nan'
    # plot_2d_projection_z_struct(nb_class, exp_name, train_test=train_test, ratio=ratio_variance)

    # plot_acc_bit_noised_per_class(exp_name,
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

    # _ = distance_matrix(net_trained, exp_name, train_test=train_test, plot_fig=True)

    # plot_resume(net_trained, exp_name, is_ratio, is_distance_loss, loss_distance_mean, loader, train_loader,
    #             device, cat=cat, train_test=train_test, path_scores=path_scores, diff_var=diff_var_loss,
    #             contrastive_loss=contrastive_loss)

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


def plot_histo():
    criterion_corr_filter_20 = 1
    criterion_corr_filter_50 = 1
    criterion_corr_class = 1
    criterion_loss_max = 1
    criterion_max_of_max_loss = 1
    criterion_ratio_variance = 7

    plot_histograms_models(criterion_corr_filter_20, criterion_corr_filter_50, criterion_corr_class, criterion_loss_max,
                           criterion_max_of_max_loss, criterion_ratio_variance, plot=True)

    model_selected_corr_filter_20 = np.load('list_model_selected/criterion_corr_filter_list_model_z_struct_20.npy',
                                            allow_pickle=True)
    model_selected_corr_filter_50 = np.load('list_model_selected/criterion_corr_filter_list_model_z_struct_50.npy',
                                            allow_pickle=True)
    model_selected_corr_class_20 = np.load('list_model_selected/criterion_corr_class_list_model_z_struct_20.npy',
                                           allow_pickle=True)
    model_selected_corr_class_50 = np.load('list_model_selected/criterion_corr_class_list_model_z_struct_50.npy',
                                           allow_pickle=True)
    model_selected_loss_max_20 = np.load('list_model_selected/criterion_loss_max_list_model_z_struct_20.npy',
                                         allow_pickle=True)
    model_selected_loss_max_50 = np.load('list_model_selected/criterion_loss_max_list_model_z_struct_50.npy',
                                         allow_pickle=True)
    model_selected_loss_max_of_max_20 = np.load(
        'list_model_selected/criterion_loss_max_of_max_list_model_z_struct_20.npy',
        allow_pickle=True)
    model_selected_loss_max_of_max_50 = np.load(
        'list_model_selected/criterion_loss_max_of_max_list_model_z_struct_50.npy',
        allow_pickle=True)
    model_selected_ratio_variance_20 = np.load(
        'list_model_selected/criterion_ratio_variance_list_model_z_struct_20.npy',
        allow_pickle=True)
    model_selected_ratio_variance_50 = np.load(
        'list_model_selected/criterion_ratio_variance_list_model_z_struct_50.npy',
        allow_pickle=True)

    list_model_selected_20 = []
    list_model_selected_50 = []

    for i in range(len(model_selected_corr_filter_20)):
        list_model_selected_20.append(model_selected_corr_filter_20[i][0])
    for i in range(len(model_selected_corr_filter_50)):
        list_model_selected_50.append(model_selected_corr_filter_50[i][0])
    for i in range(len(model_selected_corr_class_20)):
        list_model_selected_20.append(model_selected_corr_class_20[i][0])
    for i in range(len(model_selected_corr_class_50)):
        list_model_selected_50.append(model_selected_corr_class_50[i][0])
    for i in range(len(model_selected_loss_max_20)):
        list_model_selected_20.append(model_selected_loss_max_20[i][0])
    for i in range(len(model_selected_loss_max_50)):
        list_model_selected_50.append(model_selected_loss_max_50[i][0])
    for i in range(len(model_selected_loss_max_of_max_20)):
        list_model_selected_20.append(model_selected_loss_max_of_max_20[i][0])
    for i in range(len(model_selected_loss_max_of_max_50)):
        list_model_selected_50.append(model_selected_loss_max_of_max_50[i][0])
    for i in range(len(model_selected_ratio_variance_20)):
        list_model_selected_20.append(model_selected_ratio_variance_20[i][0])
    for i in range(len(model_selected_ratio_variance_50)):
        list_model_selected_50.append(model_selected_ratio_variance_50[i][0])

    list_model_selected_20 = np.array(list_model_selected_20)
    list_model_selected_50 = np.array(list_model_selected_50)

    list_model_selected_20 = np.unique(list_model_selected_20)
    list_model_selected_50 = np.unique(list_model_selected_50)

    np.save('list_model_selected/criterion_select_for_analyse_20.npy', list_model_selected_20)
    np.save('list_model_selected/criterion_select_for_analyse_50.npy', list_model_selected_50)

    print(len(list_model_selected_20), list_model_selected_20)
    print(len(list_model_selected_50), list_model_selected_50)

    # get model name who respect all criterions:
    list_models_select_all_criterion_20 = []

    for name in list_model_selected_20:
        print(name)
        if (name in model_selected_corr_filter_20) and (name in model_selected_corr_class_20) and \
                (name in model_selected_loss_max_of_max_20) and (name in model_selected_ratio_variance_20):
            list_models_select_all_criterion_20.append(name)

    print('select:')
    print(len(list_models_select_all_criterion_20), list_models_select_all_criterion_20)

    list_models_select_all_criterion_50 = []

    for name in list_model_selected_50:
        if (name in model_selected_corr_filter_50) and (name in model_selected_corr_class_50) and \
                (name in model_selected_loss_max_of_max_50) and (name in model_selected_ratio_variance_50):
            list_models_select_all_criterion_50.append(name)

    print(len(list_models_select_all_criterion_50), list_models_select_all_criterion_50)

    return


# plot_histo()


def selection(net, exp_name):
    path = 'checkpoints_CNN/'
    path_scores = 'checkpoint_scores_CNN'
    net_trained, _, nb_epochs = get_checkpoints(net, path, exp_name)
    # scores and losses:
    train_score, test_score = plot_scores_and_loss_CNN(net_trained, exp_name, path_scores, save=True, return_score=True)
    receptive_field = get_receptive_field(net, img_size)
    last_receptive_field_size = int(list(receptive_field.items())[-2][1]['r'])
    if train_score > 95 and test_score > 95 and last_receptive_field_size < 13:
        return exp_name

    return


# parameters:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_test = torch.load('data/batch_mnist.pt')
batch_size = 128  # 10000 to run regions visualization (we must have only one so all data set test in one batch)
train_loader, test_loader = get_mnist_dataset(batch_size=batch_size)

# if batch_size == 10000:
#     path_image_save = 'regions_of_interest/images/images.npy'
#     if not os.path.exists(path_image_save):
#         _, test_loader = get_mnist_dataset(batch_size=batch_size)
#         dataiter_test = iter(test_loader)
#         images_test, label_test = dataiter_test.next()
# 
#         print('load mnist dataset test with images shape: {}', images_test.shape)
# 
#         np.save('regions_of_interest/labels/labels.npy', label_test)
#         np.save(path_image_save, images_test)
#     else:
#         # load labels:
#         label_test = np.load('regions_of_interest/labels/labels.npy', allow_pickle=True)
#         images_test = torch.tensor(np.load('regions_of_interest/images/images.npy', allow_pickle=True))

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


def compute_heatmap(net_trained, train_loader, test_loader, latent_spec, device, exp_name, is_partial_rand_class,
                    is_E1):
    compute_heatmap_avg(train_loader, net_trained, latent_spec, device, exp_name, 'train', save=True, captum=False,
                        is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)
    compute_heatmap_avg(train_loader, net_trained, latent_spec, device, exp_name, 'train_captum', save=True,
                        captum=True,
                        is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)
    compute_heatmap_avg(test_loader, net_trained, latent_spec, device, exp_name, 'test', save=True, captum=False,
                        is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)
    compute_heatmap_avg(test_loader, net_trained, latent_spec, device, exp_name, 'test_captum', save=True, captum=True,
                        is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)

    return


def visualize(net, net_trained, nb_class, exp_name, device, latent_spec, train_loader, test_loader, nb_epochs=None,
              path_scores=None, batch=None, img_size=None, indx_image=None, path=None, losses=True, real_img=False,
              FID=False, IS=False, psnr=False, scores=True, all_prototype=False, copute_average_z_structural=False,
              is_partial_rand_class=False, all_classes_resum=True, save=False, scores_and_losses=False,
              size_struct=None, size_var=None, plot_gaussian=False, sample_real=False,
              heatmap=False, prototype=False, all_classes_details=False, project_2d=False, is_E1=False,
              reconstruction=False, plot_img_traversal=False, z_component_traversal=None, plot_sample=False,
              real_distribution=False, both_latent_traversal=False, is_zvar_sim_loss=False):
    if scores_and_losses:
        plot_scores_and_loss(net, exp_name, path_scores, save=save, partial_rand=is_partial_rand_class, losses=losses,
                             scores=scores)

    if reconstruction:
        viz_reconstruction(net, nb_epochs, exp_name, batch, latent_spec, img_size,
                           is_partial_rand_class=is_partial_rand_class,
                           partial_reconstruciton=True, is_E1=is_E1, save=save)

    if heatmap:
        plot_heatmap_avg(exp_name, latent_spec, all_classes_details=all_classes_details,
                         all_classes_resum=all_classes_resum,
                         train_test='train')
        plot_heatmap_avg(exp_name, latent_spec, all_classes_details=all_classes_details,
                         all_classes_resum=all_classes_resum,
                         train_test='train_captum')
        plot_heatmap_avg(exp_name, latent_spec, all_classes_details=all_classes_details,
                         all_classes_resum=all_classes_resum,
                         train_test='test')
        plot_heatmap_avg(exp_name, latent_spec, all_classes_details=all_classes_details,
                         all_classes_resum=all_classes_resum,
                         train_test='test_captum')

    if prototype:
        plot_prototype(net, exp_name, nb_class, latent_spec, device, test_loader, train_test='test',
                       print_per_class=True, print_per_var=True,
                       plot_traversal_struct=False, is_partial_rand_class=is_partial_rand_class, save=save)
        plot_prototype(net, exp_name, nb_class, latent_spec, device, train_loader, train_test='train',
                       print_per_class=True, print_per_var=True,
                       plot_traversal_struct=False, is_partial_rand_class=is_partial_rand_class, save=save)

    if project_2d:
        plot_prototype(net, exp_name, nb_class, latent_spec, device, train_loader, train_test='train',
                       print_per_class=False, print_per_var=False,
                       plot_traversal_struct=False, print_2d_projection=True,
                       is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)
        plot_prototype(net, exp_name, nb_class, latent_spec, device, test_loader, train_test='test',
                       print_per_class=False, print_per_var=False,
                       plot_traversal_struct=False, print_2d_projection=True,
                       is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)

    if plot_img_traversal:
        if both_latent_traversal:
            joint_latent_traversal(net, nb_epochs, path, exp_name, latent_spec, batch, img_size,
                                   both_continue=True, is_partial_rand_class=is_partial_rand_class, is_E1=is_E1,
                                   size_struct=size_struct, size_var=size_var, save=save, real_img=real_img)
        else:
            plot_images_taversal(net, exp_name, latent_spec, batch, nb_epochs, path, img_size, indx_image=indx_image,
                                 size=8, save=True, z_component_traversal=z_component_traversal,
                                 is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)

    if plot_sample:
        if sample_real:
            sample_real_distribution(net, path, exp_name, latent_spec, img_size, train_test='train', batch=batch,
                                     both_continue=True, save=True, FID=FID, IS=IS, psnr=psnr,
                                     is_partial_rand_class=is_partial_rand_class, is_E1=is_E1,
                                     is_zvar_sim_loss=is_zvar_sim_loss)
            sample_real_distribution(net, path, exp_name, latent_spec, img_size, train_test='test', batch=batch,
                                     both_continue=True, save=True, FID=FID, IS=IS, psnr=psnr,
                                     is_partial_rand_class=is_partial_rand_class, is_E1=is_E1,
                                     is_zvar_sim_loss=is_zvar_sim_loss)
        else:
            plot_samples(net, nb_epochs, path, exp_name, latent_spec, img_size, batch=batch, both_continue=True,
                         save=save,
                         FID=FID, IS=IS, psnr=psnr)

    if all_prototype:
        plot_prototype(net, exp_name, nb_class, latent_spec, device, train_loader, train_test='train',
                       print_per_class=True, print_per_var=True,
                       plot_traversal_struct=False, print_2d_projection=True,
                       is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)
        plot_prototype(net, exp_name, nb_class, latent_spec, device, test_loader, train_test='test',
                       print_per_class=True, print_per_var=True,
                       plot_traversal_struct=False, print_2d_projection=True,
                       is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)

    if copute_average_z_structural:
        plot_average_z_structural(net_trained, train_loader, device, nb_class, latent_spec, exp_name,
                                  train_test='train', both_continue=True,
                                  is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)
        plot_average_z_structural(net_trained, test_loader, device, nb_class, latent_spec, exp_name,
                                  train_test='test', both_continue=True,
                                  is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)

    if real_distribution:
        real_distribution_model(net, path, exp_name, train_loader, latent_spec, train_test='train',
                                is_both_continue=True, is_partial_rand_class=is_partial_rand_class, is_E1=is_E1,
                                plot_gaussian=plot_gaussian, save=save)
        real_distribution_model(net, path, exp_name, test_loader, latent_spec, train_test='test',
                                is_both_continue=True, is_partial_rand_class=is_partial_rand_class, is_E1=is_E1,
                                plot_gaussian=plot_gaussian, save=save)

    return


def run(expe_list, net, E1_VAE, latent_spec, train_loader, test_loader, is_E1, is_partial_rand_class,
        z_component_traversal):
    path = 'checkpoints/'
    path_scores = 'checkpoints_scores'
    for expe in expe_list:
        print(expe)
        exp_name = expe
        net_trained, _, nb_epochs = get_checkpoints(net, path, exp_name)
        # # scores and losses:
        visualize(net, net_trained, nb_class, exp_name, device, latent_spec, train_loader, test_loader,
                  path_scores=path_scores,
                  is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1,
                  losses=True)
        # sample:
        visualize(net, net_trained, nb_class, exp_name, device, latent_spec, train_loader, test_loader,
                  nb_epochs=nb_epochs,
                  path=path,
                  save=True, batch=batch, plot_sample=True, FID=False, IS=False, psnr=False)
        # reconstruction:
        visualize(net, net_trained, nb_class, exp_name, device, latent_spec, train_loader, test_loader,
                  nb_epochs=nb_epochs,
                  batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
                  save=True, is_E1=is_E1, reconstruction=True)
        # Traversal:
        visualize(net, net_trained, nb_class, exp_name, device, latent_spec, train_loader, test_loader,
                  nb_epochs=nb_epochs,
                  batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
                  is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image,
                  plot_img_traversal=True)
        # prototype
        visualize(net, net_trained, nb_class, exp_name, device, latent_spec, train_loader, test_loader,
                  copute_average_z_structural=True, is_partial_rand_class=is_partial_rand_class, save=True,
                  is_E1=is_E1)
        # projection 2d:
        visualize(net, net_trained, nb_class, exp_name, device, latent_spec, train_loader, test_loader,
                  all_prototype=True, is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1)
        # gaussian real distribution:
        if E1_VAE:
            visualize(net, net_trained, nb_class, exp_name, device, latent_spec, train_loader, test_loader, path=path,
                      save=True,
                      is_partial_rand_class=is_partial_rand_class, is_E1=is_E1, real_distribution=True,
                      plot_gaussian=True)
            # sample from real distribution:
            visualize(net, net_trained, nb_class, exp_name, device, latent_spec, train_loader, test_loader,
                      plot_sample=True,
                      sample_real=True, batch=batch, path=path,
                      save=True, FID=False, IS=False, psnr=False, is_partial_rand_class=is_partial_rand_class,
                      is_E1=is_E1)
        # traversal image with struct fixe and var fixe:
        visualize(net, net_trained, nb_class, exp_name, device, latent_spec, train_loader, test_loader, batch=batch,
                  path=path, real_img=False, size_struct=10, size_var=8,
                  is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1,
                  plot_img_traversal=True, both_latent_traversal=True)


def network(z_struct_size, big_kernel_size, stride_size, classif_layer_size, add_classification_layer,
            hidden_filters_1, hidden_filters_2, hidden_filters_3, BK_in_first_layer, two_conv_layer, three_conv_layer,
            BK_in_second_layer, BK_in_third_layer):
    model = Custom_CNN_BK(z_struct_size=z_struct_size,
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
                          BK_in_third_layer=BK_in_third_layer)
    return model


if __name__ == '__main__':
    exp_baseline = ['mnist_balanced_dataset_baseline']

    exp_classif_ratio = ['mnist_balanced_dataset_encoder_ratio_1',
                         'mnist_balanced_dataset_encoder_ratio_2',
                         'mnist_balanced_dataset_encoder_ratio_3',
                         'mnist_balanced_dataset_encoder_ratio_4',
                         'mnist_balanced_dataset_encoder_ratio_5']

    exp_classif_ratio_min = ['mnist_balanced_dataset_encoder_ratio_min_1_1',
                             'mnist_balanced_dataset_encoder_ratio_min_1_2',
                             'mnist_balanced_dataset_encoder_ratio_min_1_3',
                             'mnist_balanced_dataset_encoder_ratio_min_1_4',
                             'mnist_balanced_dataset_encoder_ratio_min_1_5',
                             'mnist_balanced_dataset_encoder_ratio_min_2_1',
                             'mnist_balanced_dataset_encoder_ratio_min_2_2',
                             'mnist_balanced_dataset_encoder_ratio_min_2_3',
                             'mnist_balanced_dataset_encoder_ratio_min_2_4',
                             'mnist_balanced_dataset_encoder_ratio_min_2_5',
                             'mnist_balanced_dataset_encoder_ratio_min_3_1',
                             'mnist_balanced_dataset_encoder_ratio_min_3_2',
                             'mnist_balanced_dataset_encoder_ratio_min_3_3',
                             'mnist_balanced_dataset_encoder_ratio_min_3_4',
                             'mnist_balanced_dataset_encoder_ratio_min_3_5']

    exp_classif_min_max = ['mnist_balanced_dataset_encoder_ratio_min_and_mean_1_1_1_1',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_1_1_2',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_1_1_3',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_1',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_2',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_3',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_3_1_1',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_3_1_2',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_3_1_3',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_1_2_1',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_1_2_2',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_1_2_3',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_2_1',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_2_2',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_2_3',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_3_2_1',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_3_2_2',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_3_2_3',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_1_3_1',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_1_3_2',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_1_3_3',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_3_1',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_3_2',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_3_3',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_3_3_1',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_3_3_2',
                           'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_3_3_3']

    exp_classif_diff_var = [# 'mnist_balanced_dataset_encoder_diff_var_1_1',
                            # 'mnist_balanced_dataset_encoder_diff_var_1_2',
                            # 'mnist_balanced_dataset_encoder_diff_var_1_3',
                            # 'mnist_balanced_dataset_encoder_diff_var_1_4',
                            # 'mnist_balanced_dataset_encoder_diff_var_1_5',
                            # 'mnist_balanced_dataset_encoder_diff_var_1_6',
                            # 'mnist_balanced_dataset_encoder_diff_var_2_1',
                            # 'mnist_balanced_dataset_encoder_diff_var_2_2',
                            # 'mnist_balanced_dataset_encoder_diff_var_2_3',
                            # 'mnist_balanced_dataset_encoder_diff_var_2_4',
                            # 'mnist_balanced_dataset_encoder_diff_var_2_5',
                            # 'mnist_balanced_dataset_encoder_diff_var_2_6',
                            # 'mnist_balanced_dataset_encoder_diff_var_3_1',
                            # 'mnist_balanced_dataset_encoder_diff_var_3_2',
                            # 'mnist_balanced_dataset_encoder_diff_var_3_3',
                            # 'mnist_balanced_dataset_encoder_diff_var_3_4',
                            # 'mnist_balanced_dataset_encoder_diff_var_3_5',
                            # 'mnist_balanced_dataset_encoder_diff_var_3_6',
                            # 'mnist_balanced_dataset_encoder_diff_var_4_1',
                            # 'mnist_balanced_dataset_encoder_diff_var_4_2',
                            # 'mnist_balanced_dataset_encoder_diff_var_4_3',
                            # 'mnist_balanced_dataset_encoder_diff_var_4_4',
                            'mnist_balanced_dataset_encoder_diff_var_4_5',
                            'mnist_balanced_dataset_encoder_diff_var_4_6',
                            'mnist_balanced_dataset_encoder_diff_var_5_1',
                            'mnist_balanced_dataset_encoder_diff_var_5_2',
                            'mnist_balanced_dataset_encoder_diff_var_5_3',
                            'mnist_balanced_dataset_encoder_diff_var_5_4',
                            'mnist_balanced_dataset_encoder_diff_var_5_5',
                            'mnist_balanced_dataset_encoder_diff_var_5_6']

    list_classif_diff_var_min = ['mnist_balanced_dataset_encoder_diff_var_min_1_1_1',
                                 'mnist_balanced_dataset_encoder_diff_var_min_1_1_2',
                                 'mnist_balanced_dataset_encoder_diff_var_min_1_1_3',
                                 'mnist_balanced_dataset_encoder_diff_var_min_1_2_1',
                                 'mnist_balanced_dataset_encoder_diff_var_min_1_2_2',
                                 'mnist_balanced_dataset_encoder_diff_var_min_1_2_3',
                                 'mnist_balanced_dataset_encoder_diff_var_min_1_3_1',
                                 'mnist_balanced_dataset_encoder_diff_var_min_1_3_2',
                                 'mnist_balanced_dataset_encoder_diff_var_min_1_3_3',
                                 'mnist_balanced_dataset_encoder_diff_var_min_2_1_1',
                                 'mnist_balanced_dataset_encoder_diff_var_min_2_1_2',
                                 'mnist_balanced_dataset_encoder_diff_var_min_2_1_3',
                                 'mnist_balanced_dataset_encoder_diff_var_min_2_2_1',
                                 'mnist_balanced_dataset_encoder_diff_var_min_2_2_2',
                                 'mnist_balanced_dataset_encoder_diff_var_min_2_2_3',
                                 'mnist_balanced_dataset_encoder_diff_var_min_2_3_1',
                                 'mnist_balanced_dataset_encoder_diff_var_min_2_3_2',
                                 'mnist_balanced_dataset_encoder_diff_var_min_2_3_3',
                                 'mnist_balanced_dataset_encoder_diff_var_min_3_1_1',
                                 'mnist_balanced_dataset_encoder_diff_var_min_3_1_2',
                                 'mnist_balanced_dataset_encoder_diff_var_min_3_1_3',
                                 'mnist_balanced_dataset_encoder_diff_var_min_3_2_1',
                                 'mnist_balanced_dataset_encoder_diff_var_min_3_2_2',
                                 'mnist_balanced_dataset_encoder_diff_var_min_3_2_3',
                                 'mnist_balanced_dataset_encoder_diff_var_min_3_3_1',
                                 'mnist_balanced_dataset_encoder_diff_var_min_3_3_2',
                                 'mnist_balanced_dataset_encoder_diff_var_min_3_3_3']

    list_classif_diff_var_min_mean = ['mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_1_1_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_1_2_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_1_3_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_2_1_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_2_2_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_2_3_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_3_1_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_3_2_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_3_3_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_1_1_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_1_2_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_1_3_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_2_1_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_2_2_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_2_3_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_3_1_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_3_2_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_3_3_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_1_1_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_1_2_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_1_3_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_2_1_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_2_2_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_2_3_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_3_1_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_3_2_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_3_3_1',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_1_1_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_1_2_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_1_3_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_2_1_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_2_2_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_2_3_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_3_1_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_3_2_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_3_3_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_1_1_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_1_2_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_1_3_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_2_1_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_2_2_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_2_3_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_3_1_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_3_2_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_3_3_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_1_1_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_1_2_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_1_3_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_2_1_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_2_2_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_2_3_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_3_1_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_3_2_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_3_3_2',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_1_1_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_1_2_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_1_3_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_2_1_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_2_2_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_2_3_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_3_1_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_3_2_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_1_3_3_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_1_1_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_1_2_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_1_3_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_2_1_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_2_2_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_2_3_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_3_1_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_3_2_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_2_3_3_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_1_1_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_1_2_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_1_3_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_2_1_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_2_2_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_2_3_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_3_1_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_3_2_3',
                                      'mnist_balanced_dataset_encoder_diff_var_min_and_mean_3_3_3_3']

    list_exp_selected = ['mnist_balanced_dataset_baseline_z_struct_8',
                         'mnist_balanced_dataset_baseline_z_struct_16',
                         'mnist_balanced_dataset_baseline_z_struct_64',
                         'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_1_z_struct_8',
                         'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_1_z_struct_16',
                         'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_1_z_struct_64',
                         'mnist_balanced_dataset_encoder_ratio_min_1_1_z_struct_8',
                         'mnist_balanced_dataset_encoder_ratio_min_1_1_z_struct_16',
                         'mnist_balanced_dataset_encoder_ratio_min_1_1_z_struct_64']

    lis_decoder = ['mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_1_decoder_1_2']

    lis_encoder = ['mnist_classif_balanced_dataset_intra_inter_1_6',
                   'mnist_baseline_balanced_128',
                   'mnist_classif_ratio_distance_intra_class_max_mean_1_6_4_balanced_dataset']
    
    list_exp_VAE = [# 'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_1_VAE_2c_32_5_1_2_5',
                    # 'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_1_VAE_2c_32_15_1_3_3',
                    # 'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_1_VAE_2c_32_30_1_3_3',
                    # 'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_1_VAE_2c_64_5_2_2_4',
                    # 'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_1_VAE_2c_64_15_2_2_3',
                    # 'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_1_VAE_2c_64_30_1_2_5',
                    'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_1_VAE_3c_32_5_1_1_4']
                    # 'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_1_VAE_3c_32_15_1_2_3',
                    # 'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_1_VAE_3c_32_30_1_2_5']
                    # 'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_1_VAE_3c_64_5_1_1_5',
                    # 'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_1_VAE_3c_64_15_1_1_3',
                    # 'mnist_balanced_dataset_encoder_ratio_min_and_mean_1_2_1_1_VAE_3c_64_30_2_1_4']

    parameters_mnist_classifier_BK_ratio = "parameters_combinations/mnist_classifier_ratio.txt"

    run_exp_extraction_and_visualization_custom_BK(parameters_mnist_classifier_BK_ratio,
                                                   2,
                                                   13,
                                                   list_exp_VAE,
                                                   is_ratio=False,
                                                   is_decoder=False,
                                                   is_VAE=True)
    
    # run_exp_extraction_and_visualization_custom_BK(parameters_mnist_classifier_BK_ratio,
    #                                                8,
    #                                                31,
    #                                                lis_decoder,
    #                                                is_ratio=False,
    #                                                is_decoder=True)

    # run_exp_extraction_and_visualization_custom_BK(parameters_mnist_classifier_BK_ratio,
    #                                                4,
    #                                                6,
    #                                                lis_encoder,
    #                                                is_ratio=True)

    # run_exp_extraction_and_visualization_custom_BK(parameters_mnist_classifier_BK_ratio,
    #                                                2,
    #                                                2,
    #                                                exp_baseline,
    #                                                is_ratio=True)
#
    # run_exp_extraction_and_visualization_custom_BK(parameters_mnist_classifier_BK_ratio,
    #                                                3,
    #                                                7,
    #                                                exp_classif_ratio,
    #                                                is_ratio=True)
#
    # run_exp_extraction_and_visualization_custom_BK(parameters_mnist_classifier_BK_ratio,
    #                                                8,
    #                                                22,
    #                                                exp_classif_ratio_min,
    #                                                is_ratio=True)
#
    # run_exp_extraction_and_visualization_custom_BK(parameters_mnist_classifier_BK_ratio,
    #                                                23,
    #                                                49,
    #                                                exp_classif_min_max,
    #                                                is_ratio=True)

    # run_exp_extraction_and_visualization_custom_BK(parameters_mnist_classifier_BK_ratio,
    #                                                50,
    #                                                79,
    #                                                exp_classif_diff_var,
    #                                                is_ratio=True)
#
    # run_exp_extraction_and_visualization_custom_BK(parameters_mnist_classifier_BK_ratio,
    #                                                80,
    #                                                106,
    #                                                list_classif_diff_var_min,
    #                                                is_ratio=True)
#
    # run_exp_extraction_and_visualization_custom_BK(parameters_mnist_classifier_BK_ratio,
    #                                                107,
    #                                                187,
    #                                                list_classif_diff_var_min_mean,
    #                                                is_ratio=True)

    # run_exp_extraction_and_visualization_custom_BK(parameters_mnist_classifier_BK_ratio,
    #                                                16,
    #                                                24,
    #                                                list_exp_selected,
    #                                                is_ratio=True)
