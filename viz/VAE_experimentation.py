from viz.visualizer_functions import model_load, plot_loss


def experimentation_VAE(model_name, loader, model_path=None, device=None, verbose=True, plot_losses=False):

    net, z_struct_size, z_var_size = model_load(model_name, model_path=model_path, device=device, verbose=verbose)
    embedding_size = z_struct_size + z_var_size

    net.eval()

    # losses:
    if plot_losses:
        plot_loss(model_name, device, save=True)

    # Compute VAE real distribution:
    mu_var, sigma_var, encoder_struct_zeros_proportion = real_distribution_model(net,
                                                                         model_name,
                                                                         z_struct_size,
                                                                         z_var_size,
                                                                         loader,
                                                                         'test',
                                                                         plot_gaussian=True,
                                                                         save=True,
                                                                         VAE_struct=VAE_struct,
                                                                         is_vae_var=is_vae_var)
#
    # # print(encoder_struct_zeros_proportion)
    # viz_reconstruction_VAE(net, loader, model_name, z_var_size, z_struct_size, nb_img=10,
    #                        nb_class=nb_class, save=True, z_reconstruction=True,
    #                        z_struct_reconstruction=True, z_var_reconstruction=True,
    #                        return_scores=False, real_distribution=True, mu_var=mu_var, std_var=sigma_var,
    #                        mu_struct=encoder_struct_zeros_proportion, is_vae_var=is_vae_var)

    # save code majoritaire with thier percent:
    # same_binary_code(net, model_name, loader, nb_class, train_test=train_test, save=True, Hmg_dist=False, is_VAE=True)
    # z_struct_code_classes(model_name, nb_class, train_test=train_test)

    # Uc bin maj:
    # percent_max = histo_count_uniq_code(model_name, train_test, plot_histo=False, return_percent=True)
    # maj_uc = np.load('binary_encoder_struct_results/uniq_code/uc_maj_class_' + model_name + '_' \
    #                            + train_test + '.npy', allow_pickle=True)
    # print(percent_max)
    # print(maj_uc)

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
    #                                  embedding_size=embedding_size, save=save,
    #                                  mu_struct=encoder_struct_zeros_proportion, traversal_latent=False)

    # score sample:
    # images_generation(net, model_name, batch, size=(8, 8), mu_var=mu_var, mu_struct=mu_struct, std_var=sigma_var,
    #                   std_struct=sigma_struct, z_var_size=z_var_size, z_struct_size=z_struct_size, FID=True, IS=True,
    #                   LPIPS=True, real_distribution=True, save=True)

    # Display a 2D manifold of the digits:
    # manifold_digit(net, model_name, device, size=(20, 20), component_var=0, component_struct=0, random=False,
    #                loader=loader, mu_var=mu_var, std_var=sigma_var, mu_struct=mu_struct, std_struct=sigma_struct,
    #                z_var_size=z_var_size, z_struct_size=z_struct_size, img_choice=None)

    net.train()

    return