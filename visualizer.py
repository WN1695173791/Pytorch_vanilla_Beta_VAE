import matplotlib.pyplot as plt
import os
import torch
from viz.visualize import Visualizer as Viz
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from matplotlib import cm
import matplotlib
#  from captum.attr import LayerConductance
from sample_scores.FID_score.fid_score import calculate_fid_given_paths
from sample_scores.inception_score import inception_score
import math
from torchvision.utils import make_grid

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def psnr_metric(mse):
    # mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def get_checkpoints(net, path, expe_name):
    file_path = os.path.join(path, expe_name, 'last')
    # print(file_path)
    checkpoint = torch.load(file_path, map_location=torch.device(device))
    net.load_state_dict(checkpoint['net'])
    nb_iter = checkpoint['iter']
    # nb_epochs = checkpoint['epochs']
    nb_epochs = nb_iter

    return net, nb_iter, nb_epochs


def get_checkpoints_scores_VAE(path_scores, expe_name):

    file_path = os.path.join(path_scores, expe_name, 'last')
    checkpoints_scores = torch.load(file_path, map_location=torch.device(device))

    global_iter = checkpoints_scores['iter']
    epochs = checkpoints_scores['epochs']
    train_score = checkpoints_scores['Total_loss_train']
    test_score = checkpoints_scores['Total_loss_test']
    BCE_train = checkpoints_scores['BCE_train']
    BCE_test = checkpoints_scores['BCE_test']
    KLD_train = checkpoints_scores['KLD_train']
    KLD_test = checkpoints_scores['KLD_test']

    return global_iter, epochs, train_score, test_score, BCE_train, BCE_test, KLD_train, KLD_test


def plot_loss_results_VAE(path_scores, expe_name, beta, lambda_BCE, save=True):

    global_iter, epochs, train_score, test_score, BCE_train, BCE_test, KLD_train, \
    KLD_test = get_checkpoints_scores_VAE(path_scores, expe_name)

    fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')

    ax.set(xlabel='nb_iter', ylabel='loss',
           title=('VAE losses: {}, with lambda BCE: {} and beta: {} '.format(expe_name, lambda_BCE, beta)))

    ax.plot(epochs, train_score, label='Total train')
    ax.plot(epochs, test_score, label='Total test')
    ax.plot(epochs, BCE_train, label='BCE train')
    ax.plot(epochs, BCE_test, label='BCE test')
    ax.plot(epochs, KLD_train, label='KLD train')
    ax.plot(epochs, KLD_test, label='KLD test')

    ax.legend(loc=1)
    plt.show()

    if save:
        fig.savefig("fig_results/losses/fig_losses_Test_Mnist_VAE_" + expe_name + ".png")

    return


def get_checkpoints_scores_CNN(path_scores, expe_name, is_ratio=False, is_distance_loss=False,
                               loss_distance_mean=False):
    file_path = os.path.join(path_scores, expe_name, 'last')
    checkpoints_scores = torch.load(file_path, map_location=torch.device(device))

    global_iter = checkpoints_scores['iter']
    epochs = checkpoints_scores['epochs']
    train_score = checkpoints_scores['train_score']
    test_score = checkpoints_scores['test_score']

    total_train = checkpoints_scores['total_train']
    total_test = checkpoints_scores['total_test']
    if 'ratio_train_loss' in checkpoints_scores:
        ratio_train_loss = checkpoints_scores['ratio_train_loss']
        ratio_test_loss = checkpoints_scores['ratio_test_loss']
    else:
        ratio_train_loss = 0
        ratio_test_loss = 0
    if 'var_distance_classes_train' in checkpoints_scores:
        var_distance_classes_train = checkpoints_scores['var_distance_classes_train']
        var_distance_classes_test = checkpoints_scores['var_distance_classes_test']
    else:
        var_distance_classes_train = 0
        var_distance_classes_test = 0
    if 'mean_distance_intra_class_train' in checkpoints_scores:
        mean_distance_intra_class_train = checkpoints_scores['mean_distance_intra_class_train']
        mean_distance_intra_class_test = checkpoints_scores['mean_distance_intra_class_test']
    else:
        mean_distance_intra_class_train = 0
        mean_distance_intra_class_test = 0
    if 'intra_var_train' in checkpoints_scores:
        intra_var_train = checkpoints_scores['intra_var_train']
        intra_var_test = checkpoints_scores['intra_var_test']
    else:
        intra_var_train = 0
        intra_var_test = 0
    if 'inter_var_train' in checkpoints_scores:
        inter_var_train = checkpoints_scores['inter_var_train']
        inter_var_test = checkpoints_scores['inter_var_test']
    else:
        inter_var_train = 0
        inter_var_test = 0
    if 'diff_var_train' in checkpoints_scores:
        diff_var_train = checkpoints_scores['diff_var_train']
        diff_var_test = checkpoints_scores['diff_var_test']
    else:
        diff_var_train = 0
        diff_var_test = 0
    if 'contrastive_train' in checkpoints_scores:
        contrastive_train = checkpoints_scores['contrastive_train']
        contrastive_test = checkpoints_scores['contrastive_test']
    else:
        contrastive_train = 0
        contrastive_test = 0
    if 'classification_test' in checkpoints_scores:
        classification_test = checkpoints_scores['classification_test']
        classification_train = checkpoints_scores['classification_train']
    else:
        classification_test = 0
        classification_train = 0

    return global_iter, epochs, train_score, test_score, total_train, total_test, ratio_train_loss, ratio_test_loss, \
           var_distance_classes_train, var_distance_classes_test, mean_distance_intra_class_train, \
           mean_distance_intra_class_test, intra_var_train, intra_var_test, inter_var_train, inter_var_test, \
           diff_var_train, diff_var_test, contrastive_train, contrastive_test, classification_test, classification_train


def get_checkpoints_scores(net, path_scores, expe_name):
    file_path = os.path.join(path_scores, expe_name, 'last')
    checkpoints_scores = torch.load(file_path, map_location=torch.device(device))

    global_iter = checkpoints_scores['iter']
    epochs = checkpoints_scores['epochs']
    Zc_Zd_train = checkpoints_scores['Zc_Zd_train']
    Zc_random_Zd_train = checkpoints_scores['Zc_random_Zd_train']
    Zc_Zd_random_train = checkpoints_scores['Zc_Zd_random_train']
    Zc_pert_Zd_train = checkpoints_scores['Zc_pert_Zd_train']
    Zc_Zd_pert_train = checkpoints_scores['Zc_Zd_pert_train']

    Zc_Zd_test = checkpoints_scores['Zc_Zd_test']
    Zc_random_Zd_test = checkpoints_scores['Zc_random_Zd_test']
    Zc_Zd_random_test = checkpoints_scores['Zc_Zd_random_test']
    Zc_pert_Zd_test = checkpoints_scores['Zc_pert_Zd_test']
    Zc_Zd_pert_test = checkpoints_scores['Zc_Zd_pert_test']

    recon_loss_train_train = checkpoints_scores['recon_loss_train']
    kl_disc_loss_train = checkpoints_scores['kl_class_loss_train']
    kl_cont_loss_train = checkpoints_scores['kl_var_loss_train']
    total_kl_train = checkpoints_scores['total_kld_train']
    if 'vae_loss_train' in checkpoints_scores:
        vae_loss_train = checkpoints_scores['vae_loss_train']
    else:
        vae_loss_train = checkpoints_scores['beta_vae_loss_train']
    if 'vae_loss_train' in checkpoints_scores:
        vae_loss_test = checkpoints_scores['vae_loss_test']
    else:
        vae_loss_test = checkpoints_scores['beta_vae_loss_test']
    classification_random_continue_loss_train = checkpoints_scores['classification_loss_train']
    classification_partial_rand_loss_train = checkpoints_scores['classification_partial_rand_loss_train']
    if 'zvar_sim_loss_train' in checkpoints_scores:
        zvar_sim_loss_train = checkpoints_scores['zvar_sim_loss_train']
    else:
        zvar_sim_loss_train = checkpoints_scores['MSE_loss_train']

    recon_loss_train_test = checkpoints_scores['recon_loss_test']
    kl_disc_loss_test = checkpoints_scores['kl_class_loss_test']
    kl_cont_loss_test = checkpoints_scores['kl_var_loss_test']
    total_kl_test = checkpoints_scores['total_kld_test']
    classification_random_continue_loss_test = checkpoints_scores['classification_loss_test']
    classification_partial_rand_loss_test = checkpoints_scores['classification_partial_rand_loss_test']
    if 'zvar_sim_loss_test' in checkpoints_scores:
        zvar_sim_loss_test = checkpoints_scores['zvar_sim_loss_test']
    else:
        zvar_sim_loss_test = checkpoints_scores['MSE_loss_test']

    one_bit_rand_mean_pred_train = checkpoints_scores['one_bit_rand_mean_pred_train']
    one_bit_rand_std_pred_train = checkpoints_scores['one_bit_rand_std_pred_train']
    one_bit_rand_mean_pred_test = checkpoints_scores['one_bit_rand_mean_pred_test']
    one_bit_rand_std_pred_test = checkpoints_scores['one_bit_rand_std_pred_test']
    one_bit_rand_noised_mean_pred_train = checkpoints_scores['one_bit_rand_noised_mean_pred_train']
    one_bit_rand_noised_std_pred_train = checkpoints_scores['one_bit_rand_noised_std_pred_train']
    one_bit_rand_noised_mean_pred_test = checkpoints_scores['one_bit_rand_noised_mean_pred_test']
    one_bit_rand_noised_std_pred_test = checkpoints_scores['one_bit_rand_noised_std_pred_test']

    return net, global_iter, epochs, Zc_Zd_train, Zc_random_Zd_train, Zc_Zd_random_train, Zc_pert_Zd_train, Zc_Zd_pert_train, \
           Zc_Zd_test, Zc_random_Zd_test, Zc_Zd_random_test, Zc_pert_Zd_test, Zc_Zd_pert_test, recon_loss_train_train, \
           kl_disc_loss_train, kl_cont_loss_train, total_kl_train, vae_loss_train, \
           classification_random_continue_loss_train, recon_loss_train_test, kl_disc_loss_test, kl_cont_loss_test, \
           total_kl_test, vae_loss_test, classification_random_continue_loss_test, classification_partial_rand_loss_train, \
           classification_partial_rand_loss_test, zvar_sim_loss_train, zvar_sim_loss_test, one_bit_rand_mean_pred_train, one_bit_rand_mean_pred_test, \
           one_bit_rand_std_pred_train, one_bit_rand_std_pred_test, one_bit_rand_noised_mean_pred_train, one_bit_rand_noised_mean_pred_test, \
           one_bit_rand_noised_std_pred_train, one_bit_rand_noised_std_pred_test


def viz_reconstruction(net, nb_epochs, expe_name, batch, latent_spec, img_size, both_continue=True, both_discrete=False,
                       is_partial_rand_class=False, partial_reconstruciton=False, is_E1=False, save=False):
    viz = Viz(net, img_size, latent_spec)
    viz.save_images = False

    recon_grid, recon_grid_random_var, recon_grid_random_classe, x_recon, recon_loss, recons_random_variability_loss, \
    recons_random_classe_loss = viz.reconstructions(batch, size=(8, 8),
                                                    both_continue=both_continue,
                                                    both_discrete=both_discrete,
                                                    partial_reconstruciton=partial_reconstruciton,
                                                    is_partial_rand_class=is_partial_rand_class,
                                                    is_E1=is_E1)

    size = recon_grid.shape[1:]
    if partial_reconstruciton:
        # The error is the amount by which the values of the original image differ from the degraded image.
        recon_loss_arounded = np.around(recon_loss * 100, 2)
        recon_loss_arounded_rand_var = np.around(recons_random_variability_loss * 100, 2)
        recon_loss_arounded_rand_classe = np.around(recons_random_classe_loss * 100, 2)

        # grid with originals data
        recon_grid = recon_grid.permute(1, 2, 0)
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='w', edgecolor='k')
        ax.set(title=('model: {}: original data reconstruction: MSE: {}'.format(expe_name, recon_loss_arounded)))

        ax.imshow(recon_grid.numpy())
        ax.axhline(y=size[0] // 2, linewidth=4, color='r')
        plt.show()
        if save:
            fig.savefig("fig_results/reconstructions/fig_reconstructions_z_" + expe_name + ".png")

        # grid with random var data
        recon_grid_random_var = recon_grid_random_var.permute(1, 2, 0)
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='w', edgecolor='k')
        ax.set(title=('model: {}: random var reconstruction: MSE: {}'.format(expe_name, recon_loss_arounded)))

        ax.imshow(recon_grid_random_var.numpy())
        ax.axhline(y=size[0] // 2, linewidth=4, color='r')
        plt.show()
        if save:
            fig.savefig("fig_results/reconstructions/fig_reconstructions_z_var_rand_" + expe_name + ".png")

        # grid with random classes data
        recon_grid_random_classe = recon_grid_random_classe.permute(1, 2, 0)
        fig, ax = plt.subplots(figsize=(10, 10), facecolor='w', edgecolor='k')
        ax.set(title=('model: {}: random struct reconstruction: MSE: {}'.format(expe_name, recon_loss_arounded)))
        ax.imshow(recon_grid_random_classe.numpy())
        ax.axhline(y=size[0] // 2, linewidth=4, color='r')
        plt.show()
        if save:
            fig.savefig("fig_results/reconstructions/fig_reconstructions_z_struct_rand_" + expe_name + ".png")

    else:
        recon_loss_arounded = np.around(recon_loss, 2)
        plt.figure(figsize=(10, 10))
        recon_grid = recon_grid.permute(1, 2, 0)
        plt.title('reconstruction: model: {}, MSE: {}'.format(expe_name, recon_loss_arounded))
        plt.imshow(recon_grid.numpy())
        plt.show()

        if save:
            plt.savefig("fig_results/reconstructions/fig_reconstructions_z_" + expe_name + ".png")

    return


def plot_samples(net, epochs, path, expe_name, latent_spec, img_size, size=(8, 8), batch=None, both_continue=False,
                 save=False, FID=False, IS=False, psnr=True):
    file_path = os.path.join(path, expe_name, 'last')
    checkpoint = torch.load(file_path, map_location=torch.device(device))
    net.load_state_dict(checkpoint['net'])

    viz = Viz(net, img_size, latent_spec)
    viz.save_images = False

    fig, ax = plt.subplots(figsize=(10, 10), facecolor='w', edgecolor='k')
    samples, generated = viz.samples(size=size, both_continue=both_continue)

    fid_value = 0
    IS_value = 0
    psnr_value = 0
    if FID:
        fid_value = calculate_fid_given_paths(batch,
                                              generated[:32],
                                              batch_size=32,
                                              cuda='',
                                              dims=2048)
        fid_value = np.around(fid_value, 3)
    if IS:
        IS_value = inception_score(generated,
                                   batch_size=32,
                                   resize=True)
        IS_value = np.around(IS_value[0], 3)
    if psnr:
        for i in range(len(batch)):
            psnr_value += psnr_metric(batch[i], generated[i])
        psnr_value /= len(batch)
        psnr_value = np.around(psnr_value.item(), 3)

    samples = samples.permute(1, 2, 0)
    ax.set(title=('samples: {}. Scores: FID(\u2193): {}, IS(\u2191): {}, PSNR(\u2193): {}'.format(expe_name,
                                                                                                  fid_value,
                                                                                                  IS_value,
                                                                                                  psnr_value)))
    ax.imshow(samples.numpy())
    plt.show()

    if save:
        fig.savefig("fig_results/sample/fig_sample_" + expe_name + ".png")

    return


def plot_all_traversal(net, epochs, path, expe_name, latent_spec, img_size, size=8, both_continue=False):
    file_path = os.path.join(path, expe_name, 'last')
    checkpoint = torch.load(file_path, map_location=torch.device(device))
    net.load_state_dict(checkpoint['net'])

    viz = Viz(net, img_size, latent_spec)
    viz.save_images = False

    traversals = viz.all_latent_traversals(size=size, both_continue=both_continue)
    plt.figure(figsize=(10, 10))
    traversals = traversals.permute(1, 2, 0)
    plt.title('all traversal: {}  with number of epochs: {}'.format(expe_name, str(epochs)))
    plt.imshow(traversals.numpy())
    plt.show()
    return


def latent_random_traversal(net, epochs, path, expe_name, latent_spec, img_size, size=8, indx=0, nb_samples=5,
                            both_continue=False):
    file_path = os.path.join(path, expe_name, 'last')
    checkpoint = torch.load(file_path, map_location=torch.device(device))
    net.load_state_dict(checkpoint['net'])

    viz = Viz(net, img_size, latent_spec)
    viz.save_images = False

    traversals = viz.latent_random_traversal(size=size, indx=indx, nb_samples=nb_samples, both_continue=both_continue)
    plt.figure(figsize=(10, 10))
    traversals = traversals.permute(1, 2, 0)
    plt.title(
        'latent random traversal: {} , with index = {}, at the epoch: {}'.format(expe_name, str(indx), str(epochs)))
    plt.imshow(traversals.numpy())
    plt.show()
    return


def latent_real_img_traversal(net, nb_epochs, path, expe_name, latent_spec, batch, img_size, indx_image=None,
                              z_component_traversal=None,
                              both_continue=True, is_partial_rand_class=False, is_E1=False,
                              size=None, save=False):
    file_path = os.path.join(path, expe_name, 'last')
    checkpoint = torch.load(file_path, map_location=torch.device(device))
    net.load_state_dict(checkpoint['net'])

    viz = Viz(net, img_size, latent_spec)
    viz.save_images = False

    traversals, recons, ori, indx_same_composante = viz.latent_real_img_traversal(batch, size=size,
                                                                                  z_component_traversal=z_component_traversal,
                                                                                  both_continue=both_continue,
                                                                                  indx_image=indx_image,
                                                                                  is_partial_rand_class=is_partial_rand_class,
                                                                                  is_E1=is_E1)
    fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
    traversals = traversals.permute(1, 2, 0)
    ax.set(title=('latent real traversal: {} , for composante: {}'.format(expe_name,
                                                                          z_component_traversal)))
    fig_size = traversals.shape
    plt.imshow(traversals.numpy())
    ax.axhline(y=(fig_size[0] * (latent_spec['cont_var'] / (latent_spec['cont_var'] + latent_spec['cont_class']))),
               linewidth=4, color='g')
    ax.axvline(x=(fig_size[1] // size) * indx_same_composante, linewidth=3, color='orange')
    ax.axvline(x=(fig_size[1] // size) * (indx_same_composante + 1), linewidth=3, color='orange')
    plt.show()

    if save:
        fig.savefig("fig_results/traversal_latent/fig_traversal_latent_" + "_" + expe_name + ".png")

    return


def joint_latent_traversal(net, nb_epochs, path, expe_name, latent_spec, batch, img_size, indx_image=None,
                           both_continue=True, is_partial_rand_class=False, is_E1=False,
                           size_struct=8, size_var=8, save=False, real_img=False):
    file_path = os.path.join(path, expe_name, 'last')
    checkpoint = torch.load(file_path, map_location=torch.device(device))
    net.load_state_dict(checkpoint['net'])

    viz = Viz(net, img_size, latent_spec)
    viz.save_images = False

    traversals = viz.joint_latent_traversal(batch, size_struct=size_struct, size_var=size_var,
                                            both_continue=both_continue, indx_image=indx_image,
                                            is_partial_rand_class=is_partial_rand_class, is_E1=is_E1,
                                            real_img=real_img)
    fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
    traversals = traversals.permute(1, 2, 0)
    ax.set(title=('both latent (real img={}) traversal: {}'.format(real_img, expe_name)))
    fig_size = traversals.shape
    plt.imshow(traversals.numpy())

    plt.show()

    if save:
        if real_img:
            fig.savefig("fig_results/traversal_latent/fig_both_traversal_latent_real_image_" + "_" + expe_name + ".png")
        else:
            fig.savefig("fig_results/traversal_latent/fig_both_traversal_latent_all_randn_" + "_" + expe_name + ".png")

    return


def plot_traversal_joint(net, path, expe_name, latent_spec, cont_idx, disc_idx, img_size, batch=None, real_im=False,
                         nb_samples=1, size=(10, 10)):
    file_path = os.path.join(path, expe_name, 'last')
    checkpoint = torch.load(file_path, map_location=torch.device(device))
    net.load_state_dict(checkpoint['net'])

    viz = Viz(net, img_size, latent_spec)
    viz.save_images = False

    size = (latent_spec['disc'][disc_idx], 8)
    cont_idx = cont_idx
    cont_axis = 1
    disc_idx = disc_idx
    disc_axis = 0

    traversals = viz.latent_traversal_grid(cont_idx=cont_idx, cont_axis=cont_axis,
                                           disc_idx=disc_idx, disc_axis=disc_axis, size=size,
                                           batch_chairs=batch, real_im=real_im,
                                           nb_samples=nb_samples)
    plt.figure(figsize=(10, 10))
    traversals = traversals.permute(1, 2, 0)
    plt.title('plot traversal joint with continue index={} and disc index={}, for expes: {}'.format(cont_idx, disc_idx,
                                                                                                    expe_name))
    plt.imshow(traversals.numpy())
    plt.show()
    return


def plot_scores_results(epochs, Zc_Zd_train, Zc_random_Zd_train, Zc_Zd_random_train, Zc_pert_Zd_train,
                        Zc_Zd_pert_train, Zc_Zd_test, Zc_random_Zd_test, Zc_Zd_random_test, Zc_pert_Zd_test,
                        Zc_Zd_pert_test, expe_name, is_wt_random, save):
    fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')

    ax.set(xlabel='nb_iter', ylabel='accuracy (%)',
           title=('MNIST sample_scores: ' + expe_name))  # ylim=(0,1))

    ax.plot(epochs, Zc_Zd_train, label='z train')
    ax.plot(epochs, Zc_random_Zd_train, label='z_var_rand + z_struct train')
    ax.plot(epochs, Zc_Zd_random_train, label='z_var + z_struct_rand train')
    ax.plot(epochs, Zc_Zd_test, label='z_test')
    ax.plot(epochs, Zc_random_Zd_test, label='z_var_rand + z_struct test')
    ax.plot(epochs, Zc_Zd_random_test, label='z_var + z_struct_rand test')

    ax.legend(loc=1)
    plt.show()

    if save:
        fig.savefig("fig_results/scores/fig_scores_Test_Mnist_L3_Classifier_" + expe_name + ".png")

    return


def plot_scores_results_CNN(epochs, train_score, test_score, expe_name, save):
    fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')

    ax.set(xlabel='nb_iter', ylabel='accuracy (%)',
           title=('MNIST scores: ' + expe_name))  # ylim=(0,1))

    ax.plot(epochs, train_score, label='train')
    ax.plot(epochs, test_score, label='test')
    ax.legend(loc=1)
    plt.show()

    if save:
        fig.savefig("fig_results/scores/fig_scores_Test_Mnist_CNN_" + expe_name + ".png")

    return


def plot_loss_results_CNN(epochs, total_train, total_test, ratio_train_loss, ratio_test_loss,
                          var_distance_classes_train, var_distance_classes_test, mean_distance_intra_class_train,
                          mean_distance_intra_class_test, intra_var_train, intra_var_test, inter_var_train,
                          inter_var_test, diff_var_train, diff_var_test, contrastive_train, contrastive_test,
                          classification_test, classification_train, expe_name, save, is_ratio=False,
                          is_distance_loss=False, loss_distance_mean=False, diff_var=False, contrastive_loss=False):

    fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')

    ax.set(xlabel='nb_iter', ylabel='loss',
           title=('MNIST total loss: ' + expe_name))

    ax.plot(epochs, total_train, label='train')
    ax.plot(epochs, total_test, label='test')
    ax.plot(epochs, classification_train, label='classification loss train')
    ax.plot(epochs, classification_test, label='classification loss test')
    if is_ratio:
        ax.plot(epochs, ratio_train_loss, label='ratio loss train')
        ax.plot(epochs, ratio_test_loss, label='ratio loss test')
    if is_distance_loss:
        ax.plot(epochs, var_distance_classes_train, label='std distance between class train')
        ax.plot(epochs, var_distance_classes_test, label='std distance between class train')
    if loss_distance_mean:
        ax.plot(epochs, mean_distance_intra_class_train, label='mean distance between class train')
        ax.plot(epochs, mean_distance_intra_class_test, label='mean distance between class train')
    if diff_var:
        ax.plot(epochs, diff_var_train, label='diff_var_train')
        ax.plot(epochs, diff_var_test, label='diff_var_test')
        ax.plot(epochs, intra_var_train, label='variance intra classes train')
        ax.plot(epochs, intra_var_test, label='variance intra classes test')
        ax.plot(epochs, inter_var_train, label='variance inter classes train')
        ax.plot(epochs, inter_var_test, label='variance inter classes test')
    if contrastive_loss:
        ax.plot(epochs, contrastive_train, label='contrastive loss train')
        ax.plot(epochs, contrastive_test, label='contrastive loss test')

    ax.legend(loc=1)
    plt.show()

    if save:
        fig.savefig("fig_results/losses/fig_losses_Test_Mnist_CNN_" + expe_name + ".png")

    return


def plot_loss_results(epochs, recon_loss_train_train, kl_disc_loss_train, kl_cont_loss_train, total_kl_train,
                      vae_loss_train, classification_random_continue_loss_train, recon_loss_train_test,
                      kl_disc_loss_test, kl_cont_loss_test, total_kl_test, vae_loss_test,
                      classification_random_continue_loss_test, classification_partial_rand_loss_train,
                      classification_partial_rand_loss_test, zvar_sim_loss_train, zvar_sim_loss_test, expe_name,
                      is_wt_random,
                      is_both_continuous, save, partial_rand=False):
    # facteur_loss_reconstruction_train = int(vae_loss_train[-1] / recon_loss_train_train[-1])
    # facteur_loss_classification_train = int(vae_loss_train[-1] / classification_random_continue_loss_train[-1])
    # facteur_loss_reconstruction_test = int(vae_loss_test[-1] / recon_loss_train_test[-1])
    # facteur_loss_classification_test = int(vae_loss_test[-1] / classification_random_continue_loss_test[-1])

    # max_vae_loss_train = np.max(vae_loss_train)
    # max_vae_loss_test = np.max(vae_loss_test)
    # max_vae = np.max([max_vae_loss_train, max_vae_loss_test])
    # recon_loss_train_train = np.array(recon_loss_train_train) * facteur_loss_reconstruction_train
    # classification_random_continue_loss_train = np.array(classification_random_continue_loss_train) * facteur_loss_classification_train
    # recon_loss_train_test = np.array(recon_loss_train_test) * facteur_loss_reconstruction_test
    # classification_random_continue_loss_test = np.array(classification_random_continue_loss_test) * facteur_loss_classification_test

    # vae_loss_train = np.array(vae_loss_train)
    # vae_loss_test = np.array(vae_loss_test)
    # zvar_sim_loss_train = np.array(zvar_sim_loss_train)
    # zvar_sim_loss_test = np.array(zvar_sim_loss_test)
    # vae_loss_train = vae_loss_train - zvar_sim_loss_train
    # vae_loss_test = vae_loss_test - zvar_sim_loss_test

    last_zvar_sim_value_train = zvar_sim_loss_train[-1]
    last_zvar_sim_value_test = zvar_sim_loss_test[-1]
    x_ann = epochs[-1]

    fig, ax = plt.subplots(4,
                           figsize=(15, 15),
                           gridspec_kw={'height_ratios': [3, 1, 1, 2]},
                           facecolor='w',
                           edgecolor='k')
    fig.suptitle('MNIST loss: ' + expe_name, fontsize=14)
    plt.subplots_adjust(left=0.05, bottom=0.035, right=0.99, top=.95, wspace=None, hspace=None)

    ax[3].set(xlabel='nb_iter', ylabel='Loss')
    ax[0].set(ylabel='Loss')
    ax[1].set(ylabel='Loss')
    ax[2].set(ylabel='Loss')
    ax[0].set(title='Total')
    ax[1].set(title='Reconstruction (MSE)')
    ax[2].set(title='Classification (NLL)')
    ax[3].set(title='Zvar (MSE)')

    ax[1].plot(epochs, recon_loss_train_train, label='recon_loss_train_train', color='royalblue')
    ax[1].plot(epochs, recon_loss_train_test, label='recon_loss_train_test', color='cornflowerblue')

    if is_both_continuous:
        ax[0].plot(epochs, kl_disc_loss_train, label='kl_cont_struct_train', color='gold')
        ax[0].plot(epochs, kl_cont_loss_train, label='kl_cont_var_train', color='burlywood')
        ax[0].plot(epochs, kl_disc_loss_test, label='kl_cont_struct_test', color='goldenrod')
        ax[0].plot(epochs, kl_cont_loss_test, label='kl_cont_var_test', color='tan')
    else:
        ax[0].plot(epochs, kl_disc_loss_train, label='kl_disc_loss_train')
        ax[0].plot(epochs, kl_cont_loss_train, label='kl_cont_loss_train')
        ax[0].plot(epochs, kl_disc_loss_test, label='kl_disc_loss_test')
        ax[0].plot(epochs, kl_cont_loss_test, label='kl_cont_loss_test')

    ax[0].plot(epochs, total_kl_train, label='total_kl_train', color='orange')
    ax[0].plot(epochs, total_kl_test, label='total_kl_test', color='darkorange')

    if not is_wt_random:
        ax[2].plot(epochs, classification_random_continue_loss_train,
                   label='classification_random_continue_loss_train', color='dodgerblue')
        ax[2].plot(epochs, classification_random_continue_loss_test,
                   label='classification_random_continue_loss_test', color='deepskyblue')

    if partial_rand:
        ax[2].plot(epochs, classification_partial_rand_loss_train,
                   label='class partial rand train',
                   color='mediumturquoise')
        ax[2].plot(epochs, classification_partial_rand_loss_test,
                   label='class partial rand train',
                   color='lightseagreen')

    ax[0].plot(epochs, vae_loss_train, label='vae_loss_train', color='red')
    ax[0].plot(epochs, vae_loss_test, label='vae_loss_test', color='darkred')

    ax[3].plot(epochs, zvar_sim_loss_train, label='zvar_sim train', color='green')
    ax[3].plot(epochs, zvar_sim_loss_test, label='zvar_sim test', color='darkgreen')

    ax[0].legend(loc=1)
    ax[1].legend(loc=1)
    ax[2].legend(loc=1)
    ax[3].legend(loc=1)

    string = 'Zvar_sim value: ' + str(last_zvar_sim_value_train)  # last_zvar_sim_value_train.detach().numpy())
    max_zvar_train = np.max(zvar_sim_loss_train)
    ax[3].annotate(string,
                   xy=(x_ann, last_zvar_sim_value_train), xycoords='data',
                   xytext=(x_ann - 50, last_zvar_sim_value_train + (max_zvar_train // 2)), textcoords='data',
                   size=20, va="center", ha="center",
                   bbox=dict(boxstyle="round4", fc="w"),
                   arrowprops=dict(arrowstyle="-|>",
                                   connectionstyle="arc3,rad=+0.2",
                                   fc="w"),
                   )

    plt.show()

    if save:
        fig.savefig("fig_results/losses/fig_losses_Test_Mnist_" + expe_name + ".png")

    """
    fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
    ax.set(xlabel='nb_iter', ylabel='Loss',
           title=('MNIST loss: ' + expe_name))  # ylim=(0,1))

    ax.plot(epochs, recon_loss_train_train, label='recon_loss_train_train', color='royalblue')
    ax.plot(epochs, recon_loss_train_test, label='recon_loss_train_test', color='cornflowerblue')

    ax.plot(epochs, classification_random_continue_loss_train, label='classification_random_continue_loss_train',
            color='dodgerblue')
    ax.plot(epochs, classification_random_continue_loss_test, label='classification_random_continue_loss_test',
            color='deepskyblue')

    ax.plot(epochs, zvar_sim_loss_train, label='zvar_sim train', color='green')
    ax.plot(epochs, zvar_sim_loss_test, label='zvar_sim test', color='darkgreen')

    ax.legend(loc=1)
    plt.show()

    if save:
        fig.savefig("fig_results/losses/fig_losses_Test_Mnist_zvar_recons_" + expe_name + ".png")

    fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
    ax.set(xlabel='nb_iter', ylabel='Loss',
           title=('MNIST loss: ' + expe_name))  # ylim=(0,1))

    ax.plot(epochs, zvar_sim_loss_train, label='zvar_sim train', color='green')
    ax.plot(epochs, zvar_sim_loss_test, label='zvar_sim test', color='darkgreen')

    ax.legend(loc=1)
    plt.show()

    if save:
        fig.savefig("fig_results/losses/fig_losses_Test_Mnist_L3_Classifier_zvar_" + expe_name + ".png")

    fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
    ax.set(xlabel='nb_iter', ylabel='Loss',
           title=('MNIST loss: ' + expe_name))  # ylim=(0,1))

    ax.plot(epochs, recon_loss_train_train, label='recon_loss_train_train', color='royalblue')
    ax.plot(epochs, recon_loss_train_test, label='recon_loss_train_test', color='cornflowerblue')

    ax.legend(loc=1)
    plt.show()

    if save:
        fig.savefig("fig_results/losses/fig_losses_Test_Mnist__recons_" + expe_name + ".png")
    """
    return


def plot_scores_and_loss(net, expe_name, path_scores, is_wt_random=False, is_both_continuous=True, save=False,
                         partial_rand=False, losses=False, scores=False):
    # net, nb_iter, epochs = get_checkpoints(net, path, expe_name)
    net, global_iter, epochs, Zc_Zd_train, Zc_random_Zd_train, Zc_Zd_random_train, Zc_pert_Zd_train, Zc_Zd_pert_train, \
    Zc_Zd_test, Zc_random_Zd_test, Zc_Zd_random_test, Zc_pert_Zd_test, Zc_Zd_pert_test, recon_loss_train_train, \
    kl_disc_loss_train, kl_cont_loss_train, total_kl_train, vae_loss_train, \
    classification_random_continue_loss_train, recon_loss_train_test, kl_disc_loss_test, kl_cont_loss_test, \
    total_kl_test, vae_loss_test, classification_random_continue_loss_test, classification_partial_rand_loss_train, \
    classification_partial_rand_loss_test, zvar_sim_loss_train, zvar_sim_loss_test, one_bit_rand_mean_pred_train, one_bit_rand_mean_pred_test, \
    one_bit_rand_std_pred_train, one_bit_rand_std_pred_test, one_bit_rand_noised_mean_pred_train, one_bit_rand_noised_mean_pred_test, \
    one_bit_rand_noised_std_pred_train, one_bit_rand_noised_std_pred_test = \
        get_checkpoints_scores(net, path_scores, expe_name)

    if losses:
        plot_loss_results(epochs, recon_loss_train_train, kl_disc_loss_train, kl_cont_loss_train, total_kl_train,
                          vae_loss_train, classification_random_continue_loss_train, recon_loss_train_test,
                          kl_disc_loss_test, kl_cont_loss_test, total_kl_test, vae_loss_test,
                          classification_random_continue_loss_test, classification_partial_rand_loss_train,
                          classification_partial_rand_loss_test, zvar_sim_loss_train, zvar_sim_loss_test, expe_name,
                          is_wt_random,
                          is_both_continuous, save, partial_rand=partial_rand)

    if scores:
        plot_scores_results(epochs, Zc_Zd_train, Zc_random_Zd_train, Zc_Zd_random_train, Zc_pert_Zd_train,
                            Zc_Zd_pert_train, Zc_Zd_test, Zc_random_Zd_test, Zc_Zd_random_test, Zc_pert_Zd_test,
                            Zc_Zd_pert_test, expe_name, is_wt_random, save)

    return


def plot_scores_and_loss_CNN(net, expe_name, path_scores, is_ratio=False, save=False, return_score=False,
                             is_distance_loss=False, loss_distance_mean=False, diff_var=False, contrastive_loss=False):
    global_iter, epochs, train_score, test_score, total_train, total_test, ratio_train_loss, ratio_test_loss, \
    var_distance_classes_train, var_distance_classes_test, mean_distance_intra_class_train, \
    mean_distance_intra_class_test, intra_var_train, intra_var_test, inter_var_train, inter_var_test, \
    diff_var_train, diff_var_test, contrastive_train, contrastive_test, classification_test, \
    classification_train = get_checkpoints_scores_CNN(path_scores,
                                                      expe_name,
                                                      is_ratio=is_ratio,
                                                      is_distance_loss=is_distance_loss,
                                                      loss_distance_mean=loss_distance_mean)

    if return_score:
        return train_score[-1], test_score[-1]
    else:
        plot_loss_results_CNN(epochs, total_train, total_test, ratio_train_loss, ratio_test_loss, \
                              var_distance_classes_train, var_distance_classes_test, mean_distance_intra_class_train, \
                              mean_distance_intra_class_test, intra_var_train, intra_var_test, inter_var_train,
                              inter_var_test, \
                              diff_var_train, diff_var_test, contrastive_train, contrastive_test, classification_test, \
                              classification_train, expe_name, save, is_ratio=is_ratio,
                              is_distance_loss=is_distance_loss, loss_distance_mean=loss_distance_mean,
                              diff_var=diff_var, contrastive_loss=contrastive_loss)

        plot_scores_results_CNN(epochs, train_score, test_score, expe_name, save)

    return


def plot_all(net, expe_name, path, path_scores, bacth, latent_spec, both_continue):
    net, nb_iter, epochs = get_checkpoints(net, path, expe_name)
    net, global_iter, epochs, Zc_Zd_train, Zc_random_Zd_train, Zc_Zd_random_train, Zc_pert_Zd_train, Zc_Zd_pert_train, \
    Zc_Zd_test, Zc_random_Zd_test, Zc_Zd_random_test, Zc_pert_Zd_test, Zc_Zd_pert_test, recon_loss_train_train, \
    kl_disc_loss_train, kl_cont_loss_train, total_kl_train, vae_loss_train, \
    classification_random_continue_loss_train, recon_loss_train_test, kl_disc_loss_test, kl_cont_loss_test, \
    total_kl_test, vae_loss_test, classification_random_continue_loss_test = \
        get_checkpoints_scores(net, path_scores, expe_name)

    viz_reconstruction(net, epochs, expe_name, bacth, latent_spec, both_continue)

    plot_scores_results(epochs, Zc_Zd_train, Zc_random_Zd_train, Zc_Zd_random_train, Zc_pert_Zd_train,
                        Zc_Zd_pert_train, Zc_Zd_test, Zc_random_Zd_test, Zc_Zd_random_test, Zc_pert_Zd_test,
                        Zc_Zd_pert_test, expe_name)

    plot_loss_results(epochs, recon_loss_train_train, kl_disc_loss_train, kl_cont_loss_train,
                      total_kl_train, vae_loss_train, classification_random_continue_loss_train,
                      recon_loss_train_test, kl_disc_loss_test, kl_cont_loss_test, total_kl_test, vae_loss_test,
                      classification_random_continue_loss_test, expe_name)

    plot_samples(net, epochs, path, expe_name, latent_spec, bacth, both_continue=both_continue)
    plot_all_traversal(net, epochs, path, expe_name, latent_spec, both_continue=both_continue)

    indx = 0
    nb_samples = 10
    latent_dim = latent_spec['cont_var'] + latent_spec['cont_class']
    latent_random_traversal(net, epochs, path, expe_name, latent_spec, indx=indx, nb_samples=nb_samples,
                            both_continue=both_continue)
    latent_real_img_traversal(net, epochs, path, expe_name, latent_spec, bacth, indx=indx, nb_samples=nb_samples,
                              both_continue=both_continue)

    """
    for i in range(latent_dim):
        indx = i
        latent_random_traversal(net, path, expe_name, latent_spec, indx=indx, nb_samples=nb_samples,
                                both_continue=both_continue)

    for i in range(latent_dim):
        indx = i
        latent_real_img_traversal(net, path, expe_name, latent_spec, batch_mnist, indx=indx, nb_samples=nb_samples,
                                  both_continue=both_continue)
    """


def real_distribution_model(net, path_expe, expe_name, loader, latent_spec, train_test, is_both_continue=False,
                            is_both_discrete=False, is_partial_rand_class=False, is_E1=False, is_zvar_sim_loss=False,
                            plot_gaussian=False, save=False):
    path = 'Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test + '_mu_var.npy'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(path):
        file_path = os.path.join(path_expe, expe_name, 'last')
        checkpoint = torch.load(file_path, map_location=torch.device(device))
        net.load_state_dict(checkpoint['net'])

        mu_var = torch.zeros(0, latent_spec['cont_var'])
        mu_struct = torch.zeros(0, latent_spec['cont_class'])
        z = torch.zeros(0, latent_spec['cont_var'] + latent_spec['cont_class'])

        sigma_var = torch.zeros(0, latent_spec['cont_var'])
        sigma_struct = torch.zeros(0, latent_spec['cont_class'])

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        nb_batch = 0
        nb_images = 0
        with torch.no_grad():
            for x in loader:
                nb_images += len(x)
                nb_batch += 1

                data = x[0]
                data = data.to(device)  # Variable(data.to(device))

                # compute loss:
                x_recon, _, _, latent_representation, latent_sample, \
                latent_sample_variability, latent_sample_class, latent_sample_random_continue, prediction, pred_noised, \
                prediction_partial_rand_class, prediction_random_variability, prediction_random_class, \
                prediction_zc_pert_zd, prediction_zc_zd_pert, z_var, \
                z_var_reconstructed = net(data,
                                          is_perturbed_score=False,
                                          is_noise_stats=False,
                                          is_prediction=False,
                                          both_continue=is_both_continue,
                                          both_discrete=is_both_discrete,
                                          is_partial_rand_class=is_partial_rand_class,
                                          random_percentage=False,
                                          is_E1=is_E1,
                                          is_zvar_sim_loss=is_zvar_sim_loss)

                mu_var_iter = latent_representation['cont_var'][0]
                mu_struct_iter = latent_representation['cont_class'][0]
                sigma_var_iter = latent_representation['cont_var'][1]
                sigma_struct_iter = latent_representation['cont_class'][1]

                mu_var = torch.cat((mu_var, mu_var_iter), 0)
                mu_struct = torch.cat((mu_struct, mu_struct_iter), 0)
                sigma_var = torch.cat((sigma_var, sigma_var_iter), 0)
                sigma_struct = torch.cat((sigma_struct, sigma_struct_iter), 0)

                z = torch.cat((z, latent_sample), 0)

        z_mean = torch.mean(z, axis=0)
        z_var = torch.std(z, axis=0)

        mu_var = torch.mean(mu_var, axis=0)
        mu_struct = torch.mean(mu_struct, axis=0)
        sigma_var = torch.mean(sigma_var, axis=0)
        sigma_struct = torch.mean(sigma_struct, axis=0)

        np.save('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
                '_mu_var.npy', mu_var)
        np.save('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
                'mu_struct.npy', mu_struct)
        np.save('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
                'sigma_var.npy', sigma_var)
        np.save('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
                'sigma_struct.npy', sigma_struct)
        np.save('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
                '_z_mean.npy', z_mean)
        np.save('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
                '_z_var.npy', z_var)

    mu_var = np.load('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
                     '_mu_var.npy', allow_pickle=True)
    mu_struct = np.load('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
                        'mu_struct.npy', allow_pickle=True)
    sigma_var = np.load('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
                        'sigma_var.npy', allow_pickle=True)
    sigma_struct = np.load(
        'Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
        'sigma_struct.npy', allow_pickle=True)
    z_mean = np.load('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
                     '_z_mean.npy', allow_pickle=True)
    z_var = np.load('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
                    '_z_var.npy', allow_pickle=True)

    if plot_gaussian:
        mu = 0
        variance = 1
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

        # mu_mean_real_var = np.mean(mu_var, axis=0)
        # variance_mean_real_var = np.mean(sigma_var, axis=0)
        sigma_mean_real_var = np.sqrt(np.abs(sigma_var))
        x_mean_real_var = np.linspace(mu_var - 3 * sigma_mean_real_var,
                                      mu_var + 3 * sigma_mean_real_var, 100)

        # mu_mean_real_struct = np.mean(mu_struct, axis=0)
        # variance_mean_real_struct = np.mean(sigma_struct, axis=0)
        sigma_mean_real_struct = np.sqrt(np.abs(sigma_struct))
        x_mean_real_struct = np.linspace(mu_struct - 3 * sigma_mean_real_struct,
                                         mu_struct + 3 * sigma_mean_real_struct, 100)

        sigma_z = np.sqrt(np.abs(z_var))
        x_mean_z = np.linspace(z_mean - 3 * sigma_z, z_mean + 3 * sigma_z, 100)

        # mu_mean_real = (mu_mean_real_var + mu_mean_real_struct)/2
        # variance_mean_real = (variance_mean_real_var + variance_mean_real_struct)/2
        # sigma_mean_real = math.sqrt(np.abs(variance_mean_real))
        # x_mean_real = np.linspace(mu_mean_real - 3 * sigma_mean_real,
        #                           mu_mean_real + 3 * sigma_mean_real, 100)

        fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
        ax.set(title=('Gaussian: ' + expe_name + "_" + train_test))

        ax.plot(x_mean_real_var, stats.norm.pdf(x_mean_real_var, mu_var, sigma_mean_real_var),
                label='real zvar', color='blue')
        ax.plot(x_mean_real_struct, stats.norm.pdf(x_mean_real_struct, mu_struct, sigma_mean_real_struct),
                label='real zstruct', color='green')
        # ax.plot(x_mean_real, stats.norm.pdf(x_mean_real, mu_mean_real, sigma_mean_real),
        #         label='mean real', color='orange')
        ax.plot(x, stats.norm.pdf(x, mu, sigma), label='Gaussian (0, I)', color='red')

        ax.plot(x_mean_z, stats.norm.pdf(x_mean_z, z_mean, sigma_z), label='Gaussian after variational', color='black')

        ax.legend(loc=1)
        plt.show()

        if save:
            fig.savefig("fig_results/plot_distribution/fig_plot_distribution_" + expe_name + "_" + train_test + ".png")

    return mu_var, mu_struct, sigma_var, sigma_struct, z_mean, z_var


def sample_real_distribution(net, path, expe_name, latent_spec, img_size, train_test, size=(8, 8), batch=None,
                             both_continue=False, save=False, FID=False, IS=False, psnr=True,
                             is_partial_rand_class=False, is_E1=False, is_zvar_sim_loss=False):
    file_path = os.path.join(path, expe_name, 'last')
    checkpoint = torch.load(file_path, map_location=torch.device(device))
    net.load_state_dict(checkpoint['net'])

    loader = None
    mu_var, mu_struct, sigma_var, sigma_struct, z_mean, z_var = real_distribution_model(net,
                                                                                        path,
                                                                                        expe_name,
                                                                                        loader,
                                                                                        latent_spec,
                                                                                        train_test,
                                                                                        is_both_continue=True,
                                                                                        is_partial_rand_class=is_partial_rand_class,
                                                                                        is_E1=is_E1,
                                                                                        is_zvar_sim_loss=is_zvar_sim_loss)

    # mu = [mu_var, mu_struct]
    # var = [sigma_var, sigma_struct]
    mu = [z_mean[:latent_spec['cont_var']], z_mean[latent_spec['cont_var']:]]
    var = [z_var[:latent_spec['cont_var']], z_var[latent_spec['cont_var']:]]

    viz = Viz(net, img_size, latent_spec)
    viz.save_images = False

    fig, ax = plt.subplots(figsize=(10, 10), facecolor='w', edgecolor='k')
    samples, generated = viz.samples(size=size, both_continue=both_continue, real_distribution=True,
                                     mu=mu, var=var)

    fid_value = 0
    IS_value = 0
    psnr_value = 0
    if FID:
        fid_value = calculate_fid_given_paths(batch,
                                              generated[:32],
                                              batch_size=32,
                                              cuda='',
                                              dims=2048)
        fid_value = np.around(fid_value, 3)
    if IS:
        IS_value = inception_score(generated,
                                   batch_size=32,
                                   resize=True)
        IS_value = np.around(IS_value[0], 3)
    if psnr:
        for i in range(len(batch)):
            psnr_value += psnr_metric(batch[i], generated[i])
        psnr_value /= len(batch)
        psnr_value = np.around(psnr_value.item(), 3)

    samples = samples.permute(1, 2, 0)
    ax.set(title=('samples: {}. Scores: FID(\u2193): {}, IS(\u2191): {}, PSNR(\u2193): {}'.format(expe_name,
                                                                                                  fid_value,
                                                                                                  IS_value,
                                                                                                  psnr_value)))
    ax.imshow(samples.numpy())
    plt.show()

    if save:
        fig.savefig("fig_results/sample/fig_sample_real_distribution_" + expe_name + ".png")

    return


def plot_Acc_each_class(one_bit_rand_mean_pred_train, one_bit_rand_std_pred_train, one_bit_rand_mean_pred_test,
                        one_bit_rand_std_pred_test, epochs, nb_epochs, expe_name):
    """
    here we see ACC score (%) for each class with z_rand: train and test
    :return:
    """

    # pred stats for z_var_rand_struct:
    one_bit_rand_mean_pred_train = np.array(one_bit_rand_mean_pred_train)
    one_bit_rand_std_pred_train = np.array(one_bit_rand_std_pred_train)
    one_bit_rand_mean_pred_test = np.array(one_bit_rand_mean_pred_test)
    one_bit_rand_std_pred_test = np.array(one_bit_rand_std_pred_test)

    # viz stat prediction at the learnign end:
    fig_train, ax_train = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
    ax_train.set(xlabel='epochs', ylabel='accuracy (%)',
                 title=('Model: {}, at the epoch: {}. ACC (%) with z_var_rand each class train').format(expe_name,
                                                                                                        nb_epochs))

    for i in range(10):
        ax_train.plot(epochs, one_bit_rand_mean_pred_train[:, i], label='class ' + str(i))
        ax_train.fill_between(epochs, one_bit_rand_mean_pred_train[:, i] + one_bit_rand_std_pred_train[:, i],
                              one_bit_rand_mean_pred_train[:, i] - one_bit_rand_std_pred_train[:, i], alpha=0.5)
    ax_train.legend(loc=1)
    fig_train.show()

    # test
    fig_test, ax_test = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
    ax_test.set(xlabel='epochs', ylabel='accuracy (%)',
                title=('Model: {}, at the epoch: {}. ACC (%) with z_var_rand each class test').format(expe_name,
                                                                                                      nb_epochs))

    for i in range(10):
        ax_test.plot(epochs, one_bit_rand_mean_pred_test[:, i], label='class ' + str(i))
        ax_test.fill_between(epochs, one_bit_rand_mean_pred_test[:, i] + one_bit_rand_std_pred_test[:, i],
                             one_bit_rand_mean_pred_test[:, i] - one_bit_rand_std_pred_test[:, i], alpha=0.5)
    ax_test.legend(loc=1)
    fig_test.show()

    return


def plot_Acc_all_classes(one_bit_rand_mean_pred_train, one_bit_rand_std_pred_train, one_bit_rand_mean_pred_test,
                         one_bit_rand_std_pred_test, epochs, nb_epochs, expe_name):
    """
    here we see ACC score (%) for all classes with z_rand: train and test
    :return:
    """

    # train and test on all class
    fig_train, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
    ax.set(xlabel='epochs', ylabel='accuracy (%)',
           title=('Model: {}, at the epoch: {}. ACC (%) with z_var_rand all class train').format(expe_name,
                                                                                                 nb_epochs))

    one_bit_rand_mean_pred_train_mean = np.mean(one_bit_rand_mean_pred_train, axis=1)
    one_bit_rand_std_pred_train_mean = np.mean(one_bit_rand_std_pred_train, axis=1)
    one_bit_rand_mean_pred_test_mean = np.mean(one_bit_rand_mean_pred_test, axis=1)
    one_bit_rand_std_pred_test_mean = np.mean(one_bit_rand_std_pred_test, axis=1)

    ax.plot(epochs, one_bit_rand_mean_pred_train_mean, label='all class')
    ax.fill_between(epochs, one_bit_rand_mean_pred_train_mean + one_bit_rand_std_pred_train_mean,
                    one_bit_rand_mean_pred_train_mean - one_bit_rand_std_pred_train_mean, alpha=0.5)
    ax.plot(epochs, one_bit_rand_mean_pred_test_mean, label='all class')
    ax.fill_between(epochs, one_bit_rand_mean_pred_test_mean + one_bit_rand_std_pred_test_mean,
                    one_bit_rand_mean_pred_test_mean - one_bit_rand_std_pred_test_mean, alpha=0.5)
    ax.legend(loc=1)
    fig_train.show()

    return


def plot_one_bit_noised_along_training(one_bit_rand_noised_mean_pred_train, one_bit_rand_noised_std_pred_train,
                                       one_bit_rand_noised_mean_pred_test, one_bit_rand_noised_std_pred_test, epochs,
                                       nb_epochs, expe_name):
    """
    for each bit i in z_struct we see the classification score on the train or test set when the bit i noised.
    :return:
    """

    one_bit_rand_noised_mean_pred_train = np.array(
        one_bit_rand_noised_mean_pred_train)  # shape: (nb_epoch, nb_z_struct_noised, nb_class)
    one_bit_rand_noised_std_pred_train = np.array(one_bit_rand_noised_std_pred_train)

    one_bit_rand_mean_pred_train_all_class_noised = np.mean(one_bit_rand_noised_mean_pred_train,
                                                            axis=2)  # get array shape (nb_epoch, nb_z_struct_noised, nb_class)
    one_bit_rand_std_pred_train_all_class_noised = np.mean(one_bit_rand_noised_std_pred_train, axis=2)

    one_bit_rand_noised_mean_pred_test = np.array(one_bit_rand_noised_mean_pred_test)
    one_bit_rand_noised_std_pred_test = np.array(one_bit_rand_noised_std_pred_test)

    one_bit_rand_mean_pred_test_all_class_noised = np.mean(one_bit_rand_noised_mean_pred_test, axis=2)
    one_bit_rand_std_pred_test_all_class_noised = np.mean(one_bit_rand_noised_std_pred_test, axis=2)

    for i in range(25):
        bit = i
        fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')

        ax.set(xlabel='epochs', ylabel='accuracy (%)',
               title=('Model: {}, at the epoch: {}. bit {} noised'.format(expe_name, nb_epochs, str(bit))))

        ax.plot(epochs, one_bit_rand_mean_pred_train_all_class_noised[:, bit], label='all class train set')
        ax.fill_between(epochs, one_bit_rand_mean_pred_train_all_class_noised[:,
                                bit] + one_bit_rand_std_pred_train_all_class_noised[:, bit],
                        one_bit_rand_mean_pred_train_all_class_noised[:,
                        bit] - one_bit_rand_std_pred_train_all_class_noised[:, bit],
                        alpha=0.5)
        ax.plot(epochs, one_bit_rand_mean_pred_test_all_class_noised[:, bit], label='all class test set')
        ax.fill_between(epochs, one_bit_rand_mean_pred_test_all_class_noised[:,
                                bit] + one_bit_rand_std_pred_test_all_class_noised[:, bit],
                        one_bit_rand_mean_pred_test_all_class_noised[:,
                        bit] - one_bit_rand_std_pred_test_all_class_noised[:, bit],
                        alpha=0.5)
        ax.legend(loc=1)
        plt.show()

    return


def plot_influence_bit_noised_classification(one_bit_rand_noised_mean_pred_train, one_bit_rand_noised_std_pred_train,
                                             one_bit_rand_mean_pred_train, one_bit_rand_std_pred_train,
                                             one_bit_rand_noised_mean_pred_test, one_bit_rand_noised_std_pred_test,
                                             one_bit_rand_mean_pred_test, one_bit_rand_std_pred_test, nb_epochs,
                                             expe_name):
    """
    for all clases, we see the importance of each bits at the last epoch.
    We see, if we noise the bit i, what is the prediction mean.
    :return:
    """

    bits_mean_train_avg = np.mean(np.array(one_bit_rand_noised_mean_pred_train[-1]), axis=1)
    bits_std_train_avg = np.mean(np.array(one_bit_rand_noised_std_pred_train[-1]), axis=1)
    mean_pred_wt_noised_train_avg = np.mean(one_bit_rand_mean_pred_train[-1], axis=0)
    std_pred_wt_noised_train_avg = np.mean(one_bit_rand_std_pred_train[-1], axis=0)

    bits_mean_test_avg = np.mean(np.array(one_bit_rand_noised_mean_pred_test[-1]), axis=1)
    bits_std_test_avg = np.mean(np.array(one_bit_rand_noised_std_pred_test[-1]), axis=1)
    mean_pred_wt_noised_test_avg = np.mean(one_bit_rand_mean_pred_test[-1], axis=0)
    std_pred_wt_noised_test_avg = np.mean(one_bit_rand_std_pred_test[-1], axis=0)

    # figure:
    fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')

    ax.set(xlabel='epochs', ylabel='accuracy (%)',
           title=(
               'Model: {}, at the epoch: {}. Influence of bits noised for all class prediction train'.format(expe_name,
                                                                                                             nb_epochs)))

    nb_bits = np.arange(len(bits_mean_train_avg))
    bits_len = len(bits_mean_train_avg)
    ax.errorbar(nb_bits, bits_mean_train_avg, yerr=bits_std_train_avg, fmt='o')
    ax.plot(nb_bits, np.repeat(mean_pred_wt_noised_train_avg, bits_len), label='all class avg train')
    ax.fill_between(nb_bits,
                    np.repeat(mean_pred_wt_noised_train_avg, bits_len) + np.repeat(std_pred_wt_noised_train_avg,
                                                                                   bits_len),
                    np.repeat(mean_pred_wt_noised_train_avg, bits_len) - np.repeat(std_pred_wt_noised_train_avg,
                                                                                   bits_len), alpha=0.5)

    ax.errorbar(nb_bits, bits_mean_test_avg, yerr=bits_std_test_avg, fmt='o')
    ax.plot(nb_bits, np.repeat(mean_pred_wt_noised_test_avg, bits_len), label='all class avg test')
    ax.fill_between(nb_bits, np.repeat(mean_pred_wt_noised_test_avg, bits_len) + np.repeat(std_pred_wt_noised_test_avg,
                                                                                           bits_len),
                    np.repeat(mean_pred_wt_noised_test_avg, bits_len) - np.repeat(std_pred_wt_noised_test_avg,
                                                                                  bits_len), alpha=0.5)

    ax.legend(loc=1)
    plt.show()

    return


def plot_stats_pred(one_bit_rand_mean_pred_train, one_bit_rand_mean_pred_test, \
                    one_bit_rand_std_pred_train, one_bit_rand_std_pred_test, one_bit_rand_noised_mean_pred_train,
                    one_bit_rand_noised_mean_pred_test, \
                    one_bit_rand_noised_std_pred_train, one_bit_rand_noised_std_pred_test, epochs, nb_epochs,
                    expe_name):
    """
    # to plot acc along training for each class:
    plot_Acc_each_class(one_bit_rand_mean_pred_train, one_bit_rand_std_pred_train, one_bit_rand_mean_pred_test,
                        one_bit_rand_std_pred_test, epochs, nb_epochs, expe_name)

    # to plot acc along training for all classes:
    plot_Acc_all_classes(one_bit_rand_mean_pred_train, one_bit_rand_std_pred_train, one_bit_rand_mean_pred_test,
                         one_bit_rand_std_pred_test, epochs, nb_epochs, expe_name)

    # plot for one representation with bit i noised:
    # plot_one_bit_noised_along_training(one_bit_rand_noised_mean_pred_train, one_bit_rand_noised_std_pred_train,
    #                                  one_bit_rand_noised_mean_pred_test, one_bit_rand_noised_std_pred_test, epochs,
    #                                  nb_epochs, expe_name)
    """

    # influence of each noised bit:
    plot_influence_bit_noised_classification(one_bit_rand_noised_mean_pred_train, one_bit_rand_noised_std_pred_train,
                                             one_bit_rand_mean_pred_train, one_bit_rand_std_pred_train,
                                             one_bit_rand_noised_mean_pred_test, one_bit_rand_noised_std_pred_test,
                                             one_bit_rand_mean_pred_test, one_bit_rand_std_pred_test,
                                             nb_epochs, expe_name)

    return


def plot_bits_stats(net, path, path_scores, expe_name):
    net, _, nb_epochs = get_checkpoints(net, path, expe_name)

    net, global_iter, epochs, Zc_Zd_train, Zc_random_Zd_train, Zc_Zd_random_train, Zc_pert_Zd_train, Zc_Zd_pert_train, \
    Zc_Zd_test, Zc_random_Zd_test, Zc_Zd_random_test, Zc_pert_Zd_test, Zc_Zd_pert_test, recon_loss_train_train, \
    kl_disc_loss_train, kl_cont_loss_train, total_kl_train, vae_loss_train, \
    classification_random_continue_loss_train, recon_loss_train_test, kl_disc_loss_test, kl_cont_loss_test, \
    total_kl_test, vae_loss_test, classification_random_continue_loss_test, classification_partial_rand_loss_train, \
    classification_partial_rand_loss_test, zvar_sim_loss_train, zvar_sim_loss_test, one_bit_rand_mean_pred_train, one_bit_rand_mean_pred_test, \
    one_bit_rand_std_pred_train, one_bit_rand_std_pred_test, one_bit_rand_noised_mean_pred_train, one_bit_rand_noised_mean_pred_test, \
    one_bit_rand_noised_std_pred_train, one_bit_rand_noised_std_pred_test = get_checkpoints_scores(net, path_scores,
                                                                                                   expe_name)

    plot_stats_pred(one_bit_rand_mean_pred_train, one_bit_rand_mean_pred_test, one_bit_rand_std_pred_train,
                    one_bit_rand_std_pred_test, one_bit_rand_noised_mean_pred_train, one_bit_rand_noised_mean_pred_test,
                    one_bit_rand_noised_std_pred_train, one_bit_rand_noised_std_pred_test, epochs, nb_epochs, expe_name)

    return


def plot_average_z_structural(net_trained, loader, device, nb_class, latent_spec, expe_name,
                              train_test='train', both_continue=True,
                              return_z_struct_representation=False, is_partial_rand_class=False, is_E1=False):
    """
    We average z_struct for all images of a specific class and decode it.
    We want to visualize the structural prototype for each classes.
    :return:
    """
    # 1) net eval mode: froward all images in E
    # 2) get z_struct and label associated for all data
    # 3) average representation z_struct per different classes
    # 4) decode each average representation with D (with random z_var)
    path = "fig_results/z_struct_2D_projections/fig_z_struct_2d_projection_per_classes_" + expe_name + train_test + ".png"

    if not os.path.exists(path) or return_z_struct_representation:
        # latent representation dimension:

        labels_list = []
        z_struct_representation = []
        average_representation_z_struct_class = []

        for data, labels in loader:

            # evaluation mode:
            net_trained.eval()

            with torch.no_grad():
                input_data = data
            if torch.cuda.is_available():
                input_data = input_data.cuda()

            _, _, _, _, latent_sample, latent_sample_variability, latent_sample_class, _, _, _, \
            _, _, _, _, _, _, _ = net_trained(input_data,
                                              reconstruction_rand=True,
                                              both_continue=both_continue,
                                              is_partial_rand_class=is_partial_rand_class,
                                              is_E1=is_E1)
            # train mode:
            net_trained.eval()

            labels_list.extend(labels.detach().numpy())
            z_struct_batch = latent_sample_class.detach().numpy()
            z_struct_representation.extend(z_struct_batch)

        z_struct_representation = np.array(z_struct_representation)  # shape: (nb_data, z_struct_dim)

        if return_z_struct_representation:
            return z_struct_representation, labels_list

        # average representation:
        for i in range(nb_class):
            z_struct_class = []
            for j in range(len(labels_list)):
                if labels_list[j] == i:
                    z_struct_class.append(z_struct_representation[j])
            z_struct_class = np.array(z_struct_class)
            average_z_struct_class = np.mean(z_struct_class, axis=0)
            average_representation_z_struct_class.append(average_z_struct_class)

        average_representation_z_struct_class = np.array(average_representation_z_struct_class)
        # shape: (nb_class, z_struct_dim)
        print('average_representation_z_struct_class shape: ', average_representation_z_struct_class.shape)

        np.save('structural_representation'
                '/average_representation_z_struct_class_' + expe_name + '_' +
                train_test + '.npy', average_representation_z_struct_class)
    else:
        average_representation_z_struct_class = np.load('structural_representation'
                                                        '/average_representation_z_struct_class_' + expe_name + '_' +
                                                        train_test + '.npy', allow_pickle=True)
        print('load average z_struct per class: {}'.format(average_representation_z_struct_class.shape))
    """
    # Decode:
    # build z with z_var rand and prototype:
    z_var_rand = torch.randn(latent_spec['cont_var'])
    z_var_rand = z_var_rand.to(device)

    # plot:
    images_arr = []
    labels_arr = np.arange(nb_class)

    for i in range(nb_class):
        latent = []
        z_struct_prototype = torch.tensor(average_representation_z_struct_class[i]).to(device)
        latent.append(z_var_rand)
        latent.append(z_struct_prototype)

        latent = torch.cat(latent, dim=0)

        prototype = net_trained._decode(latent).detach().numpy()
        images_arr.append(prototype[0][0])

    fig = plt.figure(figsize=(10, 5))
    plt.title('z_struct prototype: {} {}'.format(expe_name, train_test))
    plt.axis('off')
    for i in range(nb_class):
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(np.reshape(images_arr[i], (1, prototype[0].shape[-1], prototype[0].shape[-2]))[0], cmap='gray')
        ax.set_title("({})".format(labels_arr[i]))

    plt.show()
    """

    return


def plot_prototyoe_z_struct_per_class(nb_examples, average_representation_z_struct_class, train_test, expe_name,
                                      nb_class, latent_spec, device, net, z_var_size=None, save=False):
    """
    plot prototyoe z_struct per class
    :return:
    """

    for i in range(nb_examples):
        # plot:
        images_arr = []
        labels_arr = np.arange(nb_class)
        if latent_spec is None:
            z_var_zeros = torch.zeros(z_var_size)
            z_var_rand = z_var_zeros.to(device)
        else:
            # z_var_rand = torch.randn(latent_spec['cont_var'])
            z_var_zeros = torch.zeros(latent_spec['cont_var'])
            z_var_rand = z_var_zeros.to(device)

        for j in range(nb_class):
            latent = []
            z_struct_prototype = torch.tensor(average_representation_z_struct_class[j]).to(device)
            latent.append(z_var_rand)
            latent.append(z_struct_prototype)

            latent = torch.cat(latent, dim=0)
            prototype = net.decoder(latent).detach().numpy()
            images_arr.append(prototype[0][0])

        fig = plt.figure(figsize=(10, 5))
        plt.title('average z_struct per class, model: {}, on {} set'.format(expe_name, train_test))
        plt.axis('off')

        for k in range(nb_class):
            ax = fig.add_subplot(2, 5, k + 1, xticks=[], yticks=[])
            ax.imshow(np.reshape(images_arr[k], (1, prototype[0].shape[-1], prototype[0].shape[-2]))[0],
                      cmap='gray')
            ax.set_title("({})".format(labels_arr[k]))
        plt.show()

    if save:
        fig.savefig(
            "fig_results/protype_z_struct_class/fig_z_struct_prototype_per_classes_" + expe_name + train_test + ".png")

    return


def plot_2d_projection_z_struct(average_representation_z_struct_class, loader, net,
                                device, nb_class, latent_spec, expe_name, is_E1=False, is_partial_rand_class=False,
                                save=True, train_test=None):
    """
    we project z_struct per class in 2d dimension to see repartition:
    :return:
    """
    fig_path = "fig_results/z_struct_2D_projections/fig_z_struct_2d_projection_per_classes_" + expe_name + train_test + ".png"

    pca = PCA(n_components=2)
    reduced_average = pca.fit_transform(average_representation_z_struct_class)
    t_average = reduced_average.transpose()

    z_struct_representation, labels_list = plot_average_z_structural(net, loader, device, nb_class, latent_spec,
                                                                     expe_name,
                                                                     train_test='train', both_continue=True,
                                                                     return_z_struct_representation=True,
                                                                     is_partial_rand_class=is_partial_rand_class,
                                                                     is_E1=is_E1)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(z_struct_representation)
    t = reduced.transpose()

    fig, ax = plt.subplots()

    x = np.arange(nb_class)
    ys = [i + x + (i * x) ** 2 for i in range(10)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))

    plt.title('Projection z struct per class: {}'.format(expe_name))
    for c, color in zip(range(nb_class), colors):
        x = np.array(t[0])[np.concatenate(np.argwhere(np.array(labels_list) == c)).ravel()]
        y = np.array(t[1])[np.concatenate(np.argwhere(np.array(labels_list) == c)).ravel()]
        ax.scatter(x, y, alpha=0.6, color=color)
        ax.scatter(t_average[0][c], t_average[1][c], alpha=1, s=150, color=color)
    plt.show()

    if save:
        fig.savefig(
            "fig_results/z_struct_2D_projections/fig_z_struct_2d_projection_per_classes_" + expe_name + train_test + ".png")

    return


def plot_struct_fixe_and_z_var_moove(average_representation_z_struct_class, train_test, net_trained,
                                     device, nb_class, latent_spec, expe_name, z_var_size=None,
                                     embedding_size=None, save=True, mu_var=None, std_var=None):
    """
    Here we see traversal reconstruction of z_struct prototype wirth z_struct fixed and variable z_rand.
    We should observed always the same class (for the same prototype) but with different variability.
    :return:
    """

    nb_examples = 10
    latent_samples = []
    all_latent = []
    samples = []
    """
    for i in range(nb_class):
        z_struct_prototype = torch.tensor(average_representation_z_struct_class[i]).to(device)
        z_var_rand = torch.zeros(latent_spec['cont_var'])
        z_var_rand = z_var_rand.to(device)
        latent = []
        latent.append(z_var_rand)
        latent.append(z_struct_prototype)
        latent = torch.cat(latent, dim=0)
        all_latent.append(torch.tensor(latent))
        for j in range(nb_examples):
            z_var_rand = torch.randn(latent_spec['cont_var'])
            z_var_rand = z_var_rand.to(device)

            latent = []
            latent.append(z_var_rand)
            latent.append(z_struct_prototype)
            latent = torch.cat(latent, dim=0)
            all_latent.append(torch.tensor(latent))
    """
    z_struct_prototype = torch.tensor(average_representation_z_struct_class).to(device)  # shape: (nb_class, struct_dim)
    z_struct_prototype = np.expand_dims(z_struct_prototype, axis=0)  # shape: (1, nb_class, struct_dim)
    z_struct_prototype = torch.tensor(np.repeat(z_struct_prototype, nb_examples + 1, axis=0))
    # shape: (nb_example+1, nb_class, struct_dim)
    if latent_spec is None:
        z_var_zeros = torch.zeros((1, nb_class, z_var_size))  # shape: (1, nb_class, var_dim)
    else:
        z_var_zeros = torch.zeros((1, nb_class, latent_spec['cont_var']))  # shape: (1, nb_class, var_dim)
    z_var_zeros = z_var_zeros.to(device)

    if latent_spec is None:
        z_var_rand = std_var * torch.randn((1, nb_examples, z_var_size)) + mu_var
    else:
        z_var_rand = std_var * torch.randn((1, nb_examples, latent_spec['cont_var'])) + mu_var  # shape: (nb_examples, var_dim)
    z_var_rand = torch.tensor(np.repeat(z_var_rand, nb_class, axis=0))  # shape: (nb_class, nb_examples, var_dim)
    z_var_rand = z_var_rand.permute(1, 0, 2)  # shape: (nb_examples, nb_class, var_dim)
    z_var_rand = z_var_rand.to(device)

    latent_zeros = []
    latent_zeros.append(z_var_zeros[0])
    latent_zeros.append(z_struct_prototype[0, :])
    latent_zeros = torch.cat(latent_zeros, dim=1)  # shape: (nb_classes, z_dim)
    all_latent.append(torch.tensor(latent_zeros))  # we add the first column: original z_struct with zeros z_var

    for i in range(nb_examples):
        latent_rand = []
        latent_rand.append(z_var_rand[i])
        latent_rand.append(z_struct_prototype[i + 1])
        latent_rand = torch.cat(latent_rand, dim=1)
        all_latent.append(torch.tensor(latent_rand))

    all_latent = torch.Tensor(np.array([t.numpy() for t in all_latent])).permute(1, 0, 2)

    if latent_spec is None:
        samples.append(all_latent.reshape((nb_class * (nb_examples + 1), embedding_size)))
    else:
        samples.append(all_latent.reshape((nb_class * (nb_examples + 1), latent_spec['cont_var'] +
                                           latent_spec['cont_class'])))
    latent_samples.append(torch.cat(samples, dim=1))
    generated = net_trained.decoder(torch.cat(latent_samples, dim=0))
    prototype = make_grid(generated.data, nrow=nb_examples + 1)

    fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
    traversals = prototype.permute(1, 2, 0)
    ax.set(title=('Prototype per classes with rand zvar examples: {}'.format(expe_name)))

    fig_size = prototype.shape
    plt.imshow(traversals.numpy())

    ax.axvline(x=1.5, linewidth=3, color='orange')
    ax.axvline(x=(fig_size[1] // nb_class), linewidth=3, color='orange')
    plt.show()

    if save:
        fig.savefig(
            "fig_results/struct_fixe_zvar_moove/fig_z_struct_fixe_zvar_moove_" + expe_name + train_test + ".png")

    return


def plot_var_fixe_and_z_struct_moove(average_representation_z_struct_class, train_test, net_trained,
                                     device, nb_class, latent_spec, expe_name, z_struct_size=None, z_var_size=None,
                                     embedding_size=None, save=True, mu_struct=None, std_struct=None):
    """
    Here we see traversal reconstruction of z_struct prototype over z_struct with z_var fixed.
    We should observed always the same variability along the generated data but with a class who slightly change
    :return:
    """

    nb_examples = 10
    latent_samples = []
    all_latent = []
    samples = []

    z_struct_prototype = torch.tensor(average_representation_z_struct_class).to(device)  # shape: (nb_class, struct_dim)
    z_struct_prototype = torch.tensor(np.expand_dims(z_struct_prototype, axis=0))  # shape: (1, nb_class, struct_dim)

    z_var_zeros = torch.zeros((1, nb_class, z_var_size))  # shape: (nb_examples, nb_class, z_var_size)
    z_var_zeros = z_var_zeros.to(device)

    z_struct_rand = std_struct * torch.randn((1, nb_examples, z_struct_size)) + mu_struct
    z_struct_rand = torch.tensor(np.repeat(z_struct_rand, nb_class, axis=0))  # shape: (nb_class, nb_examples, z_struct_size)
    z_struct_rand = z_struct_rand.permute(1, 0, 2)  # shape: (nb_examples, nb_class, var_dim)
    z_struct_rand = z_struct_rand.to(device)

    latent_zeros = []
    latent_zeros.append(z_var_zeros[0])
    latent_zeros.append(z_struct_prototype[0, :])
    latent_zeros = torch.cat(latent_zeros, dim=1)  # shape: (nb_classes, z_dim)
    all_latent.append(torch.tensor(latent_zeros))  # we add the first column: original z_struct with zeros z_var

    for i in range(nb_examples):
        latent_rand = []
        latent_rand.append(z_var_zeros[0])
        latent_rand.append(z_struct_rand[i])
        latent_rand = torch.cat(latent_rand, dim=1)
        all_latent.append(torch.tensor(latent_rand))

    all_latent = torch.Tensor(np.array([t.numpy() for t in all_latent])).permute(1, 0, 2)

    samples.append(all_latent.reshape((nb_class * (nb_examples + 1), embedding_size)))

    latent_samples.append(torch.cat(samples, dim=1))
    generated = net_trained.decoder(torch.cat(latent_samples, dim=0))
    prototype = make_grid(generated.data, nrow=nb_examples + 1)

    fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
    traversals = prototype.permute(1, 2, 0)
    ax.set(title=('zvar=0 and z_struct random: {}'.format(expe_name)))

    fig_size = prototype.shape
    plt.imshow(traversals.numpy())

    ax.axvline(x=1.5, linewidth=3, color='orange')
    ax.axvline(x=(fig_size[1] // nb_class), linewidth=3, color='orange')
    plt.show()

    if save:
        fig.savefig("fig_results/var_fixe_zstruct_moove/fig_zvar_fixe_zstruct_moove_" + expe_name + train_test + ".png")

    return


def plot_prototype(net, expe_name, nb_class, latent_spec, device, loader, nb_examples=1, train_test='train',
                   print_per_class=True, print_per_var=False, plot_traversal_struct=False, size=8, bit=0,
                   print_2d_projection=False, is_partial_rand_class=False, is_E1=False, save=True):
    # TODO: find objective criterion to measure quality image generation and diversity of generate images

    path = 'structural_representation/average_representation_z_struct_class_' + expe_name + '_' + train_test + '.npy'
    assert os.path.exists(path), "path doesn't exist"

    average_representation_z_struct_class = np.load(path, allow_pickle=True)
    print('load average z_struct per class {}: {}'.format(train_test, average_representation_z_struct_class.shape))

    if print_per_class:
        plot_prototyoe_z_struct_per_class(nb_examples, average_representation_z_struct_class, train_test, expe_name,
                                          nb_class, latent_spec, device, net, save=save)

    if print_2d_projection:
        plot_2d_projection_z_struct(average_representation_z_struct_class, loader, net, device, nb_class, latent_spec,
                                    expe_name, is_E1=is_E1, is_partial_rand_class=is_partial_rand_class,
                                    train_test=train_test, save=save)

    if print_per_var:
        plot_struct_fixe_and_z_var_moove(average_representation_z_struct_class, train_test, net,
                                         device, nb_class, latent_spec, expe_name, save=save)

    if plot_traversal_struct:
        plot_var_fixe_and_z_struct_moove(average_representation_z_struct_class, nb_class, latent_spec, device, size,
                                         net, expe_name, train_test, save=save)

    return


def plot_images_taversal(net, expe_name, latent_spec, batch, nb_epochs, path, img_size, size=8, save=True,
                         z_component_traversal=None, is_partial_rand_class=False, is_E1=False,
                         indx_image=None):
    """
    
    :param net: 
    :param expe_name: 
    :param latent_spec: 
    :param batch: 
    :param nb_epochs: 
    :param path: 
    :param img_size: 
    :param size: 
    :param save: 
    :param z_component_traversal: index of the components which we want to see traversal
    :param is_partial_rand_class: 
    :param is_E1: 
    :param indx_image: index to choose an image in the batch, if None, we choose a random index between 0 and size 
    of the batch 
    :return: 
    """
    nb_examples = 1
    for example in range(nb_examples):
        latent_real_img_traversal(net, nb_epochs, path, expe_name, latent_spec, batch, img_size, indx_image=indx_image,
                                  z_component_traversal=z_component_traversal,
                                  both_continue=True, is_partial_rand_class=is_partial_rand_class, is_E1=is_E1,
                                  size=size, save=save)

    return


def traversal_values(size):
    cdf_traversal = np.linspace(0.05, 0.95, size)

    return stats.norm.ppf(cdf_traversal)


def plot_weights_values_z_struct(net_trained, latent_spec):
    weights_z_struct = net_trained.L3_classifier[0].weight[:, latent_spec['cont_var']:]
    print('weights of z_struct: ', weights_z_struct.shape)
    sum_z_struct = torch.sum(weights_z_struct, axis=0)
    print('sum z_struct coposante: ', sum_z_struct.shape)

    x = np.arange(latent_spec['cont_class'])
    fig, ax = plt.subplots()
    plt.bar(x, torch.abs(sum_z_struct).detach().numpy())
    plt.show()
    return


def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


def heatmap_vector(model_classifier, latent_sample, latent_dim, cond_vals, captum=False):
    """
    here output is z_struct vector of an image
    :param model:
    :param output:
    :return:
    """
    if captum:
        activations = cond_vals
    else:
        activation = {}
        for name, m in model_classifier.named_modules():
            m.register_forward_hook(get_activation(name, activation))

        output = model_classifier(latent_sample)

        # we want to weighted W by activation to see the 25 z_struct composante force:
        activations = activation['0']  # shape (batch, 256)
        # print('activation shape for the first layer: {}'.format(activations.shape))
        # multiply W by activation for each z_struct composante:
        activations = activations.detach().numpy()

    weights_z = model_classifier[0].weight  # shape: (256, latent_dim)
    # print('weights shape: {}'.format(weights_z_struct.shape))

    composante_z = []

    for i in range(latent_dim):
        weights = weights_z[:, i].detach().numpy()  # weigths for composante i of z_struct: shape (256)
        element_wise = np.multiply(weights, activations)  # elementwise product for all batch: shape (batch, 256)
        composante = np.sum(element_wise, axis=1)  # shape: nb_batch
        composante_z.append(composante)

    composante_z_struct = np.array(np.moveaxis(composante_z, 0, -1))  # shape (z_struct_dim, nb_batch)

    return composante_z_struct, activations, weights_z


def compute_heatmap_avg(loader, model, latent, expe_name, train_test, nb_class, save=False, captum=False,
                        is_partial_rand_class=False, is_E1=False):
    path = 'structural_representation/average_representation_z_struct_class_heatmap_' + expe_name + '_' + train_test + '.npy'
    if os.path.exists(path):
        print("this heatmap avg is already computed")
        return

    labels_list = []
    z_struct_heatmap = []
    latent_dim = latent['cont_class'] + latent['cont_var']

    if captum:
        cond = LayerConductance(model.L3_classifier, model.L3_classifier[0])

    for data, labels in loader:
        # evaluation mode:
        model.eval()

        with torch.no_grad():
            input_data = data
        if torch.cuda.is_available():
            input_data = input_data.cuda()

        _, _, _, _, latent_sample, latent_sample_variability, latent_sample_class, _, _, _, \
        _, _, _, _, _, _, _ = model(input_data,
                                    reconstruction_rand=True,
                                    both_continue=True,
                                    is_partial_rand_class=is_partial_rand_class,
                                    is_E1=is_E1)

        # train mode:
        model.train()

        if captum:
            cond_vals = cond.attribute(latent_sample, target=1)
            cond_vals = cond_vals.detach().numpy()  # shape: (nb_batch, 256)
        else:
            cond_vals = None

        heatmap_vector_z_struct, activations, weights_z = heatmap_vector(model.L3_classifier, latent_sample, latent_dim,
                                                                         cond_vals, captum=captum)

        labels_list.extend(labels.detach().numpy())
        z_struct_heatmap.extend(heatmap_vector_z_struct)

    z_struct_heatmap = np.array(z_struct_heatmap)  # shape: (nb_data, latent_dim)
    labels_list = np.array(labels_list)  # shape (nb_data, )

    # now we want to avg above representation by classes:
    average_representation_z_struct_class_heatmap = []

    if save:
        for i in range(nb_class):
            z_struct_weights = []
            for j in range(len(labels_list)):
                if labels_list[j] == i:
                    z_struct_weights.append(z_struct_heatmap[j])
            z_struct_weights = np.array(z_struct_weights)
            average_z_struct_class_heatmap = np.mean(z_struct_weights, axis=0)
            average_representation_z_struct_class_heatmap.append(average_z_struct_class_heatmap)

        average_representation_z_struct_class_heatmap = np.array(average_representation_z_struct_class_heatmap)
        # shape: (nb_class, z_struct_dim)
        np.save('structural_representation'
                '/average_representation_z_struct_class_heatmap_' + expe_name + '_' +
                train_test + '.npy', average_representation_z_struct_class_heatmap)
        return
    else:
        return z_struct_heatmap, activations, weights_z, labels_list


def plot_heatmap_avg(expe_name, latent, all_classes_details=False, all_classes_resum=True, train_test=None):
    path_train = 'structural_representation/average_representation_z_struct_class_heatmap_' + expe_name + '_' + train_test + '.npy'
    assert os.path.exists(path_train), "path doesn't exist"

    average_representation_z_struct_class_heatmap_train = np.load(path_train, allow_pickle=True)
    print('load average z_struct heatmap per class train: {}'.format(
        average_representation_z_struct_class_heatmap_train.shape))

    # plot 1D heatmap by class:
    nb_class = average_representation_z_struct_class_heatmap_train.shape[0]

    latent_dim = latent['cont_class'] + latent['cont_var']

    if all_classes_details:
        for i in range(nb_class):
            plt.rcParams["figure.figsize"] = 5, 2

            x = np.linspace(0, latent_dim, latent_dim)
            y_train = np.cumsum(average_representation_z_struct_class_heatmap_train[i])

            fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True)
            plt.title('Heatmap for class: {}, and for {} set: {}'.format(train_test, expe_name))

            extent = [x[0] - (x[1] - x[0]) / 2., x[-1] + (x[1] - x[0]) / 2., 0, 1]
            ax.imshow(y_train[np.newaxis, :], cmap="plasma", aspect="auto", extent=extent)
            ax.set_yticks([])
            ax.set_xlim(extent[0], extent[1])

            ax2.plot(x, y_train)

            plt.tight_layout()
            plt.show()

    if all_classes_resum:
        fig, ax = plt.subplots(figsize=(10, 2), facecolor='w', edgecolor='k')
        plt.title('Heatmap for all classes and for train/test set: {}'.format(expe_name))
        for i in range(nb_class):
            x = np.linspace(0, latent_dim, latent_dim)
            y_train = np.cumsum(average_representation_z_struct_class_heatmap_train[i])
            ax.plot(x, y_train)
        plt.show()

    return


# Helper method to print importances and visualize distribution
def visualize_importances(feature_names, importances, title="Average Feature Importances", plot=True,
                          axis_title="Features"):
    x_pos = (np.arange(len(feature_names)))

    plt.figure(figsize=(12, 6))
    plt.bar(x_pos, importances, align='center')
    plt.xticks(x_pos, feature_names, wrap=True)
    plt.xlabel(axis_title)
    plt.title(title)
    plt.show()

    return


def compute_importance_neuron(model, loader, both_continue=True, is_partial_rand_class=False, is_E1=False):
    cond = LayerConductance(model.L3_classifier, model.L3_classifier[0])

    for data, labels in loader:
        # evaluation mode:
        model.eval()

        with torch.no_grad():
            input_data = data
        if torch.cuda.is_available():
            input_data = input_data.cuda()

        _, _, _, _, latent_sample, latent_sample_variability, latent_sample_class, _, _, _, \
        _, _, _, _, _, _, _ = model(input_data[0].unsqueeze(dim=0),
                                    reconstruction_rand=True,
                                    both_continue=both_continue,
                                    is_partial_rand_class=is_partial_rand_class,
                                    is_E1=is_E1)

        # train mode:
        model.train()

        cond_vals = cond.attribute(latent_sample, target=1)
        cond_vals = cond_vals.detach().numpy()  # shape: (nb_batch, 256)

        visualize_importances(range(256), np.mean(cond_vals, axis=0), title="Average Neuron Importances",
                              axis_title="Neurons")
        break

    return
