import torch
from model import BetaVAE
from dataset.dataset_2 import get_dataloaders

from visualizer import *


def compute_heatmap(net_trained, train_loader, test_loader, latent_spec, device, expe_name):
    compute_heatmap_avg(train_loader, net_trained, latent_spec, device, expe_name, 'train', save=True, captum=False,
                        is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)
    compute_heatmap_avg(train_loader, net_trained, latent_spec, device, expe_name, 'train_captum', save=True,
                        captum=True,
                        is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)
    compute_heatmap_avg(test_loader, net_trained, latent_spec, device, expe_name, 'test', save=True, captum=False,
                        is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)
    compute_heatmap_avg(test_loader, net_trained, latent_spec, device, expe_name, 'test_captum', save=True, captum=True,
                        is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)

    return


def visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=None,
              path_scores=None, batch=None, img_size=None, indx_image=None, path=None, losses=True,
              FID=False, IS=False, psnr=False, scores=True, all_prototype=False, copute_average_z_structural=False,
              is_partial_rand_class=False, all_classes_resum=True, save=False, scores_and_losses=False,
              heatmap=False, prototype=False, all_classes_details=False, project_2d=False, is_E1=False,
              reconstruction=False, plot_img_traversal=False, z_component_traversal=None, plot_sample=False):
    if scores_and_losses:
        plot_scores_and_loss(net, expe_name, path_scores, save=save, partial_rand=is_partial_rand_class, losses=losses,
                             scores=scores)

    if reconstruction:
        viz_reconstruction(net, nb_epochs, expe_name, batch, latent_spec, img_size,
                           is_partial_rand_class=is_partial_rand_class,
                           partial_reconstruciton=True, is_E1=is_E1, save=save)

    if heatmap:
        plot_heatmap_avg(expe_name, latent_spec, all_classes_details=all_classes_details,
                         all_classes_resum=all_classes_resum,
                         train_test='train')
        plot_heatmap_avg(expe_name, latent_spec, all_classes_details=all_classes_details,
                         all_classes_resum=all_classes_resum,
                         train_test='train_captum')
        plot_heatmap_avg(expe_name, latent_spec, all_classes_details=all_classes_details,
                         all_classes_resum=all_classes_resum,
                         train_test='test')
        plot_heatmap_avg(expe_name, latent_spec, all_classes_details=all_classes_details,
                         all_classes_resum=all_classes_resum,
                         train_test='test_captum')

    if prototype:
        plot_prototype(net, expe_name, nb_class, latent_spec, device, test_loader, train_test='test',
                       print_per_class=True, print_per_var=False,
                       plot_traversal_struct=False, is_partial_rand_class=is_partial_rand_class, save=save)
        plot_prototype(net, expe_name, nb_class, latent_spec, device, train_loader, train_test='train',
                       print_per_class=True, print_per_var=False,
                       plot_traversal_struct=False, is_partial_rand_class=is_partial_rand_class, save=save)

    if project_2d:
        plot_prototype(net, expe_name, nb_class, latent_spec, device, train_loader, train_test='train',
                       print_per_class=False, print_per_var=False,
                       plot_traversal_struct=False, print_2d_projection=True,
                       is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)
        plot_prototype(net, expe_name, nb_class, latent_spec, device, test_loader, train_test='test',
                       print_per_class=False, print_per_var=False,
                       plot_traversal_struct=False, print_2d_projection=True,
                       is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)

    if plot_img_traversal:
        plot_images_taversal(net, expe_name, latent_spec, batch, nb_epochs, path, img_size, indx_image=indx_image,
                             size=8, save=True, z_component_traversal=z_component_traversal,
                             is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)

    if plot_sample:
        plot_samples(net, nb_epochs, path, expe_name, latent_spec, img_size, batch=batch, both_continue=True, save=save,
                     FID=FID, IS=IS, psnr=psnr)

    if all_prototype:
        plot_prototype(net, expe_name, nb_class, latent_spec, device, train_loader, train_test='train',
                       print_per_class=True, print_per_var=True,
                       plot_traversal_struct=True, print_2d_projection=True,
                       is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)
        plot_prototype(net, expe_name, nb_class, latent_spec, device, test_loader, train_test='test',
                       print_per_class=True, print_per_var=True,
                       plot_traversal_struct=True, print_2d_projection=True,
                       is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)

    if copute_average_z_structural:
        plot_average_z_structural(net_trained, train_loader, device, nb_class, latent_spec, expe_name,
                                  train_test='train', both_continue=True,
                                  is_partial_rand_class=is_partial_rand_class, is_E1=is_E1, save=save)
        plot_average_z_structural(net_trained, test_loader, device, nb_class, latent_spec, expe_name,
                                  train_test='test', both_continue=True,
                                  is_partial_rand_class=is_partial_rand_class, is_E1=is_E1, save=save)

    return


device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch = torch.load('data/batch_mnist.pt')
path = 'checkpoints/'
path_scores = 'checkpoints_scores'

# load mnist dataset:
train_loader, valid_loader, test_loader = get_dataloaders('mnist', batch_size=64)

img_size = (1, 32, 32)
nb_class = 10
nb_samples = 10
size = 8
nc = 1
four_conv = False
is_C = True
save = False

L3_without_random = False
is_binary_structural_latent = False

# for traversal real image:
indx_image = 0

# _______________________________Expe test vanilla VAE + Class + E1 + zvar_sim_____________________________________


mnist_VAE_class_E1_zvar_sim_5_5 = ['VAE_class_E1_MSE_41']

mnist_VAE_class_E1_zvar_sim_5_5_tune_lr = ['VAE_Cass_E1_Zvarsim_tune_WLr_1', 'VAE_Cass_E1_Zvarsim_tune_WLr_2',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_3', 'VAE_Cass_E1_Zvarsim_tune_WLr_4',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_5', 'VAE_Cass_E1_Zvarsim_tune_WLr_6',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_7', 'VAE_Cass_E1_Zvarsim_tune_WLr_8',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_9', 'VAE_Cass_E1_Zvarsim_tune_WLr_10',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_11', 'VAE_Cass_E1_Zvarsim_tune_WLr_12',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_13', 'VAE_Cass_E1_Zvarsim_tune_WLr_14',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_15', 'VAE_Cass_E1_Zvarsim_tune_WLr_16',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_17', 'VAE_Cass_E1_Zvarsim_tune_WLr_18',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_19', 'VAE_Cass_E1_Zvarsim_tune_WLr_20'
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_21', 'VAE_Cass_E1_Zvarsim_tune_WLr_22',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_23', 'VAE_Cass_E1_Zvarsim_tune_WLr_24',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_25', 'VAE_Cass_E1_Zvarsim_tune_WLr_26',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_27', 'VAE_Cass_E1_Zvarsim_tune_WLr_28',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_29', 'VAE_Cass_E1_Zvarsim_tune_WLr_30'
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_31', 'VAE_Cass_E1_Zvarsim_tune_WLr_32',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_33', 'VAE_Cass_E1_Zvarsim_tune_WLr_34',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_35', 'VAE_Cass_E1_Zvarsim_tune_WLr_36',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_37', 'VAE_Cass_E1_Zvarsim_tune_WLr_38',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_39', 'VAE_Cass_E1_Zvarsim_tune_WLr_40'
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_41', 'VAE_Cass_E1_Zvarsim_tune_WLr_42',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_43', 'VAE_Cass_E1_Zvarsim_tune_WLr_44',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_45', 'VAE_Cass_E1_Zvarsim_tune_WLr_46',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_47', 'VAE_Cass_E1_Zvarsim_tune_WLr_48',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_49', 'VAE_Cass_E1_Zvarsim_tune_WLr_50'
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_51', 'VAE_Cass_E1_Zvarsim_tune_WLr_52',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_53', 'VAE_Cass_E1_Zvarsim_tune_WLr_54',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_55', 'VAE_Cass_E1_Zvarsim_tune_WLr_56',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_57', 'VAE_Cass_E1_Zvarsim_tune_WLr_58',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_59', 'VAE_Cass_E1_Zvarsim_tune_WLr_60',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_61', 'VAE_Cass_E1_Zvarsim_tune_WLr_62', 
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_63', 'VAE_Cass_E1_Zvarsim_tune_WLr_64', 
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_65']

mnist_VAE_class_E1_zvar_sim_5_5_tune_lr = ['VAE_Cass_E1_Zvarsim_tune_WLr_111', 'VAE_Cass_E1_Zvarsim_tune_WLr_112',
                                           'VAE_Cass_E1_Zvarsim_tune_WLr_113']

mnist_VAE_class_E1_zvar_sim_best = ['VAE_class_E1_MSE_41', 'VAE_Cass_E1_Zvarsim_tune_WLr_25',
                                    'VAE_Cass_E1_Zvarsim_tune_WLr_112']

is_zvar_sim_loss = True
is_partial_rand_class = False
is_E1 = True
E1_conv = True
is_C = True

# for traversal real image:
indx_image = 0

# _____________ VAE 5 5 + class + E1 + zvar_sim________________
latent_spec = {'cont_var': 5, 'cont_class': 5}
BN = True
second_layer_C = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
for expe in mnist_VAE_class_E1_zvar_sim_best:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
    #           is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
    #           save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
    #           batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
    #           save=True, is_E1=is_E1, reconstruction=True)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
    #           batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
    #           is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader,
    #          copute_average_z_structural=True, is_partial_rand_class=is_partial_rand_class, save=True,
    #          is_E1=is_E1)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, all_prototype=True,
              is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1)
    break


"""
# _______________________________Expe test vanilla VAE _____________________________________

mnist_VAE_5_5_32_32_32 = ['TEst_Vanilla_VAE_1_']
mnist_VAE_5_5_16_16_32 = ['TEst_Vanilla_VAE_2_']
mnist_VAE_5_5_8_8_32 = ['TEst_Vanilla_VAE_3_']
mnist_VAE_5_5_64_64_32 = ['TEst_Vanilla_VAE_4_']
mnist_VAE_5_5_8_16_32 = ['TEst_Vanilla_VAE_5_']
mnist_VAE_5_5_16_32_32 = ['TEst_Vanilla_VAE_6_']
mnist_VAE_5_5_32_64_32 = ['TEst_Vanilla_VAE_7_']
mnist_VAE_5_5_16_64_32 = ['TEst_Vanilla_VAE_8_']
mnist_VAE_5_5_8_64_32 = ['TEst_Vanilla_VAE_9_']
mnist_VAE_5_5_32_64_32 = ['TEst_Vanilla_VAE_10_']

mnist_VAE_5_10 = ['VAE_5_10_2']
mnist_VAE_5_15 = ['VAE_5_15_2']
mnist_VAE_5_20 = ['VAE_5_20_3']
mnist_VAE_5_25 = ['VAE_5_25_6']
mnist_VAE_5_30 = ['VAE_5_30_4']

both_continue = True
second_layer_C = False
is_zvar_sim_loss = False
is_partial_rand_class = False
is_E1 = False
is_C = False

# _____________ VAE 5 5 (32, 32, 32)________________
latent_spec = {'cont_var': 5, 'cont_class': 5}
z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
BN = True
hidden_filters_1 = 32
hidden_filters_2 = 32
hidden_filters_3 = 32
stride_size = 2
kernel_size = 4

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN, hidden_filters_1=hidden_filters_1, hidden_filters_2=hidden_filters_2,
              hidden_filters_3=hidden_filters_3, stride_size=stride_size, kernel_size=kernel_size)

for expe in mnist_VAE_5_5_32_32_32:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True,
              scores=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)

# _____________ VAE 5 5 (16, 16, 32)________________
latent_spec = {'cont_var': 5, 'cont_class': 5}
z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
BN = True
hidden_filters_1 = 16
hidden_filters_2 = 16
hidden_filters_3 = 32

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN, hidden_filters_1=hidden_filters_1, hidden_filters_2=hidden_filters_2,
              hidden_filters_3=hidden_filters_3, stride_size=stride_size, kernel_size=kernel_size)

for expe in mnist_VAE_5_5_16_16_32:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True,
              scores=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)

# _____________ VAE 5 5 (8, 8, 32)________________
latent_spec = {'cont_var': 5, 'cont_class': 5}
z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
BN = True
hidden_filters_1 = 8
hidden_filters_2 = 8
hidden_filters_3 = 32

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN, hidden_filters_1=hidden_filters_1, hidden_filters_2=hidden_filters_2,
              hidden_filters_3=hidden_filters_3, stride_size=stride_size, kernel_size=kernel_size)

for expe in mnist_VAE_5_5_8_8_32:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True,
              scores=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)

# _____________ VAE 5 5 (64, 64, 32)________________
latent_spec = {'cont_var': 5, 'cont_class': 5}
z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
BN = True
hidden_filters_1 = 64
hidden_filters_2 = 64
hidden_filters_3 = 32

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN, hidden_filters_1=hidden_filters_1, hidden_filters_2=hidden_filters_2,
              hidden_filters_3=hidden_filters_3, stride_size=stride_size, kernel_size=kernel_size)

for expe in mnist_VAE_5_5_64_64_32:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True,
              scores=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)

# _____________ VAE 5 5 (8, 16, 32)________________
latent_spec = {'cont_var': 5, 'cont_class': 5}
z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
BN = True
hidden_filters_1 = 8
hidden_filters_2 = 16
hidden_filters_3 = 32

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN, hidden_filters_1=hidden_filters_1, hidden_filters_2=hidden_filters_2,
              hidden_filters_3=hidden_filters_3, stride_size=stride_size, kernel_size=kernel_size)

for expe in mnist_VAE_5_5_8_16_32:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True,
              scores=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)

# _____________ VAE 5 5 (16, 32, 32)________________
latent_spec = {'cont_var': 5, 'cont_class': 5}
z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
BN = True
hidden_filters_1 = 16
hidden_filters_2 = 32
hidden_filters_3 = 32

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN, hidden_filters_1=hidden_filters_1, hidden_filters_2=hidden_filters_2,
              hidden_filters_3=hidden_filters_3, stride_size=stride_size, kernel_size=kernel_size)

for expe in mnist_VAE_5_5_16_32_32:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True,
              scores=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)

# _____________ VAE 5 5 (32, 64, 32)________________
latent_spec = {'cont_var': 5, 'cont_class': 5}
z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
BN = True
hidden_filters_1 = 32
hidden_filters_2 = 64
hidden_filters_3 = 32

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN, hidden_filters_1=hidden_filters_1, hidden_filters_2=hidden_filters_2,
              hidden_filters_3=hidden_filters_3, stride_size=stride_size, kernel_size=kernel_size)

for expe in mnist_VAE_5_5_32_64_32:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True,
              scores=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)

# _____________ VAE 5 5 (16, 64, 32)________________
latent_spec = {'cont_var': 5, 'cont_class': 5}
z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
BN = True
hidden_filters_1 = 16
hidden_filters_2 = 64
hidden_filters_3 = 32

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN, hidden_filters_1=hidden_filters_1, hidden_filters_2=hidden_filters_2,
              hidden_filters_3=hidden_filters_3, stride_size=stride_size, kernel_size=kernel_size)

for expe in mnist_VAE_5_5_16_64_32:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True,
              scores=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)

# _____________ VAE 5 5 (8, 64, 32)________________
latent_spec = {'cont_var': 5, 'cont_class': 5}
z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
BN = True
hidden_filters_1 = 8
hidden_filters_2 = 64
hidden_filters_3 = 32

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN, hidden_filters_1=hidden_filters_1, hidden_filters_2=hidden_filters_2,
              hidden_filters_3=hidden_filters_3, stride_size=stride_size, kernel_size=kernel_size)

for expe in mnist_VAE_5_5_8_64_32:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True,
              scores=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)

# _____________ VAE 5 5 (32, 64, 32)________________
latent_spec = {'cont_var': 5, 'cont_class': 5}
z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
BN = True
hidden_filters_1 = 32
hidden_filters_2 = 64
hidden_filters_3 = 32

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN, hidden_filters_1=hidden_filters_1, hidden_filters_2=hidden_filters_2,
              hidden_filters_3=hidden_filters_3, stride_size=stride_size, kernel_size=kernel_size)

for expe in mnist_VAE_5_5_32_64_32:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True,
              scores=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)

# _____________ VAE 5 10 ________________
latent_spec = {'cont_var': 5, 'cont_class': 10}
BN = True
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN)

for expe in mnist_VAE_5_10:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True,
              scores=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)


# _____________ VAE 5 15 ________________
latent_spec = {'cont_var': 5, 'cont_class': 15}
BN = True
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN)

for expe in mnist_VAE_5_15:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True,
              scores=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)


# _____________ VAE 5 20 ________________
latent_spec = {'cont_var': 5, 'cont_class': 20}
BN = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN)

for expe in mnist_VAE_5_20:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True,
              scores=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)

# _____________ VAE 5 25 ________________
latent_spec = {'cont_var': 5, 'cont_class': 25}
BN = True
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN)

for expe in mnist_VAE_5_25:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True,
              scores=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)

# _____________ VAE 5 30 ________________
latent_spec = {'cont_var': 5, 'cont_class': 30}
BN = True
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN)

for expe in mnist_VAE_5_30:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True,
              scores=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)

# _______________________________Expe test vanilla VAE + class_____________________________________

mnist_VAE_class_5_5 = ['VAE_class_5_5_5']
mnist_VAE_class_5_10 = ['VAE_class_5_10_7']
mnist_VAE_class_5_15 = ['VAE_class_5_15_7']

both_continue = True
second_layer_C = False
is_zvar_sim_loss = False
is_partial_rand_class = False
is_E1 = False
is_C = True

# _____________ VAE 5 5 + class________________
latent_spec = {'cont_var': 5, 'cont_class': 5}
z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
BN = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN)

for expe in mnist_VAE_class_5_5:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)

# _____________ VAE 5 10 + class________________
latent_spec = {'cont_var': 5, 'cont_class': 10}
z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
BN = True
second_layer_C = True
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN)

for expe in mnist_VAE_class_5_10:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)

# _____________ VAE 5 15 + class________________
latent_spec = {'cont_var': 5, 'cont_class': 15}
z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
BN = True
second_layer_C = True
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN)

for expe in mnist_VAE_class_5_15:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)

# _______________________________Expe test VAE + E1_____________________________________

mnist_VAE_class_E1_5_5 = ['VAE_class_E1_4']
mnist_VAE_class_E1_5_10 = ['VAE_class_E1_5_10_1']
mnist_VAE_class_E1_5_15 = ['VAE_class_E1_5_15_3']

is_zvar_sim_loss = False
is_partial_rand_class = False
is_C = True

# _____________ VAE 5 5 + class + E1 ________________
latent_spec = {'cont_var': 5, 'cont_class': 5}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
for expe in mnist_VAE_class_E1_5_5:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)

# _____________ VAE 5 10 + class + E1 ________________
latent_spec = {'cont_var': 5, 'cont_class': 10}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
for expe in mnist_VAE_class_E1_5_10:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)

# _____________ VAE 5 15 + class + E1 ________________
latent_spec = {'cont_var': 5, 'cont_class': 15}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
for expe in mnist_VAE_class_E1_5_15:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
              is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)
"""
