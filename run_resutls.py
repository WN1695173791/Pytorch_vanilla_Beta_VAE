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
              path_scores=None, batch=None, img_size=None, indx_image=None, path=None, losses=True, real_img=False,
              FID=False, IS=False, psnr=False, scores=True, all_prototype=False, copute_average_z_structural=False,
              is_partial_rand_class=False, all_classes_resum=True, save=False, scores_and_losses=False,
              size_struct=None, size_var=None, plot_gaussian=False, sample_real=False,
              heatmap=False, prototype=False, all_classes_details=False, project_2d=False, is_E1=False,
              reconstruction=False, plot_img_traversal=False, z_component_traversal=None, plot_sample=False,
              real_distribution=False, both_latent_traversal=False):
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
                       print_per_class=True, print_per_var=True,
                       plot_traversal_struct=False, is_partial_rand_class=is_partial_rand_class, save=save)
        plot_prototype(net, expe_name, nb_class, latent_spec, device, train_loader, train_test='train',
                       print_per_class=True, print_per_var=True,
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
        if both_latent_traversal:
            joint_latent_traversal(net, nb_epochs, path, expe_name, latent_spec, batch, img_size,
                                   both_continue=True, is_partial_rand_class=is_partial_rand_class, is_E1=is_E1,
                                   size_struct=size_struct, size_var=size_var, save=save, real_img=real_img)
        else:
            plot_images_taversal(net, expe_name, latent_spec, batch, nb_epochs, path, img_size, indx_image=indx_image,
                                 size=8, save=True, z_component_traversal=z_component_traversal,
                                 is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)

    if plot_sample:
        if sample_real:
            sample_real_distribution(net, path, expe_name, latent_spec, img_size, train_test='train', batch=batch,
                                     both_continue=True, save=True, FID=FID, IS=IS, psnr=psnr,
                                     is_partial_rand_class=is_partial_rand_class, is_E1=is_E1,
                                     is_zvar_sim_loss=is_zvar_sim_loss)
        else:
            plot_samples(net, nb_epochs, path, expe_name, latent_spec, img_size, batch=batch, both_continue=True, save=save,
                         FID=FID, IS=IS, psnr=psnr)

    if all_prototype:
        plot_prototype(net, expe_name, nb_class, latent_spec, device, train_loader, train_test='train',
                       print_per_class=True, print_per_var=True,
                       plot_traversal_struct=False, print_2d_projection=True,
                       is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)
        plot_prototype(net, expe_name, nb_class, latent_spec, device, test_loader, train_test='test',
                       print_per_class=True, print_per_var=True,
                       plot_traversal_struct=False, print_2d_projection=True,
                       is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)

    if copute_average_z_structural:
        plot_average_z_structural(net_trained, train_loader, device, nb_class, latent_spec, expe_name,
                                  train_test='train', both_continue=True,
                                  is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)
        plot_average_z_structural(net_trained, test_loader, device, nb_class, latent_spec, expe_name,
                                  train_test='test', both_continue=True,
                                  is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)

    if real_distribution:
        real_distribution_model(net, path, expe_name, train_loader, latent_spec, train_test='train',
                                is_both_continue=True, is_partial_rand_class=is_partial_rand_class, is_E1=is_E1,
                                plot_gaussian=plot_gaussian, save=save)
        real_distribution_model(net, path, expe_name, test_loader, latent_spec, train_test='test',
                                is_both_continue=True, is_partial_rand_class=is_partial_rand_class, is_E1=is_E1,
                                plot_gaussian=plot_gaussian, save=save)

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
save = True
L3_without_random = False
is_binary_structural_latent = False

# for traversal real image:
indx_image = 0

# _______________________________Expe test VAE + class + E1_____________________________________

VAE_Class_E1_old_w_expe_beta_E1_second_new = ['VAE_Class_E1_old_w_E1_new_expe_beta_2_1']

VAE_test_5_5 = [# 'VAE_Class_E1_old_w_E1_new_expe_beta_2_2',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_3',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_4',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_6',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_7',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_8',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_9',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_11',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_12',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_13',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_14',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_15',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_16',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_17',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_18',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_19',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_20',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_21',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_22',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_23',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_24',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_25',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_26',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_27',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_28',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_29',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_30',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_31',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_32',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_33',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_34',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_35',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_36',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_37',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_38',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_39',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_40',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_41',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_42',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_43',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_44',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_45',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_46',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_47',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_48',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_49',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_50',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_51',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_52',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_53',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_54',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_55',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_56',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_57',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_1',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_2',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_3',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_4',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_5',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_6',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_7',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_8',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_9',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_11',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_12',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_13',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_14',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_15',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_16',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_17',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_18',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_19',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_20',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_21',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_22',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_23',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_24',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_25',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_26',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_27',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_28',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_29',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_30',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_31',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_32',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_33',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_34',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_35',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_36',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_37',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_38',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_39',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_40',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_41',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_42',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_43',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_44',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_45',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_46',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_47',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_48',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_49',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_50',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_51',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_52',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_53',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_54',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_55',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_56',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_57',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_1',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_2',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_3',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_4',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_5',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_6',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_7',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_8',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_9',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_11',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_12',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_13',
                'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_14',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_15',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_16',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_17',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_18',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_19',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_20',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_21',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_22',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_23',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_24',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_25',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_26',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_27',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_28',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_29',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_30']
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_31',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_32',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_33',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_34',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_35',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_36',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_37',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_38',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_39',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_40',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_41',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_42',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_43',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_44',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_45',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_46',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_47',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_48',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_49',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_50',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_51',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_52',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_53',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_54',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_55',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_56',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_57'
                ]

is_zvar_sim_loss = False
is_partial_rand_class = False
is_C = True

# _____________ VAE 5 5 + class + E1 old weights ________________
latent_spec = {'cont_var': 5, 'cont_class': 5}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
for expe in VAE_test_5_5:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    # # scores and losses:
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
    #         is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True)
    # # sample:
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
    #           save=True, batch=batch, plot_sample=True, FID=True, IS=False, psnr=False)
    # # reconstruction:
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
    #           batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
    #           save=True, is_E1=is_E1, reconstruction=True)
    # # Traversal:
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
    #           batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
    #           is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)
    # # prototype
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader,
             copute_average_z_structural=True, is_partial_rand_class=is_partial_rand_class, save=True,
             is_E1=is_E1)
    # # projection 2d:
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, all_prototype=True,
             is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1)
    # gaussian real distribution:
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path=path, save=True,
    #           is_partial_rand_class=is_partial_rand_class, is_E1=is_E1, real_distribution=True, plot_gaussian=True)
    # # traversal image with struct fixe and var fixe:
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, batch=batch,
    #           path=path, real_img=False, size_struct=10, size_var=8,
    #           is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1,
    #           plot_img_traversal=True, both_latent_traversal=True)
    # sample from real distribution:
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, plot_sample=True,
    #           sample_real=True, batch=batch, path=path,
    #           save=True, FID=True, IS=False, psnr=False, is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)


# -------------------------------- 5 10 -----------------------------------------------
VAE_Class_E1_old_w_expe_5_10_1 = ['VAE_Class_E1_old_w_E1_new_expe_beta_2_1_struct_10']

VAE_test_5_5 = []# 'VAE_Class_E1_old_w_E1_new_expe_beta_2_2_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_3_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_4_10',
                # # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_6_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_7_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_8_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_9_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_10_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_11_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_12_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_13_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_14_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_15_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_16_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_17_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_18_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_19_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_20_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_21_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_22_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_23_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_24_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_25_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_26_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_27_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_28_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_29_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_30_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_31_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_32_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_33_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_34_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_35_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_36_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_37_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_38_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_39_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_40_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_41_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_42_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_43_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_44_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_45_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_46_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_47_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_48_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_49_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_50_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_51_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_52_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_53_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_54_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_55_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_56_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_57_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_1_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_2_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_3_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_4_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_5_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_6_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_7_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_8_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_9_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_10_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_11_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_12_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_13_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_14_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_15_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_16_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_17_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_18_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_19_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_20_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_21_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_22_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_23_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_24_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_25_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_26_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_27_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_28_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_29_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_30_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_31_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_32_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_33_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_34_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_35_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_36_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_37_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_38_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_39_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_40_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_41_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_42_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_43_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_44_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_45_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_46_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_47_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_48_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_49_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_50_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_51_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_52_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_53_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_54_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_55_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_56_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_3_57_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_1_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_2_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_3_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_4_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_5_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_6_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_7_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_8_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_9_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_10_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_11_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_12_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_13_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_14_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_15_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_16_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_17_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_18_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_19_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_20_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_21_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_22_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_23_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_24_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_25_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_26_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_27_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_28_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_29_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_30_10']
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_31_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_32_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_33_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_34_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_35_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_36_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_37_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_38_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_39_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_40_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_41_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_42_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_43_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_44_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_45_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_46_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_47_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_48_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_49_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_50_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_51_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_52_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_53_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_54_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_55_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_56_10',
                # 'VAE_Class_E1_old_w_E1_new_expe_beta_2_5_57_10',]

is_zvar_sim_loss = False
is_partial_rand_class = False
is_C = True

# _____________ VAE 5 10 + class + E1 old weights ________________
latent_spec = {'cont_var': 5, 'cont_class': 10}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
for expe in VAE_test_5_5:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    # scores and losses:
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
            is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True)
    # sample:
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=False, psnr=False)
    # reconstruction:
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    # Traversal:
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)
    # prototype
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader,
             copute_average_z_structural=True, is_partial_rand_class=is_partial_rand_class, save=True,
             is_E1=is_E1)
    # projection 2d:
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, all_prototype=True,
    #          is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1)
    # # gaussian real distribution:
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path=path, save=True,
    #           is_partial_rand_class=is_partial_rand_class, is_E1=is_E1, real_distribution=True, plot_gaussian=True)
    # traversal image with struct fixe and var fixe:
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, batch=batch,
              path=path, real_img=False, size_struct=10, size_var=8,
              is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1,
              plot_img_traversal=True, both_latent_traversal=True)
    # sample from real distribution:
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, plot_sample=True,
    #           sample_real=True, batch=batch, path=path,
    #           save=True, FID=False, IS=False, psnr=False, is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)


"""
VAE_Class_E1_old_w_expe_5_15_1 = ['VAE_Class_E1_old_w_expe_5_15_1']

is_zvar_sim_loss = False
is_partial_rand_class = False
is_C = True


# _____________ VAE 5 15 + class + E1 old weights ________________
latent_spec = {'cont_var': 5, 'cont_class': 15}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
for expe in VAE_Class_E1_old_w_expe_5_15_1:
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
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader,
             copute_average_z_structural=True, is_partial_rand_class=is_partial_rand_class, save=True,
             is_E1=is_E1)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, all_prototype=True,
             is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path=path, save=True,
              is_partial_rand_class=is_partial_rand_class, is_E1=is_E1, real_distribution=True, plot_gaussian=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, batch=batch,
              path=path, real_img=False, size_struct=10, size_var=8,
              is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1,
              plot_img_traversal=True, both_latent_traversal=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, plot_sample=True,
              sample_real=True, batch=batch, path=path,
              save=True, FID=False, IS=False, psnr=False, is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)
"""
"""
# _____________ VAE 5 5 + class + E1 ________________
latent_spec = {'cont_var': 5, 'cont_class': 5}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
for expe in mnist_VAE_class_E1:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
    #         is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader,
    #          copute_average_z_structural=True, is_partial_rand_class=is_partial_rand_class, save=True,
    #          is_E1=is_E1)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, all_prototype=True,
    #           is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path=path, save=True,
    #           is_partial_rand_class=is_partial_rand_class, is_E1=is_E1, real_distribution=True, plot_gaussian=True)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, batch=batch,
    #           path=path, real_img=False, size_struct=10, size_var=8,
    #           is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1,
    #           plot_img_traversal=True, both_latent_traversal=True)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
    #           save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False, sample_real=True)

"""
"""
# _______________________________Expe test vanilla VAE + Class + E1 + zvar_sim_____________________________________

mnist_VAE_class_E1_zvar_sim_best = ['VAE_class_E1_MSE_41', 'VAE_Cass_E1_Zvarsim_tune_WLr_25',
                                    'VAE_Cass_E1_Zvarsim_tune_WLr_112', 'VAE_Cass_E1_Zvarsim_tune_WLzvar_44']

mnist_VAE_class_E1_zvar_old_w = ['VAE_full_zvar_sim_strategie_1_old_w_expe_1',
                                 'VAE_full_zvar_sim_strategie_1_old_w_expe_2',
                                 'VAE_full_zvar_sim_strategie_1_old_w_expe_3',
                                 'VAE_full_zvar_sim_strategie_1_old_w_expe_4',
                                 'VAE_full_zvar_sim_strategie_1_old_w_expe_5',
                                 'VAE_full_zvar_sim_strategie_1_old_w_expe_6',
                                 'VAE_full_zvar_sim_strategie_1_old_w_expe_7',
                                 'VAE_full_zvar_sim_strategie_1_old_w_expe_8',
                                 'VAE_full_zvar_sim_strategie_1_old_w_expe_9',
                                 'VAE_full_zvar_sim_strategie_1_old_w_expe_10',
                                 'VAE_full_zvar_sim_strategie_1_old_w_expe_11',
                                 'VAE_full_zvar_sim_strategie_1_old_w_expe_12']

mnist_VAE_class_E1_zvar = [# 'VAE_full_zvar_sim_strategie_1_expe_1',
                           # 'VAE_full_zvar_sim_strategie_1_expe_2',
                           # 'VAE_full_zvar_sim_strategie_1_expe_3',
                           # 'VAE_full_zvar_sim_strategie_1_expe_4',
                           # 'VAE_full_zvar_sim_strategie_1_expe_5',
                           # 'VAE_full_zvar_sim_strategie_1_expe_6',
                           # 'VAE_full_zvar_sim_strategie_1_expe_7',
                           # 'VAE_full_zvar_sim_strategie_1_expe_8',
                           # 'VAE_full_zvar_sim_strategie_1_expe_9',
                           # 'VAE_full_zvar_sim_strategie_1_expe_10',
                           # 'VAE_full_zvar_sim_strategie_1_expe_11',
                           'VAE_full_zvar_sim_strategie_1_expe_12',
                           'VAE_full_zvar_sim_strategie_1_expe_13',
                           'VAE_full_zvar_sim_strategie_1_expe_14',
                           'VAE_full_zvar_sim_strategie_1_expe_15',
                           'VAE_full_zvar_sim_strategie_1_expe_16',
                           'VAE_full_zvar_sim_strategie_1_expe_17',
                           'VAE_full_zvar_sim_strategie_1_expe_18',
                           'VAE_full_zvar_sim_strategie_1_expe_19',
                           'VAE_full_zvar_sim_strategie_1_expe_20',
                           'VAE_full_zvar_sim_strategie_1_expe_21',
                           'VAE_full_zvar_sim_strategie_1_expe_22',
                           'VAE_full_zvar_sim_strategie_1_expe_23',
                           'VAE_full_zvar_sim_strategie_1_expe_24']

mnist_VAE_class_E1_zvar_test = ['VAE_full_zvar_sim_strategie_1_old_w_expe_5',
                                'VAE_full_zvar_sim_strategie_1_expe_13']

is_zvar_sim_loss = True
is_partial_rand_class = False
is_E1 = True
E1_conv = True
is_C = True

# for traversal real image:
indx_image = 0


# _____________ VAE 5 5 + class + E1 + zvar_sim (old weight) ________________
latent_spec = {'cont_var': 5, 'cont_class': 5}
BN = True
second_layer_C = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
for expe in mnist_VAE_class_E1_zvar_test:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
    #         is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True)
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
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, all_prototype=True,
    #          is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path=path, save=True,
    #           is_partial_rand_class=is_partial_rand_class, is_E1=is_E1, real_distribution=True, plot_gaussian=True)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, batch=batch,
    #           path=path, real_img=False, size_struct=10, size_var=8,
    #           is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1,
    #           plot_img_traversal=True, both_latent_traversal=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, plot_sample=True,
              sample_real=True, batch=batch, path=path,
              save=True, FID=False, IS=False, psnr=False, is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)
"""

"""
# _____________ VAE 5 5 + class + E1 + zvar_sim ________________
latent_spec = {'cont_var': 5, 'cont_class': 5}
BN = True
second_layer_C = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
for expe in mnist_VAE_class_E1_zvar:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
    #         is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1, losses=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader,
    #          copute_average_z_structural=True, is_partial_rand_class=is_partial_rand_class, save=True,
    #          is_E1=is_E1)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, all_prototype=True,
    #           is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path=path, save=True,
    #           is_partial_rand_class=is_partial_rand_class, is_E1=is_E1, real_distribution=True, plot_gaussian=True)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, batch=batch,
    #           path=path, real_img=False, size_struct=10, size_var=8,
    #           is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1,
    #           plot_img_traversal=True, both_latent_traversal=True)
    # visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
    #           save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False, sample_real=True)

"""
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

mnist_VAE_5_5 = ['VAE_5-5']
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

for expe in mnist_VAE_5_5:
    expe_name = expe
    net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
            is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=False, is_E1=is_E1, losses=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs, path=path,
              save=True, batch=batch, plot_sample=True, FID=True, IS=True, psnr=False)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
              save=True, is_E1=is_E1, reconstruction=True)

    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
              batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
              is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image, plot_img_traversal=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader,
             copute_average_z_structural=True, is_partial_rand_class=is_partial_rand_class, save=True,
             is_E1=is_E1)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, all_prototype=True,
             is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path=path, save=True,
              is_partial_rand_class=is_partial_rand_class, is_E1=is_E1, real_distribution=True, plot_gaussian=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, batch=batch,
              path=path, real_img=False, size_struct=10, size_var=8,
              is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1,
              plot_img_traversal=True, both_latent_traversal=True)
    visualize(net, nb_class, expe_name, device, latent_spec, train_loader, test_loader, plot_sample=True,
              sample_real=True, batch=batch, path=path,
              save=True, FID=False, IS=False, psnr=False, is_partial_rand_class=is_partial_rand_class, is_E1=is_E1)
"""
