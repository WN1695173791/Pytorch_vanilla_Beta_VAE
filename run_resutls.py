from models.model import BetaVAE
from models.default_CNN import DefaultCNN
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


def visualize(net, net_trained, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=None,
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
            sample_real_distribution(net, path, expe_name, latent_spec, img_size, train_test='test', batch=batch,
                                     both_continue=True, save=True, FID=FID, IS=IS, psnr=psnr,
                                     is_partial_rand_class=is_partial_rand_class, is_E1=is_E1,
                                     is_zvar_sim_loss=is_zvar_sim_loss)
        else:
            plot_samples(net, nb_epochs, path, expe_name, latent_spec, img_size, batch=batch, both_continue=True,
                         save=save,
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


def run_score(expes_list, net):
    path = 'checkpoints_CNN/'
    path_scores = 'checkpoint_scores_CNN'
    latent_spec = None
    for expe in expes_list:
        print(expe)
        expe_name = expe
        net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
        # scores and losses:
        plot_scores_and_loss_CNN(net_trained, expe_name, path_scores, save=True)


def run(expe_list, net, E1_VAE):
    path = 'checkpoints/'
    path_scores = 'checkpoints_scores'
    for expe in expe_list:
        print(expe)
        expe_name = expe
        net_trained, _, nb_epochs = get_checkpoints(net, path, expe_name)
        # # scores and losses:
        visualize(net, net_trained, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path_scores=path_scores,
                  is_partial_rand_class=is_partial_rand_class, save=True, scores_and_losses=True, is_E1=is_E1,
                  losses=True)
        # sample:
        visualize(net, net_trained, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
                  path=path,
                  save=True, batch=batch, plot_sample=True, FID=False, IS=False, psnr=False)
        # reconstruction:
        visualize(net, net_trained, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
                  batch=batch, img_size=img_size, is_partial_rand_class=is_partial_rand_class,
                  save=True, is_E1=is_E1, reconstruction=True)
        # Traversal:
        visualize(net, net_trained, nb_class, expe_name, device, latent_spec, train_loader, test_loader, nb_epochs=nb_epochs,
                  batch=batch, img_size=img_size, path=path, is_partial_rand_class=is_partial_rand_class,
                  is_E1=is_E1, z_component_traversal=z_component_traversal, indx_image=indx_image,
                  plot_img_traversal=True)
        # prototype
        visualize(net, net_trained, nb_class, expe_name, device, latent_spec, train_loader, test_loader,
                  copute_average_z_structural=True, is_partial_rand_class=is_partial_rand_class, save=True,
                  is_E1=is_E1)
        # projection 2d:
        visualize(net, net_trained, nb_class, expe_name, device, latent_spec, train_loader, test_loader,
                  all_prototype=True, is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1)
        # gaussian real distribution:
        if E1_VAE:
            visualize(net, net_trained, nb_class, expe_name, device, latent_spec, train_loader, test_loader, path=path, save=True,
                      is_partial_rand_class=is_partial_rand_class, is_E1=is_E1, real_distribution=True, plot_gaussian=True)
            # sample from real distribution:
            visualize(net, net_trained, nb_class, expe_name, device, latent_spec, train_loader, test_loader,
                      plot_sample=True,
                      sample_real=True, batch=batch, path=path,
                      save=True, FID=False, IS=False, psnr=False, is_partial_rand_class=is_partial_rand_class,
                      is_E1=is_E1)
        # traversal image with struct fixe and var fixe:
        visualize(net, net_trained, nb_class, expe_name, device, latent_spec, train_loader, test_loader, batch=batch,
                  path=path, real_img=False, size_struct=10, size_var=8,
                  is_partial_rand_class=is_partial_rand_class, save=True, is_E1=is_E1,
                  plot_img_traversal=True, both_latent_traversal=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch = torch.load('data/batch_mnist.pt')

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

# ------------- expes CNN classifier: -----------------
list_CNN_defaults = ['CNN_mnist_default_1',
                     'CNN_mnist_default_2',
                     'CNN_mnist_default_3',
                     'CNN_mnist_default_4']

net = DefaultCNN()

# print(net)
run_score(list_CNN_defaults, net)

# _______________________________Expe test VAE + class + E1_____________________________________

# _____________ VAE 5 5 + class + E1 old weights ________________

VAE_resume_5_2 = ['VAE_resume_2_5_1',
                  'VAE_resume_2_5_2',
                  'VAE_resume_2_5_3',
                  'VAE_resume_2_5_4',
                  'VAE_resume_2_5_5',
                  'VAE_resume_2_5_6',
                  'VAE_resume_2_5_7',
                  'VAE_resume_2_5_8',
                  'VAE_resume_2_5_9',
                  'VAE_resume_2_5_10',
                  'VAE_resume_2_5_11',
                  'VAE_resume_2_5_12',
                  'VAE_resume_2_5_13',
                  'VAE_resume_2_5_14',
                  'VAE_resume_2_5_15',
                  'VAE_resume_2_5_16',
                  'VAE_resume_2_5_17',
                  'VAE_resume_2_5_18',
                  'VAE_resume_2_5_19',
                  'VAE_resume_2_5_20',
                  'VAE_resume_2_5_21',
                  'VAE_resume_2_5_22',
                  'VAE_resume_2_5_23',
                  'VAE_resume_2_5_24',
                  'VAE_resume_2_5_25',
                  'VAE_resume_2_5_26',
                  'VAE_resume_2_5_27',
                  'VAE_resume_2_5_28',
                  'VAE_resume_2_5_29',
                  'VAE_resume_2_5_30',
                  'VAE_resume_2_5_31',
                  'VAE_resume_2_5_32',
                  'VAE_resume_2_5_33',
                  'VAE_resume_2_5_34',
                  'VAE_resume_2_5_35',
                  'VAE_resume_2_5_36',
                  'VAE_resume_2_5_37',
                  'VAE_resume_2_5_38',
                  'VAE_resume_2_5_39',
                  'VAE_resume_2_5_40',
                  'VAE_resume_2_5_41',
                  'VAE_resume_2_5_42',
                  'VAE_resume_2_5_43',
                  'VAE_resume_2_5_44',
                  'VAE_resume_2_5_45',
                  'VAE_resume_2_5_46',
                  'VAE_resume_2_5_47',
                  'VAE_resume_2_5_48',
                  'VAE_resume_2_5_49',
                  'VAE_resume_2_L1_5_1',
                  'VAE_resume_2_L1_5_2',
                  'VAE_resume_2_L1_5_3',
                  'VAE_resume_2_L1_5_4',
                  'VAE_resume_2_L1_5_5',
                  'VAE_resume_2_L1_5_6',
                  'VAE_resume_2_L1_5_7',
                  'VAE_resume_2_L1_5_8',
                  'VAE_resume_2_L1_5_9',
                  'VAE_resume_2_L1_5_10',
                  'VAE_resume_2_L1_5_11',
                  'VAE_resume_2_L1_5_12',
                  'VAE_resume_2_L1_5_13',
                  'VAE_resume_2_L1_5_14',
                  'VAE_resume_2_L1_5_15',
                  'VAE_resume_2_L1_5_16',
                  'VAE_resume_2_L1_5_17',
                  'VAE_resume_2_L1_5_18',
                  'VAE_resume_2_L1_5_19',
                  'VAE_resume_2_L1_5_20',
                  'VAE_resume_2_L1_5_21',
                  'VAE_resume_2_L1_5_22',
                  'VAE_resume_2_L1_5_23',
                  'VAE_resume_2_L1_5_24',
                  'VAE_resume_2_L1_5_25',
                  'VAE_resume_2_L1_5_26',
                  'VAE_resume_2_L1_5_27',
                  'VAE_resume_2_L1_5_28',
                  'VAE_resume_2_L1_5_29',
                  'VAE_resume_2_L1_5_30',
                  'VAE_resume_2_L1_5_31',
                  'VAE_resume_2_L1_5_32',
                  'VAE_resume_2_L1_5_33',
                  'VAE_resume_2_L1_5_34',
                  'VAE_resume_2_L1_5_35',
                  'VAE_resume_2_L1_5_36',
                  'VAE_resume_2_L1_5_37',
                  'VAE_resume_2_L1_5_38',
                  'VAE_resume_2_L1_5_39',
                  'VAE_resume_2_L1_5_40'
                  ]

VAE_resume_5_3 = ['VAE_resume_3_5_1',
                  'VAE_resume_3_5_2',
                  'VAE_resume_3_5_3',
                  'VAE_resume_3_5_4',
                  'VAE_resume_3_5_5',
                  'VAE_resume_3_5_6',
                  'VAE_resume_3_5_7',
                  'VAE_resume_3_5_8',
                  'VAE_resume_3_5_9',
                  'VAE_resume_3_5_10',
                  'VAE_resume_3_5_11',
                  'VAE_resume_3_5_12',
                  'VAE_resume_3_5_13',
                  'VAE_resume_3_5_14',
                  'VAE_resume_3_5_15',
                  'VAE_resume_3_5_16',
                  'VAE_resume_3_5_17',
                  'VAE_resume_3_5_18',
                  'VAE_resume_3_5_19',
                  'VAE_resume_3_5_20',
                  'VAE_resume_3_5_21',
                  'VAE_resume_3_5_22',
                  'VAE_resume_3_5_23',
                  'VAE_resume_3_5_24',
                  'VAE_resume_3_5_25',
                  'VAE_resume_3_5_26',
                  'VAE_resume_3_5_27',
                  'VAE_resume_3_5_28',
                  'VAE_resume_3_5_29',
                  'VAE_resume_3_5_30',
                  'VAE_resume_3_5_31',
                  'VAE_resume_3_5_32',
                  'VAE_resume_3_5_33',
                  'VAE_resume_3_5_34',
                  'VAE_resume_3_5_35',
                  'VAE_resume_3_5_36',
                  'VAE_resume_3_5_37',
                  'VAE_resume_3_5_38',
                  'VAE_resume_3_5_39',
                  'VAE_resume_3_5_40',
                  'VAE_resume_3_5_41',
                  'VAE_resume_3_5_42',
                  'VAE_resume_3_5_43',
                  'VAE_resume_3_5_44',
                  'VAE_resume_3_5_45',
                  'VAE_resume_3_5_46',
                  'VAE_resume_3_5_47',
                  'VAE_resume_3_5_48',
                  'VAE_resume_3_5_49',
                  'VAE_resume_3_L1_5_1',
                  'VAE_resume_3_L1_5_2',
                  'VAE_resume_3_L1_5_3',
                  'VAE_resume_3_L1_5_4',
                  'VAE_resume_3_L1_5_5',
                  'VAE_resume_3_L1_5_6',
                  'VAE_resume_3_L1_5_7',
                  'VAE_resume_3_L1_5_8',
                  'VAE_resume_3_L1_5_9',
                  'VAE_resume_3_L1_5_10',
                  'VAE_resume_3_L1_5_11',
                  'VAE_resume_3_L1_5_12',
                  'VAE_resume_3_L1_5_13',
                  'VAE_resume_3_L1_5_14',
                  'VAE_resume_3_L1_5_15',
                  'VAE_resume_3_L1_5_16',
                  'VAE_resume_3_L1_5_17',
                  'VAE_resume_3_L1_5_18',
                  'VAE_resume_3_L1_5_19',
                  'VAE_resume_3_L1_5_20',
                  'VAE_resume_3_L1_5_21',
                  'VAE_resume_3_L1_5_22',
                  'VAE_resume_3_L1_5_23',
                  'VAE_resume_3_L1_5_24',
                  'VAE_resume_3_L1_5_25',
                  'VAE_resume_3_L1_5_26',
                  'VAE_resume_3_L1_5_27',
                  'VAE_resume_3_L1_5_28',
                  'VAE_resume_3_L1_5_29',
                  'VAE_resume_3_L1_5_30',
                  'VAE_resume_3_L1_5_31',
                  'VAE_resume_3_L1_5_32',
                  'VAE_resume_3_L1_5_33',
                  'VAE_resume_3_L1_5_34',
                  'VAE_resume_3_L1_5_35',
                  'VAE_resume_3_L1_5_36',
                  'VAE_resume_3_L1_5_37',
                  'VAE_resume_3_L1_5_38',
                  'VAE_resume_3_L1_5_39',
                  'VAE_resume_3_L1_5_40',
                  'VAE_resume_3_L1_5_41',
                  'VAE_resume_3_L1_5_42',
                  'VAE_resume_3_L1_5_43',
                  'VAE_resume_3_L1_5_44',
                  'VAE_resume_3_L1_5_45',
                  'VAE_resume_3_L1_5_46',
                  'VAE_resume_3_L1_5_47',
                  'VAE_resume_3_L1_5_48',
                  'VAE_resume_3_L1_5_49'
                  ]

VAE_resume_5_4 = [# 'VAE_resume_4_5_1',
                  # 'VAE_resume_4_5_2',
                  # 'VAE_resume_4_5_3',
                  # 'VAE_resume_4_5_4',
                  # 'VAE_resume_4_5_5',
                  # 'VAE_resume_4_5_6',
                  # 'VAE_resume_4_5_7',
                  # 'VAE_resume_4_5_8',
                  # 'VAE_resume_4_5_9',
                  # 'VAE_resume_4_5_10',
                  # 'VAE_resume_4_5_11',
                  # 'VAE_resume_4_5_12',
                  # 'VAE_resume_4_5_13',
                  # 'VAE_resume_4_5_14',
                  # 'VAE_resume_4_5_15',
                  # 'VAE_resume_4_5_16',
                  # 'VAE_resume_4_5_17',
                  # 'VAE_resume_4_5_18',
                  # 'VAE_resume_4_5_19',
                  # 'VAE_resume_4_5_20',
                  # 'VAE_resume_4_5_21',
                  # 'VAE_resume_4_5_22',
                  # 'VAE_resume_4_5_23',
                  # 'VAE_resume_4_5_24',
                  # 'VAE_resume_4_5_25',
                  # 'VAE_resume_4_5_26',
                  # 'VAE_resume_4_5_27',
                  # 'VAE_resume_4_5_28',
                  # 'VAE_resume_4_5_29',
                  # 'VAE_resume_4_5_30',
                  # 'VAE_resume_4_5_31',
                  # 'VAE_resume_4_5_32',
                  # 'VAE_resume_4_5_33',
                  # 'VAE_resume_4_5_34',
                  # 'VAE_resume_4_5_35',
                  # 'VAE_resume_4_5_36',
                  # 'VAE_resume_4_5_37',
                  # 'VAE_resume_4_5_38',
                  # 'VAE_resume_4_5_39',
                  # 'VAE_resume_4_5_40',
                  # 'VAE_resume_4_5_41',
                  # 'VAE_resume_4_5_42',
                  # 'VAE_resume_4_5_43',
                  # 'VAE_resume_4_5_44',
                  # 'VAE_resume_4_5_45',
                  # 'VAE_resume_4_5_46',
                  # 'VAE_resume_4_5_47',
                  # 'VAE_resume_4_5_48',
                  # 'VAE_resume_4_5_49',
                  'VAE_resume_4_L1_5_1',
                  'VAE_resume_4_L1_5_2',
                  'VAE_resume_4_L1_5_3',
                  'VAE_resume_4_L1_5_4',
                  'VAE_resume_4_L1_5_5',
                  'VAE_resume_4_L1_5_6',
                  'VAE_resume_4_L1_5_7',
                  'VAE_resume_4_L1_5_8',
                  'VAE_resume_4_L1_5_9',
                  'VAE_resume_4_L1_5_10',
                  'VAE_resume_4_L1_5_11',
                  'VAE_resume_4_L1_5_12',
                  'VAE_resume_4_L1_5_13',
                  'VAE_resume_4_L1_5_14',
                  'VAE_resume_4_L1_5_15',
                  'VAE_resume_4_L1_5_16',
                  'VAE_resume_4_L1_5_17',
                  'VAE_resume_4_L1_5_18',
                  'VAE_resume_4_L1_5_19',
                  'VAE_resume_4_L1_5_20',
                  'VAE_resume_4_L1_5_21',
                  'VAE_resume_4_L1_5_22',
                  'VAE_resume_4_L1_5_23',
                  'VAE_resume_4_L1_5_24',
                  'VAE_resume_4_L1_5_25',
                  'VAE_resume_4_L1_5_26',
                  'VAE_resume_4_L1_5_27',
                  'VAE_resume_4_L1_5_28',
                  'VAE_resume_4_L1_5_29',
                  'VAE_resume_4_L1_5_30',
                  'VAE_resume_4_L1_5_31',
                  'VAE_resume_4_L1_5_32',
                  'VAE_resume_4_L1_5_33',
                  'VAE_resume_4_L1_5_34',
                  'VAE_resume_4_L1_5_35',
                  'VAE_resume_4_L1_5_36',
                  'VAE_resume_4_L1_5_37',
                  'VAE_resume_4_L1_5_38',
                  'VAE_resume_4_L1_5_39',
                  'VAE_resume_4_L1_5_40',
                  'VAE_resume_4_L1_5_41',
                  'VAE_resume_4_L1_5_42',
                  'VAE_resume_4_L1_5_43',
                  'VAE_resume_4_L1_5_44',
                  'VAE_resume_4_L1_5_45',
                  'VAE_resume_4_L1_5_46',
                  'VAE_resume_4_L1_5_47',
                  'VAE_resume_4_L1_5_48',
                  'VAE_resume_4_L1_5_49'
                  ]

E1_VAE = True
E1_AE = False
E1_second_conv_adapt = False
two_encoder = True
GMP = True
big_kernel = True

is_zvar_sim_loss = False
is_partial_rand_class = False
is_C = True

latent_spec = {'cont_var': 5, 'cont_class': 5}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
# print(net)
z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
# run(VAE_resume_5_2, net, E1_VAE)
# run(VAE_resume_5_3, net, E1_VAE)
# run(VAE_resume_5_4, net, E1_VAE)

# --------------------------------------- VAE_full_zvar_sim_strategie_1_expe_13 ------------------
VAE_test_5_5_1 = ['VAE_full_zvar_sim_strategie_1_expe_13']

E1_VAE = True
E1_AE = False
E1_second_conv_adapt = False
two_encoder = False
GMP = False
big_kernel = False

is_zvar_sim_loss = False
is_partial_rand_class = False
is_C = True

latent_spec = {'cont_var': 5, 'cont_class': 5}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
# print(net)
z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
# run(VAE_test_5_5_1, net, E1_VAE)

"""
VAE_test_5_5_1 = [# 'VAE_design_new_5_1',
                  # 'VAE_design_new_5_2',
                  # 'VAE_design_new_5_9',
                  # 'VAE_design_new_5_10'
                 ]
E1_VAE = True
E1_AE = False
E1_second_conv_adapt = True
two_encoder = False
GMP = True
big_kernel = False

is_zvar_sim_loss = False
is_partial_rand_class = False
is_C = True

latent_spec = {'cont_var': 5, 'cont_class': 5}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
# print(net)
z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
run(VAE_test_5_5_1, net, E1_VAE)

# ------- Big Kernel: -----------
VAE_test_5_5_1_bk = ['VAE_design_new_bk_5_1',
                     'VAE_design_new_bk_5_2',
                     'VAE_design_new_bk_5_9',
                     'VAE_design_new_bk_5_10']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
# run(VAE_test_5_5_1_bk, net, E1_VAE)

# ------- wt GMP: -----------
VAE_design_wt_GMP_5 = ['VAE_design_wt_GMP_new_5_1',
                       'VAE_design_wt_GMP_new_5_2',
                       'VAE_design_wt_GMP_new_5_9',
                       'VAE_design_wt_GMP_new_5_10',
                       'VAE_design_zeroW_wt_GMP_new_5_1',
                       'VAE_design_zeroW_wt_GMP_new_5_2',
                       'VAE_design_zeroW_wt_GMP_new_5_9',
                       'VAE_design_zeroW_wt_GMP_new_5_10']
big_kernel = False
GMP = False

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

# run(VAE_design_wt_GMP_5, net, E1_VAE)

# ------- Big Kernel + wt GMP: -----------
VAE_design_wt_GMP_bk_5 = ['VAE_design_wt_GMP_new_bk_5_1',
                       'VAE_design_wt_GMP_new_bk_5_2',
                       'VAE_design_wt_GMP_new_bk_5_9',
                       'VAE_design_wt_GMP_new_bk_5_10',
                            'VAE_design_zeroW_wt_GMP_new_bk_5_1',
                            'VAE_design_zeroW_wt_GMP_new_bk_5_2',
                            'VAE_design_zeroW_wt_GMP_new_bk_5_9',
                            'VAE_design_zeroW_wt_GMP_new_bk_5_10']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

# run(VAE_design_wt_GMP_bk_5, net, E1_VAE)

# _____________ VAE 5 5 + class + E1 old weights ________________

VAE_test_5_5_2 = ['VAE_design_new_5_3',
                  'VAE_design_new_5_4',
                  'VAE_design_new_5_11',
                  'VAE_design_new_5_12']

E1_VAE = False
E1_AE = True
E1_second_conv_adapt = True
two_encoder = False
GMP = True
big_kernel = False

latent_spec = {'cont_var': 5, 'cont_class': 5}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
# run(VAE_test_5_5_2, net, E1_VAE)

# ------- Big Kernel: -----------
VAE_test_5_5_2_bk = [# 'VAE_design_new_bk_5_3',
                     # 'VAE_design_new_bk_5_4',
                     # 'VAE_design_new_bk_5_11']
                     'VAE_design_new_bk_5_12']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
# run(VAE_test_5_5_2_bk, net, E1_VAE)

# ------- wt GMP: -----------
VAE_design_wt_GMP_5_2 = ['VAE_design_wt_GMP_new_5_3',
                       'VAE_design_wt_GMP_new_5_4',
                       'VAE_design_wt_GMP_new_5_11',
                       'VAE_design_wt_GMP_new_5_12',
                       'VAE_design_zeroW_wt_GMP_new_5_3',
                       'VAE_design_zeroW_wt_GMP_new_5_4',
                       'VAE_design_zeroW_wt_GMP_new_5_11',
                       'VAE_design_zeroW_wt_GMP_new_5_12']
big_kernel = False
GMP = False

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

# run(VAE_design_wt_GMP_5_2, net, E1_VAE)

# ------- Big Kernel + wt GMP: -----------
VAE_design_wt_GMP_bk_5_2 = ['VAE_design_wt_GMP_new_bk_5_3',
                       'VAE_design_wt_GMP_new_bk_5_4',
                       'VAE_design_wt_GMP_new_bk_5_11',
                       'VAE_design_wt_GMP_new_bk_5_12',
                            'VAE_design_zeroW_wt_GMP_new_bk_5_3',
                            'VAE_design_zeroW_wt_GMP_new_bk_5_4',
                            'VAE_design_zeroW_wt_GMP_new_bk_5_11',
                            'VAE_design_zeroW_wt_GMP_new_bk_5_12']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

# run(VAE_design_wt_GMP_bk_5_2, net, E1_VAE)

# _____________ VAE 5 5 + class + E1 old weights ________________

VAE_test_5_5_3 = ['VAE_design_new_5_5',
                  'VAE_design_new_5_6',
                  'VAE_design_new_5_13',
                  'VAE_design_new_5_14']

E1_VAE = True
E1_AE = False
E1_second_conv_adapt = False
two_encoder = True
GMP = True
big_kernel = False

latent_spec = {'cont_var': 5, 'cont_class': 5}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
# run(VAE_test_5_5_3, net, E1_VAE)

# ------- Big Kernel: -----------
VAE_test_5_5_3_bk = ['VAE_design_new_bk_5_5',
                     'VAE_design_new_bk_5_6',
                     'VAE_design_new_bk_5_13',
                     'VAE_design_new_bk_5_14']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
run(VAE_test_5_5_3_bk, net, E1_VAE)

# ------- wt GMP: -----------
VAE_design_wt_GMP_5_3 = ['VAE_design_wt_GMP_new_5_5',
                       'VAE_design_wt_GMP_new_5_6',
                       'VAE_design_wt_GMP_new_5_13',
                       'VAE_design_wt_GMP_new_5_14',
                       'VAE_design_zeroW_wt_GMP_new_5_5',
                       'VAE_design_zeroW_wt_GMP_new_5_6',
                       'VAE_design_zeroW_wt_GMP_new_5_13',
                       'VAE_design_zeroW_wt_GMP_new_5_14']
big_kernel = False
GMP = False

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

# run(VAE_design_wt_GMP_5_3, net, E1_VAE)

# ------- Big Kernel + wt GMP: -----------
VAE_design_wt_GMP_bk_5_3 = ['VAE_design_wt_GMP_new_bk_5_5',
                       'VAE_design_wt_GMP_new_bk_5_6',
                       'VAE_design_wt_GMP_new_bk_5_13',
                       'VAE_design_wt_GMP_new_bk_5_14',
                            'VAE_design_zeroW_wt_GMP_new_bk_5_5',
                            'VAE_design_zeroW_wt_GMP_new_bk_5_6',
                            'VAE_design_zeroW_wt_GMP_new_bk_5_13',
                            'VAE_design_zeroW_wt_GMP_new_bk_5_14']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

# run(VAE_design_wt_GMP_bk_5_3, net, E1_VAE)

# _____________ VAE 5 5 + class + E1 old weights ________________

VAE_test_5_5_4 = ['VAE_design_new_5_7',
                  'VAE_design_new_5_8',
                  'VAE_design_new_5_15',
                  'VAE_design_new_5_16']

E1_VAE = False
E1_AE = True
E1_second_conv_adapt = False
two_encoder = True
GMP = True
big_kernel = False

latent_spec = {'cont_var': 5, 'cont_class': 5}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
# run(VAE_test_5_5_4, net, E1_VAE)

# ------- Big Kernel: -----------
VAE_test_5_5_4_bk = ['VAE_design_new_bk_5_7',
                     'VAE_design_new_bk_5_8',
                     'VAE_design_new_bk_5_15',
                     'VAE_design_new_bk_5_16']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
run(VAE_test_5_5_4_bk, net, E1_VAE)

# ------- wt GMP: -----------
VAE_design_wt_GMP_5_4 = ['VAE_design_wt_GMP_new_5_7',
                       'VAE_design_wt_GMP_new_5_8',
                       'VAE_design_wt_GMP_new_5_15',
                       'VAE_design_wt_GMP_new_5_16',
                       'VAE_design_zeroW_wt_GMP_new_5_7',
                       'VAE_design_zeroW_wt_GMP_new_5_8',
                       'VAE_design_zeroW_wt_GMP_new_5_15',
                       'VAE_design_zeroW_wt_GMP_new_5_16']
big_kernel = False
GMP = False

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

# run(VAE_design_wt_GMP_5_4, net, E1_VAE)

# ------- Big Kernel + wt GMP: -----------
VAE_design_wt_GMP_bk_5_4 = ['VAE_design_wt_GMP_new_bk_5_7',
                       'VAE_design_wt_GMP_new_bk_5_8',
                       'VAE_design_wt_GMP_new_bk_5_15',
                       'VAE_design_wt_GMP_new_bk_5_16',
                            'VAE_design_zeroW_wt_GMP_new_bk_5_7',
                            'VAE_design_zeroW_wt_GMP_new_bk_5_8',
                            'VAE_design_zeroW_wt_GMP_new_bk_5_15',
                            'VAE_design_zeroW_wt_GMP_new_bk_5_16']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

# run(VAE_design_wt_GMP_bk_5_4, net, E1_VAE)


# -------------------------------- 5 10 -----------------------------------------------

# _____________ VAE 5 10 + class + E1 old weights ________________

VAE_test_5_10_1 = ['VAE_design_new_10_1',
                   'VAE_design_new_10_2',
                   'VAE_design_new_10_9',
                   'VAE_design_new_10_10']
E1_VAE = True
E1_AE = False
E1_second_conv_adapt = True
two_encoder = False
GMP = True
big_kernel = False

is_zvar_sim_loss = False
is_partial_rand_class = False
is_C = True

latent_spec = {'cont_var': 5, 'cont_class': 10}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
# run(VAE_test_5_10_1, net, E1_VAE)

# ------- Big Kernel: -----------
VAE_test_5_10_1_bk = ['VAE_design_new_bk_10_1',
                      'VAE_design_new_bk_10_2',
                      'VAE_design_new_bk_10_9',
                      'VAE_design_new_bk_10_10']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
run(VAE_test_5_10_1_bk, net, E1_VAE)

# ------- wt GMP: -----------
VAE_design_wt_GMP_new_10_1 = ['VAE_design_wt_GMP_new_10_1',
                       'VAE_design_wt_GMP_new_10_2',
                       'VAE_design_wt_GMP_new_10_9',
                       'VAE_design_wt_GMP_new_10_10',
                       'VAE_design_zeroW_wt_GMP_new_10_1',
                       'VAE_design_zeroW_wt_GMP_new_10_2',
                       'VAE_design_zeroW_wt_GMP_new_10_9',
                       'VAE_design_zeroW_wt_GMP_new_10_10']
big_kernel = False
GMP = False

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

# run(VAE_design_wt_GMP_new_10_1, net, E1_VAE)

# ------- Big Kernel + wt GMP: -----------
VAE_design_wt_GMP_bk_10_1 = [# 'VAE_design_wt_GMP_new_bk_10_1',
                             # 'VAE_design_wt_GMP_new_bk_10_2',
                             # 'VAE_design_wt_GMP_new_bk_10_9',
                             # 'VAE_design_wt_GMP_new_bk_10_10',
                             # 'VAE_design_zeroW_wt_GMP_new_bk_10_1',
                             # 'VAE_design_zeroW_wt_GMP_new_bk_10_2',
                             # 'VAE_design_zeroW_wt_GMP_new_bk_10_9',
                             'VAE_design_zeroW_wt_GMP_new_bk_10_10'
                             ]
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

# run(VAE_design_wt_GMP_bk_10_1, net, E1_VAE)

# _____________ VAE 5 10 + class + E1 old weights ________________

VAE_test_5_10_2 = ['VAE_design_new_10_3',
                   'VAE_design_new_10_4',
                   'VAE_design_new_10_11',
                   'VAE_design_new_10_12']
E1_VAE = False
E1_AE = True
E1_second_conv_adapt = True
two_encoder = False
GMP = True
big_kernel = False

latent_spec = {'cont_var': 5, 'cont_class': 10}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
# run(VAE_test_5_10_2, net, E1_VAE)

# ------- Big Kernel: -----------
VAE_test_5_10_2_bk = ['VAE_design_new_bk_10_3',
                      'VAE_design_new_bk_10_4',
                      'VAE_design_new_bk_10_11',
                      'VAE_design_new_bk_10_12']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
run(VAE_test_5_10_2_bk, net, E1_VAE)

# ------- wt GMP: -----------
VAE_design_wt_GMP_new_10_2 = ['VAE_design_wt_GMP_new_10_3',
                              'VAE_design_wt_GMP_new_10_4',
                              'VAE_design_wt_GMP_new_10_11',
                              'VAE_design_wt_GMP_new_10_12',
                              'VAE_design_zeroW_wt_GMP_new_10_3',
                              'VAE_design_zeroW_wt_GMP_new_10_4',
                              'VAE_design_zeroW_wt_GMP_new_10_11',
                              'VAE_design_zeroW_wt_GMP_new_10_12']
big_kernel = False
GMP = False

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

# run(VAE_design_wt_GMP_new_10_2, net, E1_VAE)

# ------- Big Kernel + wt GMP: -----------
VAE_design_wt_GMP_bk_10_2 = [# 'VAE_design_wt_GMP_new_bk_10_3',
                             # 'VAE_design_wt_GMP_new_bk_10_4',
                             # 'VAE_design_wt_GMP_new_bk_10_11',
                             # 'VAE_design_wt_GMP_new_bk_10_12',
                             'VAE_design_zeroW_wt_GMP_new_bk_10_3',
                             'VAE_design_zeroW_wt_GMP_new_bk_10_4',
                             'VAE_design_zeroW_wt_GMP_new_bk_10_11',
                             'VAE_design_zeroW_wt_GMP_new_bk_10_12'
                             ]
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

# run(VAE_design_wt_GMP_bk_10_2, net, E1_VAE)

# _____________ VAE 5 10 + class + E1 old weights ________________

VAE_test_5_10_3 = ['VAE_design_new_10_5',
                   'VAE_design_new_10_6',
                   'VAE_design_new_10_13',
                   'VAE_design_new_10_14']
E1_VAE = True
E1_AE = False
E1_second_conv_adapt = False
two_encoder = True
GMP = True
big_kernel = False

latent_spec = {'cont_var': 5, 'cont_class': 10}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
# run(VAE_test_5_10_3, net, E1_VAE)

# ------- Big Kernel: -----------
VAE_test_5_10_3_bk = ['VAE_design_new_bk_10_5',
                      'VAE_design_new_bk_10_6',
                      'VAE_design_new_bk_10_13',
                      'VAE_design_new_bk_10_14']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
run(VAE_test_5_10_3_bk, net, E1_VAE)

# ------- wt GMP: -----------
VAE_design_wt_GMP_new_10_3 = ['VAE_design_wt_GMP_new_10_5',
                       'VAE_design_wt_GMP_new_10_6',
                       'VAE_design_wt_GMP_new_10_13',
                       'VAE_design_wt_GMP_new_10_14',
                       'VAE_design_zeroW_wt_GMP_new_10_5',
                       'VAE_design_zeroW_wt_GMP_new_10_6',
                       'VAE_design_zeroW_wt_GMP_new_10_13',
                       'VAE_design_zeroW_wt_GMP_new_10_14']
big_kernel = False
GMP = False

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

# run(VAE_design_wt_GMP_new_10_3, net, E1_VAE)

# ------- Big Kernel + wt GMP: -----------
VAE_design_wt_GMP_bk_10_3 = [#  'VAE_design_wt_GMP_new_bk_10_5',
                             #  'VAE_design_wt_GMP_new_bk_10_6',
                             #  'VAE_design_wt_GMP_new_bk_10_13',
                             #  'VAE_design_wt_GMP_new_bk_10_14',
                             # 'VAE_design_zeroW_wt_GMP_new_bk_10_5',
                             # 'VAE_design_zeroW_wt_GMP_new_bk_10_6',
                             # 'VAE_design_zeroW_wt_GMP_new_bk_10_13',
                             'VAE_design_zeroW_wt_GMP_new_bk_10_14'
                             ]
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

run(VAE_design_wt_GMP_bk_10_3, net, E1_VAE)

# _____________ VAE 5 10 + class + E1 old weights ________________

VAE_test_5_10_4 = ['VAE_design_new_10_7',
                   'VAE_design_new_10_8',
                   'VAE_design_new_10_15',
                   'VAE_design_new_10_16']
E1_VAE = False
E1_AE = True
E1_second_conv_adapt = False
two_encoder = True
GMP = True
big_kernel = False

latent_spec = {'cont_var': 5, 'cont_class': 10}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
# run(VAE_test_5_10_4, net, E1_VAE)

# ------- Big Kernel: -----------
VAE_test_5_10_4_bk = ['VAE_design_new_bk_10_7',
                      'VAE_design_new_bk_10_8',
                      'VAE_design_new_bk_10_15',
                      'VAE_design_new_bk_10_16']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
run(VAE_test_5_10_4_bk, net, E1_VAE)

# ------- wt GMP: -----------
VAE_design_wt_GMP_new_10_4 = ['VAE_design_wt_GMP_new_10_7',
                              'VAE_design_wt_GMP_new_10_8',
                              'VAE_design_wt_GMP_new_10_15',
                              'VAE_design_wt_GMP_new_10_16',
                              'VAE_design_zeroW_wt_GMP_new_10_7',
                              'VAE_design_zeroW_wt_GMP_new_10_8',
                              'VAE_design_zeroW_wt_GMP_new_10_15',
                              'VAE_design_zeroW_wt_GMP_new_10_16']
big_kernel = False
GMP = False

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

# run(VAE_design_wt_GMP_new_10_4, net, E1_VAE)

# ------- Big Kernel + wt GMP: -----------
VAE_design_wt_GMP_bk_10_4 = [# 'VAE_design_wt_GMP_new_bk_10_7',
                             # 'VAE_design_wt_GMP_new_bk_10_8',
                             # 'VAE_design_wt_GMP_new_bk_10_15',
                             # 'VAE_design_wt_GMP_new_bk_10_16',
                             'VAE_design_zeroW_wt_GMP_new_bk_10_7',
                             'VAE_design_zeroW_wt_GMP_new_bk_10_8',
                             'VAE_design_zeroW_wt_GMP_new_bk_10_15',
                             'VAE_design_zeroW_wt_GMP_new_bk_10_16'
                             ]
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

run(VAE_design_wt_GMP_bk_10_4, net, E1_VAE)

# -------------------------------- 5 15 -----------------------------------------------

# _____________ VAE 5 15 + class + E1 old weights ________________

VAE_test_5_15_1 = ['VAE_design_new_15_1',
                   'VAE_design_new_15_2',
                   'VAE_design_new_15_9',
                   'VAE_design_new_15_10']
E1_VAE = True
E1_AE = False
E1_second_conv_adapt = True
two_encoder = False
GMP = True
big_kernel = False

is_zvar_sim_loss = False
is_partial_rand_class = False
is_C = True

latent_spec = {'cont_var': 5, 'cont_class': 15}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
# run(VAE_test_5_15_1, net, E1_VAE)

# ------- Big Kernel: -----------
VAE_test_5_15_1_bk = ['VAE_design_new_bk_15_1',
                      'VAE_design_new_bk_15_2',
                      'VAE_design_new_bk_15_9',
                      'VAE_design_new_bk_15_10']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
run(VAE_test_5_15_1_bk, net, E1_VAE)

# ------- wt GMP: -----------
VAE_design_wt_GMP_new_15_1 = [# 'VAE_design_wt_GMP_new_15_1',
                              # 'VAE_design_wt_GMP_new_15_2',
                              # 'VAE_design_wt_GMP_new_15_9',
                              # 'VAE_design_wt_GMP_new_15_10',
                              'VAE_design_zeroW_wt_GMP_new_15_1',
                              'VAE_design_zeroW_wt_GMP_new_15_2',
                              'VAE_design_zeroW_wt_GMP_new_15_9',
                              'VAE_design_zeroW_wt_GMP_new_15_10'
                              ]
big_kernel = False
GMP = False

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

run(VAE_design_wt_GMP_new_15_1, net, E1_VAE)

# ------- Big Kernel + wt GMP: -----------
VAE_design_wt_GMP_bk_15_1 = [# 'VAE_design_wt_GMP_new_bk_15_1',
                             # 'VAE_design_wt_GMP_new_bk_15_2',
                             # 'VAE_design_wt_GMP_new_bk_15_9',
                             # 'VAE_design_wt_GMP_new_bk_15_10',
                             'VAE_design_zeroW_wt_GMP_new_bk_15_1',
                             'VAE_design_zeroW_wt_GMP_new_bk_15_2',
                             'VAE_design_zeroW_wt_GMP_new_bk_15_9',
                             'VAE_design_zeroW_wt_GMP_new_bk_15_10'
                             ]
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

run(VAE_design_wt_GMP_bk_15_1, net, E1_VAE)

# _____________ VAE 5 15 + class + E1 old weights ________________

VAE_test_5_15_2 = ['VAE_design_new_15_3',
                   'VAE_design_new_15_4',
                   'VAE_design_new_15_11',
                   'VAE_design_new_15_12']
E1_VAE = False
E1_AE = True
E1_second_conv_adapt = True
two_encoder = False
GMP = True
big_kernel = False

latent_spec = {'cont_var': 5, 'cont_class': 15}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
# run(VAE_test_5_15_2, net, E1_VAE)

# ------- Big Kernel: -----------
VAE_test_5_15_2_bk = ['VAE_design_new_bk_15_3',
                      'VAE_design_new_bk_15_4',
                      'VAE_design_new_bk_15_11',
                      'VAE_design_new_bk_15_12']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
run(VAE_test_5_15_2_bk, net, E1_VAE)

# ------- wt GMP: -----------
VAE_design_wt_GMP_new_15_2 = [# 'VAE_design_wt_GMP_new_15_3',
                              # 'VAE_design_wt_GMP_new_15_4',
                              # 'VAE_design_wt_GMP_new_15_11',
                              # 'VAE_design_wt_GMP_new_15_12',
                              'VAE_design_zeroW_wt_GMP_new_15_3',
                              'VAE_design_zeroW_wt_GMP_new_15_4',
                              'VAE_design_zeroW_wt_GMP_new_15_11',
                              'VAE_design_zeroW_wt_GMP_new_15_12'
                              ]
big_kernel = False
GMP = False

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

run(VAE_design_wt_GMP_new_15_2, net, E1_VAE)

# ------- Big Kernel + wt GMP: -----------
VAE_design_wt_GMP_bk_15_2 = [# 'VAE_design_wt_GMP_new_bk_15_3',
                             # 'VAE_design_wt_GMP_new_bk_15_4',
                             # 'VAE_design_wt_GMP_new_bk_15_11',
                             # 'VAE_design_wt_GMP_new_bk_15_12',
                             'VAE_design_zeroW_wt_GMP_new_bk_15_3',
                             'VAE_design_zeroW_wt_GMP_new_bk_15_4',
                             'VAE_design_zeroW_wt_GMP_new_bk_15_11',
                             'VAE_design_zeroW_wt_GMP_new_bk_15_12'
                             ]
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

run(VAE_design_wt_GMP_bk_15_2, net, E1_VAE)

# _____________ VAE 5 15 + class + E1 old weights ________________

VAE_test_5_15_3 = ['VAE_design_new_15_5',
                   'VAE_design_new_15_6',
                   'VAE_design_new_15_13',
                   'VAE_design_new_15_14']
E1_VAE = True
E1_AE = False
E1_second_conv_adapt = False
two_encoder = True
GMP = True
big_kernel = False

latent_spec = {'cont_var': 5, 'cont_class': 15}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
# run(VAE_test_5_15_3, net, E1_VAE)

# ------- Big Kernel: -----------
VAE_test_5_15_3_bk = ['VAE_design_new_bk_15_5',
                      'VAE_design_new_bk_15_6',
                      'VAE_design_new_bk_15_13',
                      'VAE_design_new_bk_15_14']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
run(VAE_test_5_15_3_bk, net, E1_VAE)

# ------- wt GMP: -----------
VAE_design_wt_GMP_new_15_3 = [# 'VAE_design_wt_GMP_new_15_5',
                              # 'VAE_design_wt_GMP_new_15_6',
                              # 'VAE_design_wt_GMP_new_15_13',
                              # 'VAE_design_wt_GMP_new_15_14',
                              'VAE_design_zeroW_wt_GMP_new_15_5',
                              'VAE_design_zeroW_wt_GMP_new_15_6',
                              'VAE_design_zeroW_wt_GMP_new_15_13',
                              'VAE_design_zeroW_wt_GMP_new_15_14'
                              ]
big_kernel = False
GMP = False

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

run(VAE_design_wt_GMP_new_15_3, net, E1_VAE)

# ------- Big Kernel + wt GMP: -----------
VAE_design_wt_GMP_bk_15_3 = [# 'VAE_design_wt_GMP_new_bk_15_5',
                             # 'VAE_design_wt_GMP_new_bk_15_6',
                             # 'VAE_design_wt_GMP_new_bk_15_13',
                             # 'VAE_design_wt_GMP_new_bk_15_14',
                             'VAE_design_zeroW_wt_GMP_new_bk_15_5',
                             'VAE_design_zeroW_wt_GMP_new_bk_15_6',
                             'VAE_design_zeroW_wt_GMP_new_bk_15_13',
                             'VAE_design_zeroW_wt_GMP_new_bk_15_14'
                             ]
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

run(VAE_design_wt_GMP_bk_15_3, net, E1_VAE)

# _____________ VAE 5 15 + class + E1 old weights ________________

VAE_test_5_15_4 = ['VAE_design_new_15_7',
                   'VAE_design_new_15_8',
                   'VAE_design_new_15_15',
                   'VAE_design_new_15_16']
E1_VAE = False
E1_AE = True
E1_second_conv_adapt = False
two_encoder = True
GMP = True
big_kernel = False

latent_spec = {'cont_var': 5, 'cont_class': 15}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
# run(VAE_test_5_15_4, net, E1_VAE)

# ------- Big Kernel: -----------
VAE_test_5_15_4_bk = ['VAE_design_new_bk_15_7',
                      'VAE_design_new_bk_15_8',
                      'VAE_design_new_bk_15_15',
                      'VAE_design_new_bk_15_16']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
run(VAE_test_5_15_4_bk, net, E1_VAE)

# ------- wt GMP: -----------
VAE_design_wt_GMP_new_15_4 = [# 'VAE_design_wt_GMP_new_15_7',
                              # 'VAE_design_wt_GMP_new_15_8',
                              # 'VAE_design_wt_GMP_new_15_15',
                              # 'VAE_design_wt_GMP_new_15_16',
                              'VAE_design_zeroW_wt_GMP_new_15_7',
                              'VAE_design_zeroW_wt_GMP_new_15_8',
                              'VAE_design_zeroW_wt_GMP_new_15_15',
                              'VAE_design_zeroW_wt_GMP_new_15_16'
                              ]
big_kernel = False
GMP = False

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

run(VAE_design_wt_GMP_new_15_4, net, E1_VAE)

# ------- Big Kernel + wt GMP: -----------
VAE_design_wt_GMP_bk_15_4 = [# 'VAE_design_wt_GMP_new_bk_15_7',
                             # 'VAE_design_wt_GMP_new_bk_15_8',
                             # 'VAE_design_wt_GMP_new_bk_15_15',
                             # 'VAE_design_wt_GMP_new_bk_15_16',
                             'VAE_design_zeroW_wt_GMP_new_bk_15_7',
                             'VAE_design_zeroW_wt_GMP_new_bk_15_8',
                             'VAE_design_zeroW_wt_GMP_new_bk_15_15',
                             'VAE_design_zeroW_wt_GMP_new_bk_15_16'
                             ]
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

run(VAE_design_wt_GMP_bk_15_4, net, E1_VAE)

# -------------------------------- 5 20 -----------------------------------------------

# _____________ VAE 5 20 + class + E1 old weights ________________

VAE_test_5_20_1 = ['VAE_design_new_20_1',
                   'VAE_design_new_20_2',
                   'VAE_design_new_20_9',
                   'VAE_design_new_20_10']
E1_VAE = True
E1_AE = False
E1_second_conv_adapt = True
two_encoder = False
GMP = True
big_kernel = False

is_zvar_sim_loss = False
is_partial_rand_class = False
is_C = True

latent_spec = {'cont_var': 5, 'cont_class': 20}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
# run(VAE_test_5_20_1, net, E1_VAE)

# ------- Big Kernel: -----------
VAE_test_5_20_1_bk = ['VAE_design_new_bk_20_1',
                      'VAE_design_new_bk_20_2',
                      'VAE_design_new_bk_20_9',
                      'VAE_design_new_bk_20_10']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
# run(VAE_test_5_20_1_bk, net, E1_VAE)

# _____________ VAE 5 20 + class + E1 old weights ________________

VAE_test_5_20_2 = ['VAE_design_new_20_3',
                   'VAE_design_new_20_4',
                   'VAE_design_new_20_11',
                   'VAE_design_new_20_12']
E1_VAE = False
E1_AE = True
E1_second_conv_adapt = True
two_encoder = False
GMP = True
big_kernel = False

latent_spec = {'cont_var': 5, 'cont_class': 20}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
# run(VAE_test_5_20_2, net, E1_VAE)

# ------- Big Kernel: -----------
VAE_test_5_20_2_bk = ['VAE_design_new_bk_20_3',
                      'VAE_design_new_bk_20_4',
                      'VAE_design_new_bk_20_11',
                      'VAE_design_new_bk_20_12']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
# run(VAE_test_5_20_2_bk, net, E1_VAE)

# _____________ VAE 5 20 + class + E1 old weights ________________

VAE_test_5_20_3 = ['VAE_design_new_20_5',
                   'VAE_design_new_20_6',
                   'VAE_design_new_20_13',
                   'VAE_design_new_20_14']
E1_VAE = True
E1_AE = False
E1_second_conv_adapt = False
two_encoder = True
GMP = True
big_kernel = False

latent_spec = {'cont_var': 5, 'cont_class': 20}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
# run(VAE_test_5_20_3, net, E1_VAE)

# ------- Big Kernel: -----------
VAE_test_5_20_3_bk = ['VAE_design_new_bk_20_5',
                      'VAE_design_new_bk_20_6',
                      'VAE_design_new_bk_20_13',
                      'VAE_design_new_bk_20_14']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
# run(VAE_test_5_20_3_bk, net, E1_VAE)

# _____________ VAE 5 20 + class + E1 old weights ________________

VAE_test_5_20_4 = ['VAE_design_new_20_7',
                   'VAE_design_new_20_8',
                   'VAE_design_new_20_15',
                   'VAE_design_new_20_16']
E1_VAE = False
E1_AE = True
E1_second_conv_adapt = False
two_encoder = True
GMP = True
big_kernel = False

latent_spec = {'cont_var': 5, 'cont_class': 20}
BN = True
is_E1 = True
E1_conv = True
second_layer_C = False
E1_second_conv = False
net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)

z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
# run(VAE_test_5_20_4, net, E1_VAE)

# ------- Big Kernel: -----------
VAE_test_5_20_4_bk = ['VAE_design_new_bk_20_7',
                      'VAE_design_new_bk_20_8',
                      'VAE_design_new_bk_20_15',
                      'VAE_design_new_bk_20_16']
big_kernel = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, E1_conv=E1_conv, BN=BN, E1_second_conv=E1_second_conv,
              E1_second_conv_adapt=E1_second_conv_adapt, E1_VAE=E1_VAE, E1_AE=E1_AE, two_encoder=two_encoder,
              GMP=GMP, big_kernel=big_kernel)
# run(VAE_test_5_20_4_bk, net, E1_VAE)
"""

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
              save=True, batch=batch, plot_sample=True, FID=False, IS=True, psnr=False)
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
              save=True, batch=batch, plot_sample=True, FID=False, IS=True, psnr=False)
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
    #           save=True, batch=batch, plot_sample=True, FID=False, IS=True, psnr=False, sample_real=True)

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
    #           save=True, batch=batch, plot_sample=True, FID=False, IS=True, psnr=False)
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
              save=True, batch=batch, plot_sample=True, FID=False, IS=True, psnr=False)
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
    #           save=True, batch=batch, plot_sample=True, FID=False, IS=True, psnr=False, sample_real=True)

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
              save=True, batch=batch, plot_sample=True, FID=False, IS=True, psnr=False)
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
