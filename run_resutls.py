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
              FID=False, IS=False, psnr=False, scores=True,
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

mnist_VAE_class_E1_zvar_sim_5_5 = ['VAE_Cass_E1_Zvarsim_tune_WLr_1', 'VAE_Cass_E1_Zvarsim_tune_WLr_2',
                                   'VAE_Cass_E1_Zvarsim_tune_WLr_3', 'VAE_Cass_E1_Zvarsim_tune_WLr_4',
                                   'VAE_Cass_E1_Zvarsim_tune_WLr_5', 'VAE_Cass_E1_Zvarsim_tune_WLr_6',
                                   'VAE_Cass_E1_Zvarsim_tune_WLr_7', 'VAE_Cass_E1_Zvarsim_tune_WLr_8',
                                   'VAE_Cass_E1_Zvarsim_tune_WLr_9', 'VAE_Cass_E1_Zvarsim_tune_WLr_10']

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
for expe in mnist_VAE_class_E1_zvar_sim_5_5:
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
# _______________________________Expe test vanilla VAE _____________________________________

mnist_VAE_5_5 = ['VAE_5_5_2']
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

# _____________ VAE 5 5 ________________
latent_spec = {'cont_var': 5, 'cont_class': 5}
z_component_traversal = np.arange(latent_spec['cont_var'] + latent_spec['cont_class'])
BN = True

net = BetaVAE(latent_spec, nb_class, is_C, device, nc=nc, four_conv=four_conv, second_layer_C=second_layer_C,
              is_E1=is_E1, BN=BN)

for expe in mnist_VAE_5_5:
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