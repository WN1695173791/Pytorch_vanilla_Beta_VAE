import collections
import torch.nn as nn
from functools import partial
import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset.dataset_2 import get_dataloaders
from viz.get_regions import get_all_regions_max
from visualizer_CNN import get_receptive_field
import os
import cv2
from viz.get_regions_interest import *


def get_activation_values(net, exp_name, images):
    # a dictionary that keeps saving the activations as they come
    activations = collections.defaultdict(list)

    def save_activations(name, mod, inp, out):
        activations[name].append(out.cpu())

    # Registering hooks for all the Conv2d layers
    # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
    # called repeatedly at different stages of the forward pass (like RELUs), this will save different
    # activations. Editing the forward pass code to save activations is the way to go for these cases.
    for name, m in net.named_modules():
        if type(m) == nn.Conv2d:
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activations, name))

    out, _ = net(images)

    # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    activations = {name: torch.cat(outputs, 0) for name, outputs in
                   activations.items()}

    np.save('regions_of_interest/activations/activations_' + exp_name + '.npy', activations)

    return


def load_activations(exp_name):
    """
    load activations ectract for all dataset
    :param exp_name:
    :return:
    """

    activations = np.load('regions_of_interest/activations/activations_' + exp_name + '.npy', allow_pickle=True)

    # just print out the sizes of the saved activations as a sanity check
    for name, fm in activations.item().items():
        print('layer: {} with shape: {}'.format(name, fm.size()))

    return activations


def get_regions(net, activations, len_img_h, len_img_w, images, exp_name):
    """
    extract regions that maximize specific neuron/
    :param net:
    :param activations:
    :param len_img_h:
    :param len_img_w:
    :param images:
    :param img_size:
    :return:
    """

    region_final, activation_final, activation_final_normalized = get_all_regions_max(images,
                                                                                      activations,
                                                                                      len_img_h,
                                                                                      len_img_w,
                                                                                      net)

    if 'net.0' in region_final:
        regions_layer_1 = region_final['net.0']
        activation_layer1 = activation_final['net.0']
        activation_layer1_normalized = activation_final_normalized['net.0']
        print(regions_layer_1.shape)
        print(activation_layer1.shape)
        np.save('regions_of_interest/MNIST_regions/regions_layer1_' + exp_name + '.npy', regions_layer_1)
        np.save('regions_of_interest/MNIST_regions/activations_layer1_' + exp_name + '.npy', activation_layer1)
        np.save('regions_of_interest/MNIST_regions/activations_normalized_layer1_' + exp_name + '.npy',
                activation_layer1_normalized)

    if 'net.3' in region_final:
        regions_layer_2 = region_final['net.3']
        activation_layer2 = activation_final['net.3']
        activation_layer2_normalized = activation_final_normalized['net.3']
        print(regions_layer_2.shape)
        print(activation_layer2.shape)
        np.save('regions_of_interest/MNIST_regions/regions_layer2_' + exp_name + '.npy', regions_layer_2)
        np.save('regions_of_interest/MNIST_regions/activations_layer2_' + exp_name + '.npy', activation_layer2)
        np.save('regions_of_interest/MNIST_regions/activations_normalized_layer2_' + exp_name + '.npy',
                activation_layer2_normalized)

    if 'net.6' in region_final:
        regions_layer_3 = region_final['net.6']
        activation_layer3 = activation_final['net.6']
        activation_layer3_normalized = activation_final_normalized['net.6']
        print(regions_layer_3.shape)
        print(activation_layer3.shape)
        np.save('regions_of_interest/MNIST_regions/regions_layer3_' + exp_name + '.npy', regions_layer_3)
        np.save('regions_of_interest/MNIST_regions/activations_layer3_' + exp_name + '.npy', activation_layer3)
        np.save('regions_of_interest/MNIST_regions/activations_normalized_layer3_' + exp_name + '.npy',
                activation_layer3_normalized)

    if 'net.9' in region_final:
        regions_layer_4 = region_final['net.9']
        activation_layer4 = activation_final['net.9']
        activation_layer4_normalized = activation_final_normalized['net.9']
        print(regions_layer_4.shape)
        print(activation_layer4.shape)
        np.save('regions_of_interest/MNIST_regions/regions_layer4_' + exp_name + '.npy', regions_layer_4)
        np.save('regions_of_interest/MNIST_regions/activations_layer4_' + exp_name + '.npy', activation_layer4)
        np.save('regions_of_interest/MNIST_regions/activations_normalized_layer4_' + exp_name + '.npy',
                activation_layer4_normalized)

    return


def visualize_regions(exp_name, net, len_img_h, len_img_w, plot_activation_value=False):
    # print(net)

    # load labels:
    labels = np.load('regions_of_interest/labels/labels.npy', allow_pickle=True)
    images = torch.tensor(np.load('regions_of_interest/images/images.npy', allow_pickle=True))

    print('loader shape', images.shape)
    # plt.hist(labels)
    # plt.show()

    if not os.path.exists('regions_of_interest/activations/activations_' + exp_name + '.npy'):
        get_activation_values(net, exp_name, images)

    activations = load_activations(exp_name)

    if not os.path.exists('regions_of_interest/MNIST_regions/activations_normalized_layer1_' + exp_name + '.npy'):
        get_regions(net, activations, len_img_h, len_img_w, images, exp_name)

    regions = []
    activations = []
    activations_normalized = []
    if os.path.exists('regions_of_interest/MNIST_regions/regions_layer1_' + exp_name + '.npy'):
        regions_layer_1 = np.load('regions_of_interest/MNIST_regions/regions_layer1_' + exp_name + '.npy',
                                  allow_pickle=True)
        activation_layer1 = np.load('regions_of_interest/MNIST_regions/activations_layer1_' + exp_name +
                                    '.npy', allow_pickle=True)
        activation_layer1_normalized = np.load('regions_of_interest/MNIST_regions/activations_normalized_layer1_' +
                                               exp_name + '.npy', allow_pickle=True)
        regions.append(regions_layer_1)
        activations.append(activation_layer1)
        activations_normalized.append(activation_layer1_normalized)
    if os.path.exists('regions_of_interest/MNIST_regions/regions_layer2_' + exp_name + '.npy'):
        regions_layer_2 = np.load('regions_of_interest/MNIST_regions/regions_layer2_' + exp_name + '.npy',
                                  allow_pickle=True)
        activation_layer2 = np.load('regions_of_interest/MNIST_regions/activations_layer2_' + exp_name +
                                    '.npy', allow_pickle=True)
        activation_layer2_normalized = np.load('regions_of_interest/MNIST_regions/activations_normalized_layer2_' +
                                               exp_name + '.npy', allow_pickle=True)
        regions.append(regions_layer_2)
        activations.append(activation_layer2)
        activations_normalized.append(activation_layer2_normalized)
    if os.path.exists('regions_of_interest/MNIST_regions/regions_layer3_' + exp_name + '.npy'):
        regions_layer_3 = np.load('regions_of_interest/MNIST_regions/regions_layer3_' + exp_name + '.npy',
                                  allow_pickle=True)
        activation_layer3 = np.load('regions_of_interest/MNIST_regions/activations_layer3_' + exp_name +
                                    '.npy', allow_pickle=True)
        activation_layer3_normalized = np.load('regions_of_interest/MNIST_regions/activations_normalized_layer3_' +
                                               exp_name + '.npy', allow_pickle=True)
        regions.append(regions_layer_3)
        activations.append(activation_layer3)
        activations_normalized.append(activation_layer3_normalized)
    if os.path.exists('regions_of_interest/MNIST_regions/regions_layer4_' + exp_name + '.npy'):
        regions_layer_4 = np.load('regions_of_interest/MNIST_regions/regions_layer4_' + exp_name + '.npy',
                                  allow_pickle=True)
        activation_layer4 = np.load('regions_of_interest/MNIST_regions/activations_layer4_' + exp_name +
                                    '.npy', allow_pickle=True)
        activation_layer4_normalized = np.load('regions_of_interest/MNIST_regions/activations_normalized_layer4_' +
                                               exp_name + '.npy', allow_pickle=True)
        regions.append(regions_layer_4)
        activations.append(activation_layer4)
        activations_normalized.append(activation_layer4_normalized)


    th_layer = 0
    for name, m in net.named_modules():
        if type(m) == nn.Conv2d:
            layer_name = name
            nb_filters = m.weight.data.clone().shape[0]
            print('Layer: {}, number of filter: {}'.format(name, nb_filters))
            list_filter = range(nb_filters)
            visualize(regions[th_layer], activations[th_layer], activations_normalized[th_layer],
                      labels, list_filter, net, layer_name, plot_activation_value)
            th_layer += 1

    return


def get_last_regions_activation(exp_name):
    assert os.path.exists('regions_of_interest/MNIST_regions/activations_normalized_layer1_' + exp_name + '.npy'), \
        "regions and activations doesn't exist for this model"

    if os.path.exists('regions_of_interest/MNIST_regions/regions_layer1_' + exp_name + '.npy'):
        region = np.load('regions_of_interest/MNIST_regions/regions_layer1_' + exp_name + '.npy',
                         allow_pickle=True)
        activation = np.load('regions_of_interest/MNIST_regions/activations_layer1_' + exp_name +
                             '.npy', allow_pickle=True)
        activation_normalized = np.load('regions_of_interest/MNIST_regions/activations_normalized_layer1_' +
                                        exp_name + '.npy', allow_pickle=True)
    if os.path.exists('regions_of_interest/MNIST_regions/regions_layer2_' + exp_name + '.npy'):
        region = np.load('regions_of_interest/MNIST_regions/regions_layer2_' + exp_name + '.npy',
                         allow_pickle=True)
        activation = np.load('regions_of_interest/MNIST_regions/activations_layer2_' + exp_name +
                             '.npy', allow_pickle=True)
        activation_normalized = np.load('regions_of_interest/MNIST_regions/activations_normalized_layer2_' +
                                        exp_name + '.npy', allow_pickle=True)
    if os.path.exists('regions_of_interest/MNIST_regions/regions_layer3_' + exp_name + '.npy'):
        region = np.load('regions_of_interest/MNIST_regions/regions_layer3_' + exp_name + '.npy',
                         allow_pickle=True)
        activation = np.load('regions_of_interest/MNIST_regions/activations_layer3_' + exp_name +
                             '.npy', allow_pickle=True)
        activation_normalized = np.load('regions_of_interest/MNIST_regions/activations_normalized_layer3_' +
                                        exp_name + '.npy', allow_pickle=True)
    if os.path.exists('regions_of_interest/MNIST_regions/regions_layer4_' + exp_name + '.npy'):
        region = np.load('regions_of_interest/MNIST_regions/regions_layer4_' + exp_name + '.npy',
                         allow_pickle=True)
        activation = np.load('regions_of_interest/MNIST_regions/activations_layer4_' + exp_name +
                             '.npy', allow_pickle=True)
        activation_normalized = np.load('regions_of_interest/MNIST_regions/activations_normalized_layer4_' +
                                        exp_name + '.npy', allow_pickle=True)

    return region, activation, activation_normalized


def visualize(regions, activation, activations_normalized, labels, list_filter, net, layer_name, plot_activation_value):

    # parameters
    best = True
    worst = False
    viz_mean_img = True
    viz_grid = False  # True
    plot_histogram = False
    details = False  # True

    percentage = 1
    nrow = 14

    # regions and activation of interest
    list_filter = list_filter
    regions = regions
    activations = activation
    activations_normalized = activations_normalized

    _, activation, _ = get_regions_interest_fct(regions,
                                       labels,
                                       activations,
                                       activations_normalized,
                                       details=details,
                                       best=best,
                                       worst=worst,
                                       viz_mean_img=viz_mean_img,
                                       viz_grid=viz_grid,
                                       percentage=percentage,
                                       list_filter=list_filter,
                                       nrow=nrow,
                                       plot_histogram=plot_histogram)

    if plot_activation_value:
        activation = np.array(activation)
        act = np.sum(np.abs(activation), axis=1)
        x = list_filter
        plt.bar(x, act)
        plt.show()
    # regions = np.array(selected_regions)
    # print(regions.shape)
    # print(regions)
    viz_filters(net, nrow, layer_name)

    return


def viz_filters(model, nrow, layer_name):
    for name, m in model.named_modules():
        if name == layer_name:
            filters = m.weight.data.clone()
            filters_normalized = np.zeros(filters.shape)
            for i in range(filters.shape[0]):
                result = filters[i, 0, :, :]
                # print(result)
                # print(torch.min(result), torch.max(result))
                # print(result - torch.min(result))
                result = (result - torch.min(result)) / (torch.max(result) - torch.min(result))
                filters_normalized[i, 0, :, :] = result
            filters_normalized = torch.tensor(filters_normalized)
            visTensor(filters_normalized.cpu(), ch=0, allkernels=False, nrow=nrow)
            plt.ioff()
            print('Visualization filters learned for layer: {}'.format(name))
            plt.show()
            # print(filters)
        else:
            pass
    return


def viz_region_im(exp_name, net, random_index=False, choice_label=None, label=None, nb_im=None, best_region=False,
                  worst_regions=False, any_label=False, average_result=False, plot_activation_value=False):
    """
    :param plot_activation_value:
    :param average_result:
    :param any_label:
    :param worst_regions:
    :param best_region:
    :param nb_im: if is an integer > 0 so select nb_im images else if in a float between 0 and 1, select nb_im % images
    :param choice_label:
    :param label:
    :param exp_name: the model name
    :param net: the network: pytorch model
    :param random_index:
    :return:
    """
    # load labels:
    labels = np.load('regions_of_interest/labels/labels.npy', allow_pickle=True)
    images = torch.tensor(np.load('regions_of_interest/images/images.npy', allow_pickle=True))
    # load activation model
    path_load_activation = 'regions_of_interest/MNIST_regions/activations_normalized_layer1_' + exp_name + '.npy'
    assert os.path.exists(path_load_activation), "Activation doesn't extracted, please run 'get_activation_values'"

    region, activation, activation_normalized = get_last_regions_activation(exp_name)
    nb_filter = activation.shape[-1]
    im = None
    if random_index:
        index_selected = np.random.randint(len(images))
        regions_selected = region[index_selected]
        activation_values = activation[index_selected]
        activation_normalized_values = activation_normalized[index_selected]
        im = images[index_selected]
    elif choice_label:
        assert label is not None, 'label must be a positive integer'
        assert nb_im is not None and nb_im >= 0, 'nb_im must be a positive integer'
        # select index:
        index = np.where(labels == label)[0]
        index_len = len(index)
        region = region[index]
        activation = activation[index]
        activation_normalized = activation_normalized[index]
        images = images[index]
        # select number of images:
        if 0 < nb_im < 1:
            nb_im = int(index_len * nb_im)
        else:
            assert 1 <= nb_im < index_len, 'choose a valid image number'
            nb_im = nb_im
        print('select {} images for label {}'.format(nb_im, label))

        if best_region:
            # select range:
            regions_selected = []
            activation_values = []
            activation_normalized_values = []
            for i in range(nb_filter):
                index_selected = (-activation[:, i]).argsort()[:nb_im]
                regions_selected.append(region[index_selected, i])
                activation_values.append(activation[index_selected, i])
                activation_normalized_values.append(activation_normalized[index_selected, i])
            if average_result:
                regions_selected = np.expand_dims(np.mean(regions_selected, axis=0), axis=0)
        elif worst_regions:
            # select range:
            regions_selected = []
            activation_values = []
            activation_normalized_values = []
            for i in range(nb_filter):
                index_selected = activation[:, i].argsort()[:nb_im]
                regions_selected.append(region[index_selected, i])
                activation_values.append(activation[index_selected, i])
                activation_normalized_values.append(activation_normalized[index_selected, i])
            if average_result:
                regions_selected = np.expand_dims(np.mean(regions_selected, axis=0), axis=0)
        else:
            index_selected = np.random.randint(0, index_len, nb_im)
            regions_selected = region[index_selected]
            activation_values = activation[index_selected]
            activation_normalized_values = activation_normalized[index_selected]
            im = images[index_selected]
            if average_result:
                regions_selected = np.expand_dims(np.mean(regions_selected, axis=0), axis=0)
    elif any_label:
        assert nb_im is not None and nb_im >= 0, 'nb_im must be a positive integer'
        # select index:
        index_len = len(labels)
        # select number of images:
        if 0 < nb_im < 1:
            nb_im = int(index_len * nb_im)
        else:
            assert 1 <= nb_im < index_len, 'choose a valid image number'
            nb_im = nb_im
        print('select {} images for label {}'.format(nb_im, label))
        # select range:
        if best_region:
            regions_selected = []
            activation_values = []
            activation_normalized_values = []
            for i in range(nb_filter):
                index_selected = (-activation[:, i]).argsort()[:nb_im]
                regions_selected.append(region[index_selected, i])
                activation_values.append(activation[index_selected, i])
                activation_normalized_values.append(activation_normalized[index_selected, i])
            if average_result:
                regions_selected = np.expand_dims(np.mean(regions_selected, axis=0), axis=0)
        elif worst_regions:
            regions_selected = []
            activation_values = []
            activation_normalized_values = []
            for i in range(nb_filter):
                index_selected = activation[:, i].argsort()[:nb_im]
                regions_selected.append(region[index_selected, i])
                activation_values.append(activation[index_selected, i])
                activation_normalized_values.append(activation_normalized[index_selected, i])
            if average_result:
                regions_selected = np.expand_dims(np.mean(regions_selected, axis=0), axis=0)
        else:
            index_selected = np.random.randint(0, index_len, nb_im)
            regions_selected = region[index_selected]
            activation_values = activation[index_selected]
            activation_normalized_values = activation_normalized[index_selected]
            if average_result:
                regions_selected = np.expand_dims(np.mean(regions_selected, axis=0), axis=0)

    if im is not None:
        viz_regions(len(im), im, nrow=8, save=False, path_save=None)
        plt.show()

    nrow = 8
    for i in range(len(regions_selected)):
        # plot regions extracted:
        region = torch.tensor(regions_selected[i])
        region = region.reshape((len(regions_selected[i]), 1,
                                                     region.shape[-2],
                                                     region.shape[-1]))
        if plot_activation_value:
            if average_result:
                act = np.sum(np.abs(activation_values), axis=0)
            else:
                act = np.abs(activation_values[i])

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), facecolor='w', edgecolor='k')

            # plot bar activations value
            x = np.arange(nb_filter)
            ax2.bar(x, act)

            # plot regions extracted:
            tensor = region[:, 0, :, :].unsqueeze(dim=1)
            rows = np.min((tensor.shape[0] // nrow + 1, 64))
            grid = make_grid(tensor, nrow=nrow, normalize=True, padding=1, pad_value=1)
            plt.figure(figsize=(nrow, rows))
            ax1.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.ioff()
            plt.show()
        else:
            visTensor(regions_selected, ch=0, allkernels=False, nrow=8, save=False, path_save=None)
            plt.ioff()
            plt.show()

    return
