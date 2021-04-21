import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
# from torch_receptive_field import receptive_field
from torchvision.utils import make_grid
from scores_classifier import compute_scores
import torch.nn.functional as F
from visualizer import *
from torch.autograd import Variable
from viz.latent_traversal import LatentTraverser
import importlib
import collections
from functools import partial

from scipy.spatial import distance

lpips_exists = importlib.util.find_spec("lpips") is not None
if lpips_exists:
    import lpips
from torchvision import transforms
from scipy.stats import norm

EPS = 1e-12


def compute_scores_pred(prediction, labels):
    """
    return nb of correct prediction for the current batch
    :param prediction:
    :param labels:
    :return:
    """
    predicted = prediction.argmax(dim=1, keepdim=True)
    correct = predicted.eq(labels.view_as(predicted)).sum().item()
    scores = correct / len(labels)
    return float(scores)


def get_layer_zstruct_num(net, add_layer=3):
    # get layer num for GMP:
    for name, m in net.named_modules():
        if type(m) == nn.AdaptiveMaxPool2d:
            z_struct_layer_num = int(name.split('.')[-1])

    return z_struct_layer_num + add_layer


def compute_z_struct_mean_VAE(net_trained, exp_name, loader, train_test=None, return_results=False):
    """
    Extract all z_struct representation for a specific model and all images in loader and save it.
    :param net_type:
    :param return_results:
    :param net_trained:
    :param exp_name:
    :param loader:
    :param custom_CNN:
    :param train_test:
    :return:
    """

    path_save = 'structural_representation/z_struct_representation_VAE_' + exp_name + '_' + train_test + '.npy'

    if os.path.exists(path_save):
        print("path already exist")
        return
    else:
        labels_list = []
        z_struct_representation = []
        prediction = []

        for data, labels in loader:
            # evaluation mode:
            net_trained.eval()

            with torch.no_grad():
                input_data = data
            if torch.cuda.is_available():
                input_data = input_data.cuda()

            x_recons, z_struct, z_var, z_var_sample, latent_representation, z, _ = net_trained(input_data)

            # train mode:
            net_trained.eval()

            labels_list.extend(labels.cpu().detach().numpy())
            z_struct_batch = z_struct.squeeze().cpu().detach().numpy()
            z_struct_representation.extend(z_struct_batch)

        z_struct_representation = np.array(z_struct_representation)
        labels_list = np.array(labels_list)

        if return_results:
            return z_struct_representation, labels_list
        else:
            np.save('structural_representation/z_struct_representation_VAE_' + exp_name + '_' + train_test + '.npy',
                    z_struct_representation)
            print('save z_struct representation for model {}'.format(exp_name))
            np.save('structural_representation/label_list_VAE_' + exp_name + '_' + train_test + '.npy',
                    labels_list)
            print('save label list')

            return


def get_z_struct_per_class_VAE(exp_name, train_test=None, nb_class=10):
    """
    return z_struct sort by classes with average z_struct for each class.
    :param exp_name:
    :param train_test:
    :param nb_class:
    :return:
    """

    path_save = 'structural_representation/z_struct_representation_per_class_VAE_' + exp_name + '_' + \
                train_test + '.npy'
    if os.path.exists(path_save):
        print("path already exist")
        representation_z_struct_class = np.load(path_save, allow_pickle=True)
        average_z_struct_class = np.load(
            'structural_representation/average_z_struct_representation_per_class_VAE_' + exp_name + '_' + \
            train_test + '.npy', allow_pickle=True)
        return representation_z_struct_class, average_z_struct_class
    else:
        path = 'structural_representation/z_struct_representation_VAE_' + exp_name + '_' + train_test + '.npy'

        z_struct_representation = np.load(path, allow_pickle=True)
        label_list = np.load('structural_representation/label_list_VAE_' + exp_name + '_' + train_test + '.npy',
                             allow_pickle=True)

        nb_class = nb_class
        representation_z_struct_class = []
        average_z_struct_class = []
        for class_id in range(nb_class):
            z_struct_class = z_struct_representation[np.where(label_list == class_id)]
            representation_z_struct_class.append(z_struct_class)
            average_z_struct_class.append(np.mean(z_struct_class, axis=0))

        representation_z_struct_class = np.array(representation_z_struct_class)
        average_z_struct_class = np.array(average_z_struct_class)

        np.save(path_save, representation_z_struct_class)
        print('save z_struct representation per class for model {}'.format(exp_name))

        np.save('structural_representation/average_z_struct_representation_per_class_VAE_' + exp_name + '_' + \
                train_test + '.npy', average_z_struct_class)

        print('save z_struct representation per class for model {}'.format(exp_name))
        print('save average z_struct representation per class for model {}'.format(exp_name))

    return representation_z_struct_class, average_z_struct_class


def compute_z_struct(net_trained, exp_name, loader, train_test=None, net_type=None, return_results=False,
                     bin_after_GMP=False):
    """
    Extract all z_struct representation for a specific model and all images in loader and save it.
    :param net_type:
    :param return_results:
    :param net_trained:
    :param exp_name:
    :param loader:
    :param custom_CNN:
    :param train_test:
    :return:
    """

    path_save = 'structural_representation/z_struct_representation_' + exp_name + '__' + train_test + '.npy'

    if os.path.exists(path_save):
        print("path already exist")
        return
    else:
        # get layer num for GMP:
        z_struct_layer_num = get_layer_zstruct_num(net_trained)

        labels_list = []
        z_struct_representation = []
        prediction = []
        for data, labels in loader:
            # evaluation mode:
            net_trained.eval()

            with torch.no_grad():
                input_data = data
            if torch.cuda.is_available():
                input_data = input_data.cuda()

            _, z_struct, _, _, _, _, _, _, _, _, _, _ = net_trained(input_data,
                                                           z_struct_out=True,
                                                           z_struct_layer_num=z_struct_layer_num)
            pred, _, _, _, _, _, _, _, _, _, _, _ = net_trained(input_data)

            # train mode:
            net_trained.eval()

            labels_list.extend(labels.cpu().detach().numpy())
            z_struct_batch = z_struct.squeeze().cpu().detach().numpy()
            z_struct_representation.extend(z_struct_batch)
            prediction.extend(pred.squeeze().cpu().detach().numpy())

        z_struct_representation = np.array(z_struct_representation)
        labels_list = np.array(labels_list)
        prediction = np.array(prediction)
        # print(z_struct_representation.shape)  # shape: (nb_data, z_struct_dim)
        # print(labels_list.shape)
        # print(prediction.shape)

        if return_results:
            return z_struct_representation, labels_list
        else:
            np.save('structural_representation/z_struct_representation_' + exp_name + '_' + train_test + '.npy',
                    z_struct_representation)
            print('save z_struct representation for model {}'.format(exp_name))
            np.save('structural_representation/label_list_' + exp_name + '_' + train_test + '.npy',
                    labels_list)
            print('save label list')
            np.save('structural_representation/prediction_' + exp_name + '_' + train_test + '.npy',
                    prediction)
            print('save prediction')

            return


def compute_z_struct_representation_noised(net, exp_name, train_test=None, nb_repeat=100, nb_class=10, net_type=None,
                                           bin_after_GMP=False):
    """
    Compute prediction mean for each element of z_struct noised.
    :param train_test:
    :param exp_name:
    :param net:
    :param nb_repeat:
    :param nb_class:
    :return: np.array of shape: (z_struct_size, nb_images, nb_class)
    """

    prediction_path = "structural_representation/prediction_bit_noised/prediction_bit_noised_" + exp_name + \
                      train_test + ".npy"
    if os.path.exists(prediction_path):
        print("path already exist")
        return
    else:
        # get layer num for GMP:
        z_struct_layer_num = get_layer_zstruct_num(net)
        z_struct_representation, label_list, _ = load_z_struct_representation(exp_name, train_test=train_test)

        z_struct_size = z_struct_representation.shape[-1]
        z_struct_representation = torch.tensor(z_struct_representation)
        std_z_struct_max = z_struct_representation.max(axis=1)[0] - z_struct_representation.min(axis=1)[0]
        mean_z_struct = z_struct_representation.mean(axis=1)

        # eval mode
        net.eval()

        prediction_noised = []
        for i in range(z_struct_size):
            prediction = []
            for rep in range(nb_repeat):
                z_struct_representation_noised = torch.tensor(z_struct_representation)
                z_struct_representation_noised[:, i] = std_z_struct_max[i] * torch.randn(
                    (z_struct_representation.shape[0])) \
                                                       + mean_z_struct[i]
                pred, _, _, _, _, _, _, _ = net(z_struct_representation_noised, z_struct_prediction=True,
                                             z_struct_layer_num=z_struct_layer_num)
                prediction.append(pred.detach().numpy())
            prediction_noised.append(np.mean(np.array(prediction), axis=0))

        prediction_noised = np.array(prediction_noised)
        np.save(prediction_path, prediction_noised)
        print('save prediction_noised for model {}, shape: {}'.format(exp_name, prediction_noised.shape))

    return


def get_z_struct_per_class(exp_name, train_test=None, nb_class=10):
    """
    return z_struct sort by classes with average z_struct for each class.
    :param exp_name:
    :param train_test:
    :param nb_class:
    :return:
    """

    path_save = 'structural_representation/z_struct_representation_per_class_' + exp_name + '_' + \
                train_test + '.npy'
    if os.path.exists(path_save):
        print("path already exist")
        return
    else:
        z_struct_representation, label_list, _ = load_z_struct_representation(exp_name, train_test=train_test)

        nb_class = nb_class
        representation_z_struct_class = []
        for class_id in range(nb_class):
            z_struct_class = z_struct_representation[np.where(label_list == class_id)]
            representation_z_struct_class.append(z_struct_class)

        representation_z_struct_class = np.array(representation_z_struct_class)

        np.save(path_save, representation_z_struct_class)
        print('save z_struct representation per class for model {}'.format(exp_name))

        return


def get_average_z_struct_per_classes(exp_name, train_test=None):
    """
    return z_struct sort by classes with average z_struct for each class.
    :param train_test:
    :param exp_name:
    :return:
    """

    path_save = 'structural_representation/average_z_struct/average_z_struct_representation_' + exp_name + '_' + \
                train_test + '.npy'
    if os.path.exists(path_save):
        print("path already exist")
        return
    else:
        representation_z_struct_class = load_z_struct_representation_per_class(exp_name, train_test=train_test)

        average_representation_z_struct_class = []
        std_representation_z_struct_class = []
        for i in range(len(representation_z_struct_class)):
            average_representation_z_struct_class.append(np.mean(representation_z_struct_class[i], axis=0))
            std_representation_z_struct_class.append(np.std(representation_z_struct_class[i], axis=0))

        average_representation_z_struct_class = np.array(average_representation_z_struct_class)
        std_representation_z_struct_class = np.array(std_representation_z_struct_class)

        np.save(path_save, average_representation_z_struct_class)
        print('save average z_struct representation for model {}, shape:'
              ' {}'.format(exp_name, average_representation_z_struct_class.shape))
        np.save('structural_representation/average_z_struct/std_z_struct_representation_' + exp_name + '_' + \
                train_test + '.npy', std_representation_z_struct_class)
        print('save std z_struct representation for model {}, shaape: '
              '{}'.format(exp_name, std_representation_z_struct_class.shape))

        return


def get_prediction_per_classes(exp_name, train_test=None):
    """
    return z_struct sort by classes with average z_struct for each class.
    :param exp_name:
    :param train_test:
    :return:
    """

    path_save = 'structural_representation/prediction_per_class_' + exp_name + '_' + train_test + '.npy'
    if os.path.exists(path_save):
        print("path already exist")
        return
    else:
        _, label_list, prediction = load_z_struct_representation(exp_name, train_test=train_test)

        nb_class = prediction.shape[-1]
        prediction_class = []

        for class_id in range(nb_class):
            pred_class = prediction[np.where(label_list == class_id)]
            prediction_class.append(pred_class)

        prediction_class = np.array(prediction_class)

        np.save(path_save, prediction_class)
        print('save prediction per class from model {}'.format(exp_name))

        return


def get_prediction_noised_per_class(exp_name, train_test=None):
    """
    Return all prediction for each class with one bit noised and this for each bit of z_struct
    :param label_list:
    :param exp_name:
    :param train_test:
    :return:
    """
    path_save = 'structural_representation/prediction_bit_noised/prediction_noised_per_class' + exp_name + '_' + \
                train_test + '.npy'

    if os.path.exists(path_save):
        print("path already exist")
        return
    else:
        _, label_list, _ = load_z_struct_representation(exp_name, train_test=train_test)
        noised_prediction = load_prediction_noised(exp_name, train_test=train_test)

        nb_class = noised_prediction.shape[-1]
        prediction_per_class = []

        for class_id in range(nb_class):
            prediction_per_class_class_id = noised_prediction[:, np.where(label_list == class_id)].squeeze(axis=1)
            prediction_per_class.append(np.transpose(prediction_per_class_class_id, axes=[1, 0, 2]))

        prediction_per_class = np.array(prediction_per_class)
        np.save('structural_representation/prediction_bit_noised/prediction_noised_per_class' + exp_name + '_' + \
                train_test + '.npy', prediction_per_class)
        print('save prediction noised per class for model {}'.format(exp_name))

        return


def compute_all_score_acc(exp_name, train_test=None):
    path = 'structural_representation/score/score_prediction_' + exp_name + '_' + \
           train_test + '.npy'

    if os.path.exists(path):
        print("path already exist")
        return
    else:
        # load predictions:
        _, label_list, prediction = load_z_struct_representation(exp_name, train_test=train_test)
        prediction_per_class = load_prediction_per_class(exp_name, train_test=train_test)
        prediction_noised = load_prediction_noised(exp_name, train_test=train_test)
        prediction_noised_per_class = load_prediction_noised_per_class(exp_name, train_test=train_test)

        # score for prediction: shape: int
        prediction = torch.tensor(prediction)
        label_list = torch.tensor(label_list)
        score_z_struct_all_class = compute_scores_pred(prediction, label_list)

        # score for prediction noised: shape: (len_z_struct, int)
        prediction_noised = torch.tensor(prediction_noised)
        score_prediction_noised = []
        for i in range(len(prediction_noised)):
            score = compute_scores_pred(prediction_noised[i], label_list)
            score_prediction_noised.append(score)
        score_prediction_noised = np.array(score_prediction_noised)

        # score prediction per class, shape: (n_class, score_int)
        score_z_struct_each_class = []
        for class_id in range(len(prediction_per_class)):
            pred = torch.tensor(prediction_per_class[class_id])
            label_list = torch.ones(len(pred)) * class_id
            score = compute_scores_pred(pred, label_list)
            score_z_struct_each_class.append(score)

        score_z_struct_each_class = np.array(score_z_struct_each_class)

        # score prediction noised per class, shape: (len_z_struct, nb_class, int)
        score_z_struct_noised_each_class = []
        for class_id in range(len(prediction_noised_per_class)):
            pred = torch.tensor(np.transpose(prediction_noised_per_class[class_id], axes=[1, 0, 2]))
            label_list = torch.ones(pred.shape[1]) * class_id
            score_z_struct_noised_each_class_iter = []
            for i in range(len(pred)):
                score = compute_scores_pred(pred[i], label_list)
                score_z_struct_noised_each_class_iter.append(score)
            score_z_struct_noised_each_class.append(score_z_struct_noised_each_class_iter)
        score_z_struct_noised_each_class = np.array(score_z_struct_noised_each_class)

        np.save('structural_representation/score/score_prediction_' + exp_name + '_' + \
                train_test + '.npy', score_z_struct_all_class)
        np.save('structural_representation/score/score_prediction_per_class_' + exp_name + '_' + \
                train_test + '.npy', score_z_struct_each_class)
        np.save('structural_representation/score/score_prediction_noised_' + exp_name + '_' + \
                train_test + '.npy', score_prediction_noised)
        np.save('structural_representation/score/score_prediction_noised_per_class_' + exp_name + '_' + \
                train_test + '.npy', score_z_struct_noised_each_class)
        print('save all score for model {}'.format(exp_name))

    return


def compute_mean_std_prediction(exp_name, train_test=None):
    path = 'structural_representation/score/prediction_mean_' + exp_name + '_' + \
           train_test + '.npy'

    if os.path.exists(path):
        print("path already exist")
        return
    else:
        _, label_list, prediction = load_z_struct_representation(exp_name, train_test=train_test)
        prediction_noised_per_class = load_prediction_noised_per_class(exp_name, train_test=train_test)
        prediction_per_class = load_prediction_per_class(exp_name, train_test=train_test)
        noised_prediction = load_prediction_noised(exp_name, train_test=train_test)

        # get parameters:
        nb_bit = len(noised_prediction)
        nb_class = len(prediction_per_class)

        # get mean and std prediction noised:
        mean_prediction_noised_acc_per_class = []
        std_prediction_noised_acc_per_class = []
        mean_prediction_acc_per_class = []
        std_prediction_acc_per_class = []

        # get mean and std of all prediction:
        score_prediction = []
        for class_id in range(nb_class):
            score = prediction[np.where(label_list == class_id), class_id].squeeze(axis=0)
            score_prediction.append(score)

        score_prediction = np.array([item for sublist in score_prediction for item in sublist])
        mean_prediction_acc = np.mean(score_prediction)
        std_prediction_acc = np.std(score_prediction)

        # prediction noised:
        mean_prediction_noised_acc = []
        std_prediction_noised_acc = []
        for i in range(nb_bit):
            score_prediction_noised_iter = []
            pred = noised_prediction[i]
            for class_id in range(nb_class):
                score = pred[np.where(label_list == class_id), class_id].squeeze(axis=0)
                score_prediction_noised_iter.append(score)
            score_prediction_noised_iter = np.array(
                [item for sublist in score_prediction_noised_iter for item in sublist])
            mean_prediction_noised_acc_iter = np.mean(score_prediction_noised_iter)
            std_prediction_noised_acc_iter = np.std(score_prediction_noised_iter)
            mean_prediction_noised_acc.append(mean_prediction_noised_acc_iter)
            std_prediction_noised_acc.append(std_prediction_noised_acc_iter)

        mean_prediction_noised_acc = np.array(mean_prediction_noised_acc)
        std_prediction_noised_acc = np.array(std_prediction_noised_acc)

        # prediction per class:
        for class_id in range(nb_class):
            mean_prediction_noised_acc_per_class.append(np.mean(prediction_noised_per_class[class_id][:, :, class_id],
                                                                axis=0))
            std_prediction_noised_acc_per_class.append(np.std(prediction_noised_per_class[class_id][:, :, class_id],
                                                              axis=0))
            mean_prediction_acc_per_class.append(np.mean(prediction_per_class[class_id], axis=0))
            std_prediction_acc_per_class.append(np.std(prediction_per_class[class_id], axis=0))

        mean_prediction_noised_acc_per_class = np.array(
            mean_prediction_noised_acc_per_class)  # shape: nb_class, nb_bit, nb_class
        std_prediction_noised_acc_per_class = np.array(std_prediction_noised_acc_per_class)
        mean_prediction_acc_per_class = np.array(mean_prediction_acc_per_class)
        std_prediction_acc_per_class = np.array(std_prediction_acc_per_class)

        np.save('structural_representation/score/prediction_mean_' + exp_name + '_' + \
                train_test + '.npy', mean_prediction_acc)
        np.save('structural_representation/score/prediction_std_' + exp_name + '_' + \
                train_test + '.npy', std_prediction_acc)
        np.save('structural_representation/score/prediction_noised_mean_' + exp_name + '_' + \
                train_test + '.npy', mean_prediction_noised_acc)
        np.save('structural_representation/score/prediction_noised_std_' + exp_name + '_' + \
                train_test + '.npy', std_prediction_noised_acc)

        np.save('structural_representation/score/prediction_mean_per_class_' + exp_name + '_' + \
                train_test + '.npy', mean_prediction_acc_per_class)
        np.save('structural_representation/score/prediction_std_per_class_' + exp_name + '_' + \
                train_test + '.npy', std_prediction_acc_per_class)
        np.save('structural_representation/score/prediction_noised_mean_per_class_' + exp_name + '_' + \
                train_test + '.npy', mean_prediction_noised_acc_per_class)
        np.save('structural_representation/score/prediction_noised_std_per_class_' + exp_name + '_' + \
                train_test + '.npy', std_prediction_noised_acc_per_class)

        print('save prediction mean and std for model {}'.format(exp_name))

    return


def load_z_struct_representation(exp_name, train_test=None):
    """
    laod z_struct representation and label list.
    :param exp_name:
    :param train_test:
    :return:
    """

    path = 'structural_representation/z_struct_representation_' + exp_name + '_' + train_test + '.npy'

    assert os.path.exists(path), "path doesn't exist, run compute_z_struct_per_class to extract z_struct"

    z_struct_representation = np.load(path, allow_pickle=True)
    label_list = np.load('structural_representation/label_list_' + exp_name + '_' + train_test + '.npy',
                         allow_pickle=True)
    prediction = np.load('structural_representation/prediction_' + exp_name + '_' + train_test + '.npy',
                         allow_pickle=True)

    print('load z_struct representation, shape: {}'.format(z_struct_representation.shape))
    print('load label list, shape: {}'.format(label_list.shape))
    print('load prediction, shape: {}'.format(prediction.shape))

    return z_struct_representation, label_list, prediction


def load_average_z_struct_representation(exp_name, train_test=None):
    """
    laod z_struct representation and label list.
    :param exp_name:
    :param train_test:
    :return:
    """

    path = 'structural_representation/average_z_struct/average_z_struct_representation_' + exp_name + '_' + \
           train_test + '.npy'
    assert os.path.exists(path), "path doesn't exist, run get_average_z_struct_per_classes to have average z_struct"

    average_representation_z_struct_class = np.load(path, allow_pickle=True)
    std_representation_z_struct_class = np.load('structural_representation/average_z_struct/std_z_struct_'
                                                'representation_' + exp_name + '_' + train_test + '.npy',
                                                allow_pickle=True)
    print('load average z_struct representation, shape: {}'.format(average_representation_z_struct_class.shape))
    print('load std z_struct representation, shape: {}'.format(std_representation_z_struct_class.shape))

    return average_representation_z_struct_class, std_representation_z_struct_class


def load_z_struct_representation_per_class(exp_name, train_test=None):
    """
    laod z_struct representation and label list.
    :param exp_name:
    :param train_test:
    :return:
    """

    path = 'structural_representation/z_struct_representation_per_class_' + exp_name + '_' + \
           train_test + '.npy'
    assert os.path.exists(path), "path doesn't exist, run get_z_struct_per_class to have z_struct per class"

    representation_z_struct_class = np.load(path, allow_pickle=True)
    print('load z_struct representation per class, shape: {}'.format(representation_z_struct_class.shape))

    return representation_z_struct_class


def load_prediction_noised(exp_name, train_test=None):
    path = "structural_representation/prediction_bit_noised/prediction_bit_noised_" + exp_name + \
           train_test + ".npy"
    assert os.path.exists(path), "path doesn't exist, run compute_z_struct_representation_noised to have prediction"

    prediction_noised = np.load(path, allow_pickle=True)

    print('load prediction noised, shape: {}'.format(prediction_noised.shape))

    return prediction_noised


def load_prediction_noised_per_class(exp_name, train_test=None):
    path = 'structural_representation/prediction_bit_noised/prediction_noised_per_class' + exp_name + '_' + \
           train_test + '.npy'
    assert os.path.exists(path), "path doesn't exist, run get_prediction_noised_per_class to have prediction"

    prediction_per_class = np.load(path, allow_pickle=True)

    print('load prediction noised per class, dictionary')

    return prediction_per_class


def load_prediction_per_class(exp_name, train_test=None):
    path = 'structural_representation/prediction_per_class_' + exp_name + '_' + train_test + '.npy'
    assert os.path.exists(path), "path doesn't exist, run get_prediction_noised_per_class to have prediction"

    prediction_per_class = np.load(path, allow_pickle=True)

    print('load prediction per class')

    return prediction_per_class


def load_scores(exp_name, train_test=None):
    score_z_struct_all_class = np.load('structural_representation/score/score_prediction_' + exp_name + '_' + \
                                       train_test + '.npy', allow_pickle=True)
    score_z_struct_each_class = np.load('structural_representation/score/score_prediction_per_class_' + exp_name + '_' + \
                                        train_test + '.npy', allow_pickle=True)
    score_prediction_noised = np.load('structural_representation/score/score_prediction_noised_' + exp_name + '_' + \
                                      train_test + '.npy', allow_pickle=True)
    score_z_struct_noised_each_class = np.load(
        'structural_representation/score/score_prediction_noised_per_class_' + exp_name + '_' + \
        train_test + '.npy', allow_pickle=True)

    print("load all scores for model {}".format(exp_name))

    return score_z_struct_all_class, score_z_struct_each_class, score_prediction_noised, \
           score_z_struct_noised_each_class


def load_mean_std_prediction(exp_name, train_test=None):
    """
    laod mean and std of prediction tensors for original z_struct and noised z_struct.
    :param exp_name:
    :param train_test:
    :return:
    """

    mean_prediction_acc = np.load('structural_representation/score/prediction_mean_' + exp_name + '_' + \
                                  train_test + '.npy', allow_pickle=True)
    std_prediction_acc = np.load('structural_representation/score/prediction_std_' + exp_name + '_' + \
                                 train_test + '.npy', allow_pickle=True)
    mean_prediction_noised_acc = np.load('structural_representation/score/prediction_noised_mean_' + exp_name + '_' + \
                                         train_test + '.npy', allow_pickle=True)
    std_prediction_noised_acc = np.load('structural_representation/score/prediction_noised_std_' + exp_name + '_' + \
                                        train_test + '.npy', allow_pickle=True)

    mean_prediction_noised_acc_per_class = np.load(
        'structural_representation/score/prediction_mean_per_class_' + exp_name + '_' + \
        train_test + '.npy', allow_pickle=True)
    std_prediction_noised_acc_per_class = np.load(
        'structural_representation/score/prediction_std_per_class_' + exp_name + '_' + \
        train_test + '.npy', allow_pickle=True)
    mean_prediction_acc_per_class = np.load(
        'structural_representation/score/prediction_noised_mean_per_class_' + exp_name + '_' + \
        train_test + '.npy', allow_pickle=True)
    std_prediction_acc_per_class = np.load(
        'structural_representation/score/prediction_noised_std_per_class_' + exp_name + '_' + \
        train_test + '.npy', allow_pickle=True)

    return mean_prediction_acc, std_prediction_acc, mean_prediction_noised_acc, std_prediction_noised_acc, \
           mean_prediction_noised_acc_per_class, std_prediction_noised_acc_per_class, mean_prediction_acc_per_class, \
           std_prediction_acc_per_class


def plot_2d_projection_z_struct(nb_class, exp_name, train_test=None, ratio=None):
    """
    Projection 2d of all z_struct for each class.
    :param nb_class:
    :param exp_name:
    :param train_test:
    :return:
    """
    z_struct_representation, label_list, _ = load_z_struct_representation(exp_name, train_test=train_test)

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(z_struct_representation)
    t = reduced.transpose()

    lim_min = np.min(t)
    lim_max = np.max(t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), facecolor='w', edgecolor='k')
    fig.suptitle('PCA of z_struct for each images, model: {}. With ratio: {}'.format(exp_name, str(ratio)))

    ax1.set(xlim=(lim_min, lim_max), ylim=(lim_min, lim_max))

    x = np.arange(nb_class)
    ys = [i + x + (i * x) ** 2 for i in range(10)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))

    for c, color in zip(range(nb_class), colors):
        x = np.array(t[0])[np.concatenate(np.argwhere(np.array(label_list) == c)).ravel()]
        y = np.array(t[1])[np.concatenate(np.argwhere(np.array(label_list) == c)).ravel()]
        ax1.scatter(x, y, alpha=0.6, color=color, label='class ' + str(c))

    # Explained variance ratio:
    sc = StandardScaler()
    sc.fit(z_struct_representation)
    X_train_std = sc.transform(z_struct_representation)

    pca = PCA()
    X_train_pca = pca.fit_transform(X_train_std)
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    ax2.bar(range(0, len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center',
            label='Individual explained variance')
    ax2.step(range(0, len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',
             label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    ax2.legend(loc='best')
    plt.tight_layout()
    ax1.legend(loc=1)
    plt.show()

    fig.savefig('fig_results/PCA/z_struct_2D_projections/2d_projection_' + exp_name + '_' + train_test + '.png')

    return


def plot_acc_bit_noised_per_class(exp_name, plot_all_classes=False, train_test=None, plot_prediction_mean_std=False,
                                  plot_score_global=False, plot_each_class_score=False,
                                  plot_each_class_prediction_mean_std=False, plot_loss=False,
                                  separate_plot_class=False, cat=None):
    """
    Plot acc score for bit noised for each class.
    Compute also the loss between original score and noised score.
    :param train_test: 
    :param plot_all_classes:
    :param exp_name:
    :return:
    """
    score_z_struct_all_class, score_z_struct_each_class, score_prediction_noised, \
    score_z_struct_noised_each_class = load_scores(exp_name, train_test=train_test)

    mean_prediction_acc, std_prediction_acc, mean_prediction_noised_acc, \
    std_prediction_noised_acc, mean_prediction_noised_acc_per_class, \
    std_prediction_noised_acc_per_class, mean_prediction_acc_per_class, \
    std_prediction_acc_per_class = load_mean_std_prediction(exp_name, train_test=train_test)

    # get parameters:
    nb_bit = len(score_prediction_noised)
    nb_class = len(score_z_struct_each_class)

    # define rainbow colors:
    x = np.arange(nb_class)
    ys = [i + x + (i * x) ** 2 for i in range(10)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))

    # compute loss score or prediction:
    max_loss = score_z_struct_all_class - np.min(score_prediction_noised)
    bit_max_loss = np.argmin(score_prediction_noised)

    # loss between score global for all data and score global for all data noised
    global_score_loss = score_z_struct_all_class - score_prediction_noised

    # loss between score global of all data and score for each class noised
    global_score_loss_per_class = score_z_struct_all_class - score_z_struct_noised_each_class
    # loss between score for each class and score for each class noised
    each_class_score_loss_per_class = np.repeat(np.expand_dims(score_z_struct_each_class, axis=-1),
                                                score_z_struct_noised_each_class.shape[-1], axis=-1) - \
                                      score_z_struct_noised_each_class

    # save loss per category of z_struct len:
    path_save_global_loss_mean = 'values_CNNs_to_compare/' + str(cat) + '/loss_mean_of_means_' + exp_name + \
                                 train_test + '.npy'
    if not os.path.exists(path_save_global_loss_mean):
        np.save(path_save_global_loss_mean, np.mean(global_score_loss))
    # save loss max of each component and average it:
    path_save_mean_loss_max = 'values_CNNs_to_compare/' + str(cat) + '/loss_mean_of_max_component_loss_' + exp_name \
                              + train_test + '.npy'
    if not os.path.exists(path_save_mean_loss_max):
        np.save(path_save_mean_loss_max, np.mean(np.max(each_class_score_loss_per_class, axis=1)))
    # save loss max of max of each component:
    path_save_max_loss_max = 'values_CNNs_to_compare/' + str(cat) + '/loss_max_of_max_component_loss_' + exp_name \
                             + train_test + '.npy'
    if not os.path.exists(path_save_max_loss_max):
        np.save(path_save_max_loss_max, np.max(np.max(each_class_score_loss_per_class, axis=1)))

    if plot_loss:
        # loss between score global for all data and score global for all data noised
        max_loss = global_score_loss.max()
        bit_loss_max = np.argmax(global_score_loss)

        # fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
        # ax.set(xlabel='bit noised', ylabel='Loss',
        #        title=('Model: {}, loss of score (%) global (all data). Loss max: {}, for bit: {}'.format(exp_name,
        #                                                                                                  max_loss,
        #                                                                                                  bit_loss_max)))
        # ax.bar(np.arange(nb_bit), global_score_loss)
        # plt.show()

        fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
        ax.set(xlabel='bit noised', ylabel='Loss',
               title=('Model: {}, loss of score (%) global (all data).'.format(exp_name, max_loss, bit_loss_max)))
        ax.bar(np.arange(nb_bit), global_score_loss, color='gray', alpha=0.5, label='loss all data')
        for class_id, color in zip(range(nb_class), colors):
            ax.scatter(np.arange(nb_bit), each_class_score_loss_per_class[class_id],
                       label='class' + str(class_id), color=color)
        ax.legend(loc=1)
        plt.show()

    if plot_prediction_mean_std:
        # plot prediction global:
        fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
        ax.set(xlabel='bit noised', ylabel='accuracy (%)',
               title=('Model: {}, prediction'.format(exp_name)), xlim=(-1, nb_bit))  # , ylim=(0.5, 1.1))

        ax.plot(np.arange(nb_bit), np.ones(nb_bit) * mean_prediction_acc,
                label='mean without noised', color='y')
        ax.fill_between(np.arange(nb_bit), np.ones(nb_bit) * mean_prediction_acc +
                        std_prediction_acc,
                        np.ones(nb_bit) * mean_prediction_acc -
                        std_prediction_acc,
                        alpha=0.5, color='y')
        ax.errorbar(np.arange(nb_bit), mean_prediction_noised_acc, std_prediction_noised_acc,
                    fmt='ok', lw=3, label='mean with noised')
        plt.show()

    if plot_score_global:
        # plot score: score_z_struct_all_class, , score_prediction_noised,
        fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
        ax.set(xlabel='bit noised', ylabel='accuracy (%)',
               title=('Model: {}, prediction with max loss = {} for bit {}'.format(exp_name, max_loss,
                                                                                   bit_max_loss)),
               xlim=(-1, nb_bit))

        ax.plot(np.arange(nb_bit), np.ones(nb_bit) * score_z_struct_all_class,
                label='mean without noised', color='red', lw=5)
        ax.scatter(np.arange(nb_bit), score_prediction_noised, lw=10, color='black', label='mean with noised')
        for class_id, color in zip(range(nb_class), colors):
            ax.scatter(np.arange(nb_bit), score_z_struct_noised_each_class[class_id],
                       label='class' + str(class_id), color=color, lw=5, alpha=0.5)
        ax.legend(loc=4)
        plt.show()

    if plot_all_classes:
        if plot_each_class_score:
            if not separate_plot_class:
                fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
                ax.set(xlabel='bit noised', ylabel='accuracy (%)',
                       title=('Model: {}, prediction'.format(exp_name)), xlim=(-1, nb_bit))  # , ylim=(0.5, 1.1))
            for class_id, color in zip(range(nb_class), colors):
                if separate_plot_class:
                    fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
                    ax.set(xlabel='bit noised', ylabel='accuracy (%)',
                           title=('Model: {}, prediction, class: {}'.format(exp_name,
                                                                            class_id)), xlim=(-1, nb_bit))
                ax.plot(np.arange(nb_bit), np.ones(nb_bit) * score_z_struct_each_class[class_id],
                        label=str(class_id), color=color)
                ax.scatter(np.arange(nb_bit), score_z_struct_noised_each_class[class_id],
                           lw=3, color=color)
                if separate_plot_class:
                    ax.legend(loc=4)
                    plt.show()
            ax.legend(loc=4)
            plt.show()

        if plot_each_class_prediction_mean_std:
            if not separate_plot_class:
                fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
                ax.set(xlabel='bit noised', ylabel='accuracy (%)',
                       title=('Model: {}, prediction'.format(exp_name)), xlim=(-1, nb_bit))  # , ylim=(0.5, 1.1))
            for class_id, color in zip(range(nb_class), colors):
                if separate_plot_class:
                    fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
                    ax.set(xlabel='bit noised', ylabel='accuracy (%)',
                           title=('Model: {}, prediction, class: {}'.format(exp_name, class_id)), xlim=(-1, nb_bit))

                mean = mean_prediction_acc_per_class[class_id][class_id]
                std = std_prediction_acc_per_class[class_id][class_id]
                mean_noised = mean_prediction_noised_acc_per_class[class_id]
                std_noised = std_prediction_noised_acc_per_class[class_id]

                ax.plot(np.arange(nb_bit), np.ones(nb_bit) * mean,
                        label=str(class_id), color=color)
                ax.fill_between(np.arange(nb_bit), np.ones(nb_bit) * mean + std,
                                np.ones(nb_bit) * mean - std,
                                alpha=0.5,
                                color=color)

                ax.errorbar(np.arange(nb_bit), mean_noised, std_noised,
                            fmt='ok', lw=3, ecolor=color)
                if separate_plot_class:
                    ax.legend(loc=4)
                    plt.show()
            ax.legend(loc=4)
            plt.show()

    return


def dispersion_classes(exp_name, train_test=None, plot_fig=False, cat=None):
    """
    compute dispersion inter/intra class.
    Plot correlation between z_struct mean of each class.
    See if the z_struct "prototype" per class are similar (close to 1) or different (close to 0).
    :return:
    """

    average_representation_z_struct_class, _ = load_average_z_struct_representation(exp_name, train_test=train_test)
    print(average_representation_z_struct_class[0].shape)

    cov_data = np.corrcoef(average_representation_z_struct_class)

    # compute score dispersion classes:
    cov_data = np.array(cov_data)
    # get triangle upper of matrix:
    cov_data_triu = np.triu(cov_data, k=1)
    # mean of nonzero values:
    coef_mean_all_real = np.mean(cov_data_triu[np.nonzero(cov_data_triu)])

    path_save = 'values_CNNs_to_compare/' + str(cat) + '/correlation_score_z_struct_mean_class_' + exp_name + \
                train_test + '.npy'
    np.save(path_save, coef_mean_all_real)

    if plot_fig:
        fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
        ax.set(title=('Model: {}, covariance matrix mean: {}'.format(exp_name, coef_mean_all_real)))

        img = ax.matshow(cov_data, cmap=plt.cm.rainbow)
        plt.colorbar(img, ticks=[-1, 0, 1], fraction=0.045)
        for x in range(cov_data.shape[0]):
            for y in range(cov_data.shape[1]):
                plt.text(x, y, "%0.2f" % cov_data[x, y], size=12, color='black', ha="center", va="center")
        plt.show()

    print('dispersion classes for model {}: {}'.format(exp_name, coef_mean_all_real))

    return coef_mean_all_real


def ratio(exp_name, train_test=None, cat=None, other_ratio=False, normalized=False):
    """
    compute the ratio between std of one class and std of all data.
    :param exp_name:
    :param train_test:
    :return:
    """
    z_struct_representation, label_list, _ = load_z_struct_representation(exp_name, train_test=train_test)
    representation_z_struct_class = load_z_struct_representation_per_class(exp_name, train_test=train_test)
    nb_class = len(representation_z_struct_class)

    if normalized:
        # we normalized z_struct to compare variacne value objectively
        print('normalized z_struct')
        for i in range(len(z_struct_representation)):
            norm = np.linalg.norm(z_struct_representation[i], ord=1)
            z_struct_representation[i] = z_struct_representation[i] / norm
        # min_norm = z_struct_representation.min()
        # max_norm = z_struct_representation.max()
        # representation_z_struct_class = (representation_z_struct_class - min_norm) / (max_norm - min_norm)

    # print(representation_z_struct_class[0][125])

    z_struct_mean_global_per_class = []
    z_struct_std_global_per_class = []
    for class_id in range(nb_class):
        z_struct_mean_global_per_class.append(np.mean(representation_z_struct_class[class_id], axis=0))
        z_struct_std_global_per_class.append(np.std(representation_z_struct_class[class_id], axis=0))

    z_struct_mean_global_per_class = np.array(z_struct_mean_global_per_class)
    z_struct_std_global_per_class = np.array(z_struct_std_global_per_class)

    # print(z_struct_mean_global_per_class)
    # print(z_struct_std_global_per_class)

    # compute ratio inter-class/intra-class:
    # save loss max of max of each component:
    path_save_ratio_variance = 'values_CNNs_to_compare/' + str(cat) + '/ratio_variance_' + exp_name \
                               + train_test + '.npy'
    path_save_variance = 'values_CNNs_to_compare/' + str(cat) + '/variance_intra_class_' + exp_name \
                         + train_test + '.npy'

    if not os.path.exists(path_save_ratio_variance):

        variance_intra_class = np.square(z_struct_std_global_per_class)  # shape: (nb_class, len(z_struct))
        variance_intra_class_mean_components = np.mean(variance_intra_class, axis=0)  # shape: (len(z_struct))
        variance_inter_class = np.square(np.std(z_struct_mean_global_per_class, axis=0))  # shape: len(z_struct)
        if other_ratio:
            ratio_variance = variance_inter_class / (
                    variance_intra_class_mean_components + EPS)  # shape: (len(z_struct))
        else:
            ratio_variance = variance_intra_class_mean_components / (variance_inter_class + EPS)
        ratio_variance_mean = np.mean(ratio_variance)

        print('Var intra class shape: ', variance_intra_class.shape, variance_intra_class_mean_components.shape)
        print('Var inter class shape: ', variance_inter_class.shape)
        print('ratio Var intra / Var inter: ', ratio_variance.shape, ratio_variance_mean)

        np.save(path_save_ratio_variance, ratio_variance_mean)
        np.save(path_save_variance, variance_intra_class)
        np.save('values_CNNs_to_compare/' + str(cat) + '/variance_inter_class_' + exp_name \
                + train_test + '.npy', variance_inter_class)
    else:
        ratio_variance_mean = np.load(path_save_ratio_variance, allow_pickle=True)
        variance_intra_class = np.load(path_save_variance, allow_pickle=True)
        variance_inter_class = np.load('values_CNNs_to_compare/' + str(cat) + '/variance_inter_class_' + exp_name \
                                       + train_test + '.npy', allow_pickle=True)
        print('ratio (Var intra / Var inter) for model {}: {}'.format(exp_name, ratio_variance_mean))
        print('variance_intra_class shape for model {}: {}'.format(exp_name, variance_intra_class.shape))

    return ratio_variance_mean, variance_intra_class, variance_inter_class


def get_filter_id(net, exp_name):
    for name, m in net.named_modules():
        if type(m) == nn.Conv2d:
            filter_id = int(name.split('.')[-1])

    return filter_id


def correlation_filters(net, exp_name, train_test=None, ch=1, vis_filters=False, plot_fig=False, cat=None):
    """
    Plot correlation between filters for the last layer before z_struct.
    If the correlation score if close to 0 the filters are all different and they don't redundancy.
    But if correlation score if close to 1, some filters are same and they are redundancy.
    :param net:
    :param exp_name:
    :param train_test:
    :param ch:
    :return:
    """

    filter_id = get_filter_id(net, exp_name)
    weight_tensor = net.net[filter_id].weight.data.clone()
    tensor = weight_tensor[:, ch, :, :]
    filters = np.array(tensor).reshape((tensor.shape[0], tensor.shape[1] * tensor.shape[2]))

    cov_data = np.corrcoef(filters)

    # compute score dispersion filters:
    cov_data = np.array(cov_data)
    # get triangle upper of matrix:
    cov_data_triu = np.triu(cov_data, k=1)
    # mean of nonzero values:
    coef_mean_all_real = np.mean(cov_data_triu[np.nonzero(cov_data_triu)])

    path_save = 'values_CNNs_to_compare/' + str(cat) + '/correlation_score_filters_' + exp_name + train_test + '.npy'
    np.save(path_save, coef_mean_all_real)

    if plot_fig:
        fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
        ax.set(title=('Model: {}, covariance matrix of filters, mean: {}'.format(exp_name, coef_mean_all_real)))

        img = ax.matshow(cov_data, cmap=plt.cm.rainbow)
        plt.colorbar(img, ticks=[-1, 0, 1], fraction=0.045)
        # for x in range(cov_data.shape[0]):
        #     for y in range(cov_data.shape[1]):
        #         plt.text(x, y, "%0.2f" % cov_data[x, y], size=12, color='black', ha="center", va="center")
        plt.show()

    if vis_filters:
        viz_filters(net, 8)

    print('correlation filter for model {}: {}'.format(exp_name, coef_mean_all_real))

    return coef_mean_all_real


def viz_filters(model, nrow):
    """
    Pot filter trained.
    :param model:
    :param nrow:
    :return:
    """
    for name, m in model.named_modules():
        if type(m) == nn.Conv2d:
            filters = m.weight.data.clone()
            visTensor(filters.cpu(), ch=0, allkernels=False, nrow=nrow)
            plt.ioff()
            print('Visualization filters learned for layer: {}'.format(name))
            plt.show()
    return


def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1, save=False, path_save=None):
    n, c, w, h = tensor.shape
    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = make_grid(tensor, nrow=nrow, normalize=True, padding=padding, pad_value=1)
    plt.figure(figsize=(nrow, rows))
    fig = plt.gcf()
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    return


def load_correlation_score_filters():
    # load score correlation filters cat 5:
    score_correlation_filters_cat_5 = []
    score_correlation_filters_cat_5_names = []
    path = 'values_CNNs_to_compare/zstruct_5/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'correlation_score_filters_' in i:
            score_correlation_filters_cat_5.append(
                np.load('values_CNNs_to_compare/zstruct_5/' + i, allow_pickle=True))
            name = i.split('correlation_score_filters_')[-1].split('test')[0]
            score_correlation_filters_cat_5_names.append(name)
    score_correlation_filters_cat_5 = np.array(score_correlation_filters_cat_5)
    score_correlation_filters_cat_5_names = np.array(score_correlation_filters_cat_5_names)

    # load score correlation filters cat 10:
    score_correlation_filters_cat_10 = []
    score_correlation_filters_cat_10_names = []
    path = 'values_CNNs_to_compare/zstruct_10/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'correlation_score_filters_' in i:
            score_correlation_filters_cat_10.append(
                np.load('values_CNNs_to_compare/zstruct_10/' + i, allow_pickle=True))
            name = i.split('correlation_score_filters_')[-1].split('test')[0]
            score_correlation_filters_cat_10_names.append(name)
    score_correlation_filters_cat_10 = np.array(score_correlation_filters_cat_10)
    score_correlation_filters_cat_10_names = np.array(score_correlation_filters_cat_10_names)

    # load score correlation filters cat 20:
    score_correlation_filters_cat_20 = []
    score_correlation_filters_cat_20_names = []
    path = 'values_CNNs_to_compare/zstruct_20/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'correlation_score_filters_' in i:
            score_correlation_filters_cat_20.append(
                np.load('values_CNNs_to_compare/zstruct_20/' + i, allow_pickle=True))
            name = i.split('correlation_score_filters_')[-1].split('test')[0]
            score_correlation_filters_cat_20_names.append(name)
    score_correlation_filters_cat_20 = np.array(score_correlation_filters_cat_20)
    score_correlation_filters_cat_20_names = np.array(score_correlation_filters_cat_20_names)

    # load score correlation filters cat 50:
    score_correlation_filters_cat_50 = []
    score_correlation_filters_cat_50_names = []
    path = 'values_CNNs_to_compare/zstruct_50/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'correlation_score_filters_' in i:
            score_correlation_filters_cat_50.append(
                np.load('values_CNNs_to_compare/zstruct_50/' + i, allow_pickle=True))
            name = i.split('correlation_score_filters_')[-1].split('test')[0]
            score_correlation_filters_cat_50_names.append(name)
    score_correlation_filters_cat_50 = np.array(score_correlation_filters_cat_50)
    score_correlation_filters_cat_50_names = np.array(score_correlation_filters_cat_50_names)

    return score_correlation_filters_cat_5, score_correlation_filters_cat_10, score_correlation_filters_cat_20, \
           score_correlation_filters_cat_50, score_correlation_filters_cat_5_names, \
           score_correlation_filters_cat_10_names, score_correlation_filters_cat_20_names, \
           score_correlation_filters_cat_50_names


def load_correlation_score_class():
    # load score correlation filters cat 5:
    score_correlation_class_cat_5 = []
    score_correlation_class_cat_5_names = []
    path = 'values_CNNs_to_compare/zstruct_5/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'correlation_score_z_struct_mean_class_' in i:
            score_correlation_class_cat_5.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('correlation_score_z_struct_mean_class_')[-1].split('test')[0]
            score_correlation_class_cat_5_names.append(name)
    score_correlation_class_cat_5 = np.array(score_correlation_class_cat_5)
    score_correlation_class_cat_5_names = np.array(score_correlation_class_cat_5_names)

    # load score correlation class cat 10:
    score_correlation_class_cat_10 = []
    score_correlation_class_cat_10_names = []
    path = 'values_CNNs_to_compare/zstruct_10/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'correlation_score_z_struct_mean_class_' in i:
            score_correlation_class_cat_10.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('correlation_score_z_struct_mean_class_')[-1].split('test')[0]
            score_correlation_class_cat_10_names.append(name)
    score_correlation_class_cat_10 = np.array(score_correlation_class_cat_10)
    score_correlation_class_cat_10_names = np.array(score_correlation_class_cat_10_names)

    # load score correlation class cat 20:
    score_correlation_class_cat_20 = []
    score_correlation_class_cat_20_names = []
    path = 'values_CNNs_to_compare/zstruct_20/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'correlation_score_z_struct_mean_class_' in i:
            score_correlation_class_cat_20.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('correlation_score_z_struct_mean_class_')[-1].split('test')[0]
            score_correlation_class_cat_20_names.append(name)
    score_correlation_class_cat_20 = np.array(score_correlation_class_cat_20)
    score_correlation_class_cat_20_names = np.array(score_correlation_class_cat_20_names)

    # load score correlation class cat 50:
    score_correlation_class_cat_50 = []
    score_correlation_class_cat_50_names = []
    path = 'values_CNNs_to_compare/zstruct_50/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'correlation_score_z_struct_mean_class_' in i:
            score_correlation_class_cat_50.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('correlation_score_z_struct_mean_class_')[-1].split('test')[0]
            score_correlation_class_cat_50_names.append(name)
    score_correlation_class_cat_50 = np.array(score_correlation_class_cat_50)
    score_correlation_class_cat_50_names = np.array(score_correlation_class_cat_50_names)

    return score_correlation_class_cat_5, score_correlation_class_cat_10, score_correlation_class_cat_20, \
           score_correlation_class_cat_50, score_correlation_class_cat_5_names, score_correlation_class_cat_10_names, \
           score_correlation_class_cat_20_names, score_correlation_class_cat_50_names


def load_ratio_scores():
    # load score correlation filters cat 5:
    ratio_mean_global_cat_5 = []
    ratio_mean_global_cat_5_names = []
    path = 'values_CNNs_to_compare/zstruct_5/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'std_global_' in i:
            ratio_mean_global_cat_5.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('std_global_')[-1].split('test')[0]
            ratio_mean_global_cat_5_names.append(name)
    ratio_mean_global_cat_5 = np.array(ratio_mean_global_cat_5)
    ratio_mean_global_cat_5_names = np.array(ratio_mean_global_cat_5_names)

    # load score correlation class cat 10:
    ratio_mean_global_cat_10 = []
    ratio_mean_global_cat_10_names = []
    path = 'values_CNNs_to_compare/zstruct_10/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'std_global_' in i:
            ratio_mean_global_cat_10.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('std_global_')[-1].split('test')[0]
            ratio_mean_global_cat_10_names.append(name)
    ratio_mean_global_cat_10 = np.array(ratio_mean_global_cat_10)
    ratio_mean_global_cat_10_names = np.array(ratio_mean_global_cat_10_names)

    # load score correlation class cat 20:
    ratio_mean_global_cat_20 = []
    ratio_mean_global_cat_20_names = []
    path = 'values_CNNs_to_compare/zstruct_20/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'std_global_' in i:
            ratio_mean_global_cat_20.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('std_global_')[-1].split('test')[0]
            ratio_mean_global_cat_20_names.append(name)
    ratio_mean_global_cat_20 = np.array(ratio_mean_global_cat_20)
    ratio_mean_global_cat_20_names = np.array(ratio_mean_global_cat_20_names)

    # load score correlation class cat 50:
    ratio_mean_global_cat_50 = []
    ratio_mean_global_cat_50_names = []
    path = 'values_CNNs_to_compare/zstruct_50/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'std_global_' in i:
            ratio_mean_global_cat_50.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('std_global_')[-1].split('test')[0]
            ratio_mean_global_cat_50_names.append(name)
    ratio_mean_global_cat_50 = np.array(ratio_mean_global_cat_50)
    ratio_mean_global_cat_50_names = np.array(ratio_mean_global_cat_50_names)

    return ratio_mean_global_cat_5, ratio_mean_global_cat_10, ratio_mean_global_cat_20, ratio_mean_global_cat_50, \
           ratio_mean_global_cat_5_names, ratio_mean_global_cat_10_names, ratio_mean_global_cat_20_names, \
           ratio_mean_global_cat_50_names


def load_ratio_variance():
    # load ratio_variance filters cat 5:
    ratio_variance_cat_5 = []
    ratio_variance_cat_5_names = []
    path = 'values_CNNs_to_compare/zstruct_5/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'ratio_variance_' in i:
            ratio_variance_cat_5.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('ratio_variance_')[-1].split('test')[0]
            ratio_variance_cat_5_names.append(name)
    ratio_variance_cat_5 = np.array(ratio_variance_cat_5)
    ratio_variance_cat_5_names = np.array(ratio_variance_cat_5_names)

    # load score correlation class cat 10:
    ratio_variance_cat_10 = []
    ratio_variance_cat_10_names = []
    path = 'values_CNNs_to_compare/zstruct_10/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'ratio_variance_' in i:
            ratio_variance_cat_10.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('ratio_variance_')[-1].split('test')[0]
            ratio_variance_cat_10_names.append(name)
    ratio_variance_cat_10 = np.array(ratio_variance_cat_10)
    ratio_variance_cat_10_names = np.array(ratio_variance_cat_10_names)

    # load score correlation class cat 20:
    ratio_variance_cat_20 = []
    ratio_variance_cat_20_names = []
    path = 'values_CNNs_to_compare/zstruct_20/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'ratio_variance_' in i:
            ratio_variance_cat_20.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('ratio_variance_')[-1].split('test')[0]
            ratio_variance_cat_20_names.append(name)

    ratio_variance_cat_20 = np.array(ratio_variance_cat_20)
    ratio_variance_cat_20_names = np.array(ratio_variance_cat_20_names)

    # load score correlation class cat 50:
    ratio_variance_cat_50 = []
    ratio_variance_cat_50_names = []
    path = 'values_CNNs_to_compare/zstruct_50/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'ratio_variance_' in i:
            ratio_variance_cat_50.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('ratio_variance_')[-1].split('test')[0]
            ratio_variance_cat_50_names.append(name)
    ratio_variance_cat_50 = np.array(ratio_variance_cat_50)
    ratio_variance_cat_50_names = np.array(ratio_variance_cat_50_names)

    return ratio_variance_cat_5, ratio_variance_cat_10, ratio_variance_cat_20, ratio_variance_cat_50, \
           ratio_variance_cat_5_names, ratio_variance_cat_10_names, ratio_variance_cat_20_names, \
           ratio_variance_cat_50_names


def load_loss_mean():
    # loss mean:
    mean_loss_global_cat_5 = []
    mean_loss_global_cat_5_names = []
    path = 'values_CNNs_to_compare/zstruct_5/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'loss_mean_of_means_' in i:
            mean_loss_global_cat_5.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('loss_mean_of_means_')[-1].split('test')[0]
            mean_loss_global_cat_5_names.append(name)
    mean_loss_global_cat_5 = np.array(mean_loss_global_cat_5)
    mean_loss_global_cat_5_names = np.array(mean_loss_global_cat_5_names)

    mean_loss_global_cat_10 = []
    mean_loss_global_cat_10_names = []
    path = 'values_CNNs_to_compare/zstruct_10/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'loss_mean_of_means_' in i:
            mean_loss_global_cat_10.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('loss_mean_of_means_')[-1].split('test')[0]
            mean_loss_global_cat_10_names.append(name)
    mean_loss_global_cat_10 = np.array(mean_loss_global_cat_10)
    mean_loss_global_cat_10_names = np.array(mean_loss_global_cat_10_names)

    mean_loss_global_cat_20 = []
    mean_loss_global_cat_20_names = []
    path = 'values_CNNs_to_compare/zstruct_20/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'loss_mean_of_means_' in i:
            mean_loss_global_cat_20.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('loss_mean_of_means_')[-1].split('test')[0]
            mean_loss_global_cat_20_names.append(name)
    mean_loss_global_cat_20 = np.array(mean_loss_global_cat_20)
    mean_loss_global_cat_20_names = np.array(mean_loss_global_cat_20_names)

    mean_loss_global_cat_50 = []
    mean_loss_global_cat_50_names = []
    path = 'values_CNNs_to_compare/zstruct_50/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'loss_mean_of_means_' in i:
            mean_loss_global_cat_50.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('loss_mean_of_means_')[-1].split('test')[0]
            mean_loss_global_cat_50_names.append(name)
    mean_loss_global_cat_50 = np.array(mean_loss_global_cat_50)
    mean_loss_global_cat_50_names = np.array(mean_loss_global_cat_50_names)

    # loss max mean by component:
    mean_max_loss_by_component_cat_5 = []
    mean_max_loss_by_component_cat_5_names = []
    path = 'values_CNNs_to_compare/zstruct_5/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'loss_mean_of_max_component_loss_' in i:
            mean_max_loss_by_component_cat_5.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('loss_mean_of_max_component_loss_')[-1].split('test')[0]
            mean_max_loss_by_component_cat_5_names.append(name)
    mean_max_loss_by_component_cat_5 = np.array(mean_max_loss_by_component_cat_5)
    mean_max_loss_by_component_cat_5_names = np.array(mean_max_loss_by_component_cat_5_names)

    mean_max_loss_by_component_cat_10 = []
    mean_max_loss_by_component_cat_10_names = []
    path = 'values_CNNs_to_compare/zstruct_10/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'loss_mean_of_max_component_loss_' in i:
            mean_max_loss_by_component_cat_10.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('loss_mean_of_max_component_loss_')[-1].split('test')[0]
            mean_max_loss_by_component_cat_10_names.append(name)
    mean_max_loss_by_component_cat_10 = np.array(mean_max_loss_by_component_cat_10)
    mean_max_loss_by_component_cat_10_names = np.array(mean_max_loss_by_component_cat_10_names)

    mean_max_loss_by_component_cat_20 = []
    mean_max_loss_by_component_cat_20_names = []
    path = 'values_CNNs_to_compare/zstruct_20/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'loss_mean_of_max_component_loss_' in i:
            mean_max_loss_by_component_cat_20.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('loss_mean_of_max_component_loss_')[-1].split('test')[0]
            mean_max_loss_by_component_cat_20_names.append(name)
    mean_max_loss_by_component_cat_20 = np.array(mean_max_loss_by_component_cat_20)
    mean_max_loss_by_component_cat_20_names = np.array(mean_max_loss_by_component_cat_20_names)

    mean_max_loss_by_component_cat_50 = []
    mean_max_loss_by_component_cat_50_names = []
    path = 'values_CNNs_to_compare/zstruct_50/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'loss_mean_of_max_component_loss_' in i:
            mean_max_loss_by_component_cat_50.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('loss_mean_of_max_component_loss_')[-1].split('test')[0]
            mean_max_loss_by_component_cat_50_names.append(name)
    mean_max_loss_by_component_cat_50 = np.array(mean_max_loss_by_component_cat_50)
    mean_max_loss_by_component_cat_50_names = np.array(mean_max_loss_by_component_cat_50_names)

    # loss max of max loss:
    max_max_loss_by_component_cat_5 = []
    max_max_loss_by_component_cat_5_names = []
    path = 'values_CNNs_to_compare/zstruct_5/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'loss_max_of_max_component_loss_' in i:
            max_max_loss_by_component_cat_5.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('loss_max_of_max_component_loss_')[-1].split('test')[0]
            max_max_loss_by_component_cat_5_names.append(name)
    max_max_loss_by_component_cat_5 = np.array(max_max_loss_by_component_cat_5)
    max_max_loss_by_component_cat_5_names = np.array(max_max_loss_by_component_cat_5_names)

    max_max_loss_by_component_cat_10 = []
    max_max_loss_by_component_cat_10_names = []
    path = 'values_CNNs_to_compare/zstruct_10/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'loss_max_of_max_component_loss_' in i:
            max_max_loss_by_component_cat_10.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('loss_max_of_max_component_loss_')[-1].split('test')[0]
            max_max_loss_by_component_cat_10_names.append(name)
    max_max_loss_by_component_cat_10 = np.array(max_max_loss_by_component_cat_10)
    max_max_loss_by_component_cat_10_names = np.array(max_max_loss_by_component_cat_10_names)

    max_max_loss_by_component_cat_20 = []
    max_max_loss_by_component_cat_20_names = []
    path = 'values_CNNs_to_compare/zstruct_20/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'loss_max_of_max_component_loss_' in i:
            max_max_loss_by_component_cat_20.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('loss_max_of_max_component_loss_')[-1].split('test')[0]
            max_max_loss_by_component_cat_20_names.append(name)
    max_max_loss_by_component_cat_20 = np.array(max_max_loss_by_component_cat_20)
    max_max_loss_by_component_cat_20_names = np.array(max_max_loss_by_component_cat_20_names)

    max_max_loss_by_component_cat_50 = []
    max_max_loss_by_component_cat_50_names = []
    path = 'values_CNNs_to_compare/zstruct_50/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path, i)) and 'loss_max_of_max_component_loss_' in i:
            max_max_loss_by_component_cat_50.append(
                np.load(path + i, allow_pickle=True))
            name = i.split('loss_max_of_max_component_loss_')[-1].split('test')[0]
            max_max_loss_by_component_cat_50_names.append(name)
    max_max_loss_by_component_cat_50 = np.array(max_max_loss_by_component_cat_50)
    max_max_loss_by_component_cat_50_names = np.array(max_max_loss_by_component_cat_50_names)

    return mean_loss_global_cat_5, mean_loss_global_cat_10, mean_loss_global_cat_20, mean_loss_global_cat_50, \
           mean_max_loss_by_component_cat_5, mean_max_loss_by_component_cat_10, mean_max_loss_by_component_cat_20, \
           mean_max_loss_by_component_cat_50, mean_loss_global_cat_5_names, mean_loss_global_cat_10_names, \
           mean_loss_global_cat_20_names, mean_loss_global_cat_50_names, mean_max_loss_by_component_cat_5_names, \
           mean_max_loss_by_component_cat_10_names, mean_max_loss_by_component_cat_20_names, \
           mean_max_loss_by_component_cat_50_names, max_max_loss_by_component_cat_5, max_max_loss_by_component_cat_10, \
           max_max_loss_by_component_cat_20, max_max_loss_by_component_cat_50, max_max_loss_by_component_cat_5_names, \
           max_max_loss_by_component_cat_10_names, max_max_loss_by_component_cat_20_names, \
           max_max_loss_by_component_cat_50_names


def plot_histograms_models(criterion_corr_filter_20, criterion_corr_filter_50, criterion_corr_class, criterion_loss_max,
                           criterion_max_of_max_loss, criterion_ratio_variance, plot=False):
    # load score correlation filters:
    score_correlation_filters_cat_5, score_correlation_filters_cat_10, score_correlation_filters_cat_20, \
    score_correlation_filters_cat_50, score_correlation_filters_cat_5_names, \
    score_correlation_filters_cat_10_names, score_correlation_filters_cat_20_names, \
    score_correlation_filters_cat_50_names = load_correlation_score_filters()

    # load score correlation classes:
    score_correlation_class_cat_5, score_correlation_class_cat_10, score_correlation_class_cat_20, \
    score_correlation_class_cat_50, score_correlation_class_cat_5_names, score_correlation_class_cat_10_names, \
    score_correlation_class_cat_20_names, score_correlation_class_cat_50_names = load_correlation_score_class()

    # load ratio mean std:
    ratio_mean_global_cat_5, ratio_mean_global_cat_10, ratio_mean_global_cat_20, ratio_mean_global_cat_50, \
    ratio_mean_global_cat_5_names, ratio_mean_global_cat_10_names, ratio_mean_global_cat_20_names, \
    ratio_mean_global_cat_50_names = load_ratio_scores()

    # load mean loss and max mean loss:
    mean_loss_global_cat_5, mean_loss_global_cat_10, mean_loss_global_cat_20, mean_loss_global_cat_50, \
    mean_max_loss_by_component_cat_5, mean_max_loss_by_component_cat_10, mean_max_loss_by_component_cat_20, \
    mean_max_loss_by_component_cat_50, mean_loss_global_cat_5_names, mean_loss_global_cat_10_names, \
    mean_loss_global_cat_20_names, mean_loss_global_cat_50_names, mean_max_loss_by_component_cat_5_names, \
    mean_max_loss_by_component_cat_10_names, mean_max_loss_by_component_cat_20_names, \
    mean_max_loss_by_component_cat_50_names, max_max_loss_by_component_cat_5, max_max_loss_by_component_cat_10, \
    max_max_loss_by_component_cat_20, max_max_loss_by_component_cat_50, max_max_loss_by_component_cat_5_names, \
    max_max_loss_by_component_cat_10_names, max_max_loss_by_component_cat_20_names, \
    max_max_loss_by_component_cat_50_names = load_loss_mean()

    # load ratio variance:
    ratio_variance_cat_5, ratio_variance_cat_10, ratio_variance_cat_20, ratio_variance_cat_50, \
    ratio_variance_cat_5_names, ratio_variance_cat_10_names, ratio_variance_cat_20_names, \
    ratio_variance_cat_50_names = load_ratio_variance()

    if plot:
        # plot histogram correlation score filters:
        n_bins = 50
        fig, ax = plt.subplots(1, 2, figsize=(20, 10), facecolor='w', edgecolor='k')
        fig.suptitle('Histo: average correlation score for filters', fontsize=16)
        # We can set the number of bins with the `bins` kwarg
        ax[0].hist(score_correlation_filters_cat_20, bins=n_bins, label='len z_struct: 20')
        ax[1].hist(score_correlation_filters_cat_50, bins=n_bins, label='len z_struct: 50')
        ax[0].legend(loc=1)
        ax[1].legend(loc=1)
        plt.show()

        # plot histogram correlation score class:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10), facecolor='w', edgecolor='k')
        fig.suptitle('Histo: average correlation score for class', fontsize=16)
        # We can set the number of bins with the `bins` kwarg
        ax[0].hist(score_correlation_class_cat_20, bins=n_bins, label='len z_struct: 20')
        ax[1].hist(score_correlation_class_cat_50, bins=n_bins, label='len z_struct: 50')
        ax[0].legend(loc=1)
        ax[1].legend(loc=1)
        plt.show()

        # plot histogram ratio mean std:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10), facecolor='w', edgecolor='k')
        fig.suptitle('Histo: ratio mean std', fontsize=16)
        # We can set the number of bins with the `bins` kwarg
        ax[0].hist(ratio_mean_global_cat_20, bins=n_bins, label='len z_struct: 20')
        ax[1].hist(ratio_mean_global_cat_50, bins=n_bins, label='len z_struct: 50')
        ax[0].legend(loc=1)
        ax[1].legend(loc=1)
        plt.show()

        # plot histogram mean loss and max mean loss:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10), facecolor='w', edgecolor='k')
        fig.suptitle('Histo: mean loss global', fontsize=16)
        # We can set the number of bins with the `bins` kwarg
        ax[0].hist(mean_loss_global_cat_20, bins=n_bins, label='len z_struct: 20')
        ax[1].hist(mean_loss_global_cat_50, bins=n_bins, label='len z_struct: 50')
        ax[0].legend(loc=1)
        ax[1].legend(loc=1)
        plt.show()

        # plot histogram mean loss and max mean loss:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10), facecolor='w', edgecolor='k')
        fig.suptitle('Histo: mean loss max global', fontsize=16)
        # We can set the number of bins with the `bins` kwarg
        ax[0].hist(mean_max_loss_by_component_cat_20, bins=n_bins, label='len z_struct: 20')
        ax[1].hist(mean_max_loss_by_component_cat_50, bins=n_bins, label='len z_struct: 50')
        ax[0].legend(loc=1)
        ax[1].legend(loc=1)
        plt.show()

        # plot histogram mean loss and max mean loss:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10), facecolor='w', edgecolor='k')
        fig.suptitle('Histo: max loss max global', fontsize=16)
        # We can set the number of bins with the `bins` kwarg
        ax[0].hist(max_max_loss_by_component_cat_20, bins=n_bins, label='len z_struct: 20')
        ax[1].hist(max_max_loss_by_component_cat_50, bins=n_bins, label='len z_struct: 50')
        ax[0].legend(loc=1)
        ax[1].legend(loc=1)
        plt.show()

        # plot histogram ratio variance:
        fig, ax = plt.subplots(1, 2, figsize=(20, 10), facecolor='w', edgecolor='k')
        fig.suptitle('Histo: ratio variance', fontsize=16)
        # We can set the number of bins with the `bins` kwarg
        ax[0].hist(ratio_variance_cat_20, bins=n_bins, label='len z_struct: 20')
        ax[1].hist(ratio_variance_cat_50, bins=n_bins, label='len z_struct: 50')
        ax[0].legend(loc=1)
        ax[1].legend(loc=1)
        plt.show()

    # get model name who respect criterion:
    model_selected_corr_filter_20 = score_correlation_filters_cat_20_names[
        np.argwhere(np.abs(score_correlation_filters_cat_20) < criterion_corr_filter_20)]
    model_selected_corr_filter_50 = score_correlation_filters_cat_50_names[
        np.argwhere(np.abs(score_correlation_filters_cat_50) < criterion_corr_filter_50)]

    np.save('list_model_selected/criterion_corr_filter_list_model_z_struct_20.npy', model_selected_corr_filter_20)
    np.save('list_model_selected/criterion_corr_filter_list_model_z_struct_50.npy', model_selected_corr_filter_50)

    # get model name who respect criterion:
    model_selected_corr_class_20 = score_correlation_class_cat_20_names[
        np.argwhere(np.abs(score_correlation_class_cat_20) < criterion_corr_class)]
    model_selected_corr_class_50 = score_correlation_class_cat_50_names[
        np.argwhere(np.abs(score_correlation_class_cat_50) < criterion_corr_class)]

    np.save('list_model_selected/criterion_corr_class_list_model_z_struct_20.npy', model_selected_corr_class_20)
    np.save('list_model_selected/criterion_corr_class_list_model_z_struct_50.npy', model_selected_corr_class_50)

    # get model name who respect criterion:
    model_selected_loss_max_20 = mean_max_loss_by_component_cat_20_names[
        np.argwhere(np.abs(mean_max_loss_by_component_cat_20) < criterion_loss_max)]
    model_selected_loss_max_50 = mean_max_loss_by_component_cat_50_names[
        np.argwhere(np.abs(mean_max_loss_by_component_cat_50) < criterion_loss_max)]

    np.save('list_model_selected/criterion_loss_max_list_model_z_struct_20.npy', model_selected_loss_max_20)
    np.save('list_model_selected/criterion_loss_max_list_model_z_struct_50.npy', model_selected_loss_max_50)

    # get model name who respect criterion:
    model_selected_loss_max_of_max_20 = max_max_loss_by_component_cat_20_names[
        np.argwhere(np.abs(max_max_loss_by_component_cat_20) < criterion_max_of_max_loss)]
    model_selected_loss_max_of_max_50 = max_max_loss_by_component_cat_50_names[
        np.argwhere(np.abs(max_max_loss_by_component_cat_50) < criterion_max_of_max_loss)]

    np.save('list_model_selected/criterion_loss_max_of_max_list_model_z_struct_20.npy',
            model_selected_loss_max_of_max_20)
    np.save('list_model_selected/criterion_loss_max_of_max_list_model_z_struct_50.npy',
            model_selected_loss_max_of_max_50)

    # get model name who respect criterion ratio variance:
    model_selected_ratio_variance_20 = ratio_variance_cat_20_names[
        np.argwhere(np.abs(ratio_variance_cat_20) > criterion_ratio_variance)]
    model_selected_ratio_variance_50 = ratio_variance_cat_50_names[
        np.argwhere(np.abs(ratio_variance_cat_50) > criterion_ratio_variance)]

    np.save('list_model_selected/criterion_ratio_variance_list_model_z_struct_20.npy',
            model_selected_ratio_variance_20)
    np.save('list_model_selected/criterion_ratio_variance_list_model_z_struct_50.npy',
            model_selected_ratio_variance_50)

    return


def distance_matrix(net, exp_name, train_test=None, plot_fig=False):
    """
    Distance matrix between all classes (nb_class * nb_class-1) distances.
    :return:
    """
    # define all distances between mean classes:
    average_representation_z_struct_class, _ = load_average_z_struct_representation(exp_name, train_test=train_test)
    nb_class = len(average_representation_z_struct_class)

    distance_inter_class = np.ones((nb_class, nb_class))
    for i in range(nb_class):
        dist_class = []
        for j in range(nb_class):
            dist = np.linalg.norm(average_representation_z_struct_class[i] - average_representation_z_struct_class[j])
            dist_class.append(dist)
        distance_inter_class[i] = dist_class

    if plot_fig:
        fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
        ax.set(title=('Model: {}, distances matrix:'.format(exp_name)))

        img = ax.matshow(distance_inter_class, cmap=plt.cm.rainbow)
        plt.colorbar(img, ticks=[-1, 0, 1], fraction=0.045)
        for x in range(distance_inter_class.shape[0]):
            for y in range(distance_inter_class.shape[1]):
                plt.text(x, y, "%0.2f" % distance_inter_class[x, y], size=12, color='black', ha="center", va="center")
        plt.show()

    return distance_inter_class


def plot_resume(net, exp_name, is_ratio, is_distance_loss, loss_distance_mean, loader, loader_train, device, cat=None,
                train_test=None, path_scores=None, save=True, diff_var=False, contrastive_loss=False,
                encoder_struct=False, Hmg_dst=False, z_struct_size=None):
    """
    plot interesting values to resume experiementation behavior: distance matrix, loss, acc, ratio, var intra, var inter
    :param net:
    :param exp_name:
    :param train_test:
    :return:
    """

    ratio_variance_mean, variance_intra_class, variance_inter_class = ratio(exp_name,
                                                                            train_test=train_test,
                                                                            cat=cat,
                                                                            normalized=True)

    variance_intra_class_mean_per_class = np.mean(variance_intra_class, axis=1)
    variance_intra_class_mean = np.mean(variance_intra_class_mean_per_class)
    variance_inter_class_mean = np.mean(variance_inter_class)

    nb_class = len(variance_intra_class)

    # define figure:____________________________________________________________________________________________________
    fig, axs = plt.subplots(nrows=2, ncols=2, gridspec_kw={'width_ratios': [2, 1], 'height_ratios': [1, 1]},
                            figsize=(30, 20), facecolor='w', edgecolor='k')
    fig.suptitle('Resume results for model: {}'.format(exp_name), fontsize=16)

    # loss and acc:____________________________________________________________________________________________________
    exp_name_chkpts = exp_name.split('_normalized_l1_')[0]

    global_iter, epochs, train_score, test_score, total_train, total_test, ratio_train_loss, ratio_test_loss, \
    var_distance_classes_train, var_distance_classes_test, mean_distance_intra_class_train, \
    mean_distance_intra_class_test, intra_var_train, intra_var_test, inter_var_train, inter_var_test, \
    diff_var_train, diff_var_test, contrastive_train, contrastive_test, classification_test, \
    classification_train = get_checkpoints_scores_CNN(path_scores,
                                                      exp_name,
                                                      is_ratio=is_ratio,
                                                      is_distance_loss=is_distance_loss,
                                                      loss_distance_mean=loss_distance_mean)
    # get accuracy train and test:
    # acc_train_last_epoch = train_score[-1]
    # acc_test_last_epoch = test_score[-1]
    print('Computing scores...')
    score_test, _, _, _, _, _, _, _, _, _, _ = compute_scores(net, loader, device, len(loader.dataset), False, False,
                                                              False, False, False, False, False, False, False, False,
                                                              False, False, False, False, False, False, False, False)

    # score_train, _, _, _, _, _, _, _, _, _, _ = compute_scores(net, loader_train, device, len(loader_train.dataset),
    #                                                            False, False, False, False, False, False, False, False,
    #                                                            False, False, False, False, False, False, False, False,
    #                                                            False, False)

    # losses:
    axs[0, 0].set(xlabel='nb_iter', ylabel='loss', title=('Losses: ' + exp_name))
    axs[0, 0].plot(epochs, total_train, label='Total loss train')
    axs[0, 0].plot(epochs, total_test, label='Total loss test')

    if is_ratio:
        axs[0, 0].plot(epochs, ratio_train_loss, label='ratio loss train')
        axs[0, 0].plot(epochs, ratio_test_loss, label='ratio loss test')
    if is_distance_loss:
        axs[0, 0].plot(epochs, var_distance_classes_train, label='std distance between class train')
        axs[0, 0].plot(epochs, var_distance_classes_test, label='std distance between class train')
    if loss_distance_mean:
        axs[0, 0].plot(epochs, mean_distance_intra_class_train, label='mean distance between class train')
        axs[0, 0].plot(epochs, mean_distance_intra_class_test, label='mean distance between class train')
    if diff_var:
        axs[0, 0].plot(epochs, diff_var_train, label='diff_var_train')
        axs[0, 0].plot(epochs, diff_var_test, label='diff_var_test')
        axs[0, 0].plot(epochs, intra_var_train, label='variance intra classes train')
        axs[0, 0].plot(epochs, intra_var_test, label='variance intra classes test')
        axs[0, 0].plot(epochs, inter_var_train, label='variance inter classes train')
        axs[0, 0].plot(epochs, inter_var_test, label='variance inter classes test')
    if contrastive_loss:
        axs[0, 0].plot(epochs, contrastive_train, label='contrastive loss train')
        axs[0, 0].plot(epochs, contrastive_test, label='contrastive loss test')

    axs[0, 0].legend(loc=1)

    """
    # 2d projection:____________________________________________________________________________________________________
    z_struct_representation, label_list, _ = load_z_struct_representation(exp_name, train_test=train_test)

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(z_struct_representation)
    t = reduced.transpose()

    lim_min = np.min(t)
    lim_max = np.max(t)

    axs[1, 1].set(xlabel='nb_iter', ylabel='loss', title='PCA of z_struct for each images')
    axs[1, 1].set(xlim=(lim_min, lim_max), ylim=(lim_min, lim_max))

    x = np.arange(nb_class)
    ys = [i + x + (i * x) ** 2 for i in range(10)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))

    for c, color in zip(range(nb_class), colors):
        x = np.array(t[0])[np.concatenate(np.argwhere(np.array(label_list) == c)).ravel()]
        y = np.array(t[1])[np.concatenate(np.argwhere(np.array(label_list) == c)).ravel()]
        axs[1, 1].scatter(x, y, alpha=0.6, color=color, label='class ' + str(c))
    axs[1, 1].legend(loc=1)
    """

    # distance matrix: _________________________________________________________________________________________________
    distance_inter_class = distance_matrix(net, exp_name, train_test=train_test, plot_fig=False)
    distance_classes_triu = np.triu(distance_inter_class, k=1)
    distance_classes_triu_wt_diag = distance_classes_triu[np.nonzero(distance_classes_triu)]
    distances_mean = np.sum(distance_classes_triu_wt_diag ** 2) / len(distance_classes_triu_wt_diag)
    std_distances = distances_mean / np.square(np.mean(distance_classes_triu_wt_diag))

    axs[0, 1].set(title='Distances matrix between mean classes')
    img = axs[0, 1].matshow(distance_inter_class, cmap=plt.cm.rainbow)
    fig.colorbar(img, ticks=[-1, 0, 1], fraction=0.045, ax=axs[0, 1])
    for x in range(distance_inter_class.shape[0]):
        for y in range(distance_inter_class.shape[1]):
            axs[0, 1].text(x, y, "%0.2f" % distance_inter_class[x, y], size=12, color='black', ha="center", va="center")

    # values in array: _________________________________________________________________________________________________
    # Binary encoder struct value:
    if encoder_struct:
        uniq_code, Hmg_dst, percentage_uniq_code = same_binary_code(net,
                                                                    exp_name,
                                                                    loader,
                                                                    nb_class,
                                                                    train_test=train_test,
                                                                    save=False,
                                                                    Hmg_dist=Hmg_dst)
        Hmg_dst = np.round(Hmg_dst, 2)
        avg_hmg_dst = round(np.mean(Hmg_dst), 2)
        percentage_uniq_code = np.round(percentage_uniq_code*100., 2)
        avg_perc_uniq_code = round(np.mean(percentage_uniq_code), 2)

        sc = score_with_best_code_uniq(net, exp_name, train_test, loader, z_struct_size, len(loader.dataset))
        if sc == 100.:
            perfect_score = 'True'
        else:
            perfect_score = 'False'

        percent_max = histo_count_uniq_code(exp_name, train_test, plot_histo=False, return_percent=True)

    axs[1, 0].set(title='Statistics')

    text_titles = ["Ratio: ",
                   "Variance intra class mean: ",
                   "Variance inter class mean: ",
                   "Std distances between classes: ",
                   "Accuracy test (%): "]
    if encoder_struct:
        text_titles += ['Avg Hmg dst each class (%): ',
                        'Global avg Hamming distance (%): ',
                        'Avg uc per class (%): ',
                        'Global avg uniq code (%): ',
                        'Maj uc proportion per class (%): ',
                        'Average Maj uc proportion per class (%): ',
                        'If replace each class by uniq maj binary code, perfect score: ']
    n_lines = len(text_titles)
    text_values = [ratio_variance_mean,
                   variance_intra_class_mean,
                   variance_inter_class_mean,
                   std_distances,
                   score_test]

    if encoder_struct:
        text_values += [Hmg_dst,
                        avg_hmg_dst,
                        percentage_uniq_code,
                        avg_perc_uniq_code,
                        percent_max,
                        round(np.mean(percent_max), 2),
                        perfect_score]

    for y in range(n_lines):
        axs[1, 0].text(.1,
                       1 - (y + 0.4) / n_lines,
                       text_titles[y] + str(text_values[y]),
                       size=20,
                       color='black',
                       va="center")

    fig.tight_layout()
    # plt.show()

    path_save = 'fig_results/resume/plot_resume_' + exp_name + '_' + train_test + '.png'
    if save:
        fig.savefig(path_save)
        print("figure saved for model {}".format(exp_name))

    return


def plot_VAE_resume(net, model_name, z_struct_size, z_var_size, loader, VAE_struct, is_vae_var, train_test, save=True,
                    nb_class=10, nb_img=8, std_var=None, mu_var=None, mu_struct=None, index=0):
    """
    plot interesting values to resume experiementation behavior for VAE.
    :param net:
    :param exp_name:
    :param train_test:
    :return:
    """

    # define figure:____________________________________________________________________________________________________
    fig, axs = plt.subplots(nrows=4, ncols=3, gridspec_kw={'width_ratios': [1, 1, 1],
                                                           'height_ratios': [1, 2, 1.5, 1.5]},
                            figsize=(30, 40), facecolor='w', edgecolor='k')
    fig.suptitle('Resume VAE for model: {}'.format(model_name), fontsize=16)
    plt.axis('off')

    # Real distribution:________________________________________________________________________________________________
    mu_var, sigma_var, encoder_struct_zeros_proportion = real_distribution_model(net,
                                                                                 model_name,
                                                                                 z_struct_size,
                                                                                 z_var_size,
                                                                                 loader,
                                                                                 'test',
                                                                                 plot_gaussian=False,
                                                                                 save=True,
                                                                                 VAE_struct=VAE_struct,
                                                                                 is_vae_var=is_vae_var)
    if VAE_struct:
        axs[0, 1].set(title=('Proportion of zeros: ' + model_name + "_" + train_test))
        axs[0, 1].bar(np.arange(len(encoder_struct_zeros_proportion)), encoder_struct_zeros_proportion,
                label='Proportion of zeros for each encoder struct component',
                color='blue')

    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

    # plot figure:
    axs[0, 0].set(title=('Gaussian: ' + model_name + "_" + train_test))
    axs[0, 0].plot(x, stats.norm.pdf(x, mu, sigma), label='Gaussian (0, I)', color='red')

    for i in range(len(mu_var)):
        mu_var_iter = mu_var[i]
        variance_var_iter = np.abs(sigma_var[i])
        sigma_var_iter = math.sqrt(variance_var_iter)
        x_var = np.linspace(mu_var_iter - 3 * sigma_var_iter, mu_var_iter + 3 * sigma_var_iter, 100)
        axs[0, 0].plot(x_var, stats.norm.pdf(x_var, mu_var_iter, sigma_var_iter), label='real data gaussian ' + str(i),
                  color='blue')

    axs[0, 0].legend(loc=1)

    # Reconstructions:________________________________________________________________________________________________
    size = (nb_img, nb_class * 2)

    # get n data per classes:
    first = True
    for data, label in loader:
        # print('loader ', len(label))
        for lab in range(nb_class):
            batch_lab = data[torch.where(label == lab)[0][:nb_img]]
            if first:
                batch = batch_lab
                first = False
            else:
                batch = torch.cat((batch, batch_lab), dim=0)

    # get reconstruction
    with torch.no_grad():
        input_data = batch
    if torch.cuda.is_available():
        input_data = input_data.cuda()

    # z reconstruction:
    if is_vae_var:
        x_recon, _ = net(input_data)
    else:
        x_recon, z_struct, z_var, z_var_sample, _, z, _ = net(input_data)

    # compute score reconstruction on dataset test:
    score_reconstruction = F.mse_loss(x_recon, input_data)
    
    if torch.cuda.is_available():
        score_reconstruction = np.around(score_reconstruction.cpu().detach().numpy(), 3)
    else:
        score_reconstruction = np.around(score_reconstruction.detach().numpy(), 3)

    first = True
    for class_id in range(nb_class):
        start_cl_id = class_id * nb_img
        end_cl_id = start_cl_id + nb_img
        original_batch = input_data[start_cl_id:end_cl_id]
        reconstruction_batch = x_recon[start_cl_id:end_cl_id]
        comparaison_batch = torch.cat([original_batch, reconstruction_batch])
        if first:
            comparison = comparaison_batch
            first = False
        else:
            comparison = torch.cat((comparison, comparaison_batch), dim=0)
    reconstructions = make_grid(comparison.data, nrow=size[0])
    recon_grid = reconstructions.permute(1, 2, 0)

    # z_struct reconstruction:
    random_var = std_var * torch.randn(x_recon.shape[0], z_var_size) + mu_var
    if torch.cuda.is_available():
        z_struct_random = torch.cat((random_var, z_struct.cpu()), dim=1)
    else:
        z_struct_random = torch.cat((random_var, z_struct), dim=1)
    x_recon_struct = net.decoder(z_struct_random)
    score_reconstruction_zstruct = F.mse_loss(x_recon_struct, input_data).cpu().item()
    if torch.cuda.is_available():
        score_reconstruction_zstruct = np.around(score_reconstruction_zstruct, 3)
    else:
        score_reconstruction_zstruct = np.around(score_reconstruction_zstruct.detach().numpy(), 3)

    first = True
    for class_id in range(nb_class):
        start_cl_id = class_id * nb_img
        end_cl_id = start_cl_id + nb_img
        original_batch = input_data[start_cl_id:end_cl_id]
        reconstruction_batch = x_recon_struct[start_cl_id:end_cl_id]
        comparaison_batch = torch.cat([original_batch, reconstruction_batch])
        if first:
            comparison = comparaison_batch
            first = False
        else:
            comparison = torch.cat((comparison, comparaison_batch), dim=0)
    reconstructions_struct = make_grid(comparison.data, nrow=size[0])
    recon_grid_struct = reconstructions_struct.permute(1, 2, 0)

    # z_var reconstruction:
    mu_struct = torch.tensor(1 - (mu_struct / 100))
    proba_struct = mu_struct.repeat(x_recon.shape[0], 1)
    random_struct = torch.Tensor.float(torch.bernoulli(proba_struct))
    z_var_random = torch.cat((z_var_sample, random_struct), dim=1)
    x_recon_var = net.decoder(z_var_random)
    score_reconstruction_zvar = F.mse_loss(x_recon_var, input_data)
    score_reconstruction_zvar = np.around(score_reconstruction_zvar.detach().numpy(), 3)

    first = True
    for class_id in range(nb_class):
        start_cl_id = class_id * nb_img
        end_cl_id = start_cl_id + nb_img
        original_batch = input_data[start_cl_id:end_cl_id]
        reconstruction_batch = x_recon_var[start_cl_id:end_cl_id]
        comparaison_batch = torch.cat([original_batch, reconstruction_batch])
        if first:
            comparison = comparaison_batch
            first = False
        else:
            comparison = torch.cat((comparison, comparaison_batch), dim=0)
    reconstructions_var = make_grid(comparison.data, nrow=size[0])
    recon_grid_var = reconstructions_var.permute(1, 2, 0)

    # plot:
    axs[1, 0].set(title=('Z reconstruction: BCE score: {}'.format(score_reconstruction)))
    axs[1, 0].imshow(recon_grid.numpy())

    axs[1, 1].set(title=('Z_struct reconstruction: BCE score: {}'.format(score_reconstruction_zstruct)))
    axs[1, 1].imshow(recon_grid_struct.numpy())

    axs[1, 2].set(title=('Z_var reconstruction: BCE score: {}'.format(score_reconstruction_zvar)))
    axs[1, 2].imshow(recon_grid_var.numpy())

    # Traversal:________________________________________________________________________________________________
    # load both prototype:
    traversal_size = 8
    maj_uc_base = torch.tensor(np.load('binary_encoder_struct_results/uniq_code/uc_maj_class_' + model_name + '_' \
                                       + train_test + '.npy', allow_pickle=True)).unsqueeze(dim=0)
    maj_uc = torch.repeat_interleave(maj_uc_base, traversal_size, dim=0)
    z_var_zeros = torch.zeros((traversal_size, nb_class, z_var_size))
    maj_uc_proto = torch.cat((z_var_zeros, maj_uc), dim=2)  # shape: size, nb_class, embedding_size

    base_prototype_maj_uc = torch.cat((torch.zeros(1, nb_class, z_var_size), maj_uc_base), dim=2)

    if index is not None:
        # Sweep over linearly spaced coordinates transformed through the
        # inverse CDF (ppf) of a gaussian since the prior of the latent
        # space is gaussian
        # cdf_traversal = np.linspace(0.05, 0.95, traversal_size)
        # cont_traversal = stats.norm.ppf(cdf_traversal)
        max_value = np.abs(sigma_var.mean())
        cont_traversal = np.linspace(-max_value, max_value, traversal_size)
        for class_id in range(nb_class):
            for column in range(traversal_size):
                maj_uc_proto[column][class_id][index] = cont_traversal[column]

    maj_uc_proto = torch.cat((base_prototype_maj_uc, maj_uc_proto), dim=0)
    maj_uc_proto = maj_uc_proto.permute(1, 0, 2)
    latent_samples = maj_uc_proto
    # Map samples through decoder
    generated_maj_uc = net.decoder(latent_samples)
    traversals_maj_uc = make_grid(generated_maj_uc.data, nrow=traversal_size + 1)
    traversals_maj_uc = traversals_maj_uc.permute(1, 2, 0)

    _, avg_z_struct_base = get_z_struct_per_class_VAE(model_name, train_test='test', nb_class=nb_class)
    avg_z_struct_base = torch.tensor(avg_z_struct_base).unsqueeze(dim=0)
    avg_z_struct = torch.repeat_interleave(avg_z_struct_base, traversal_size, dim=0)
    z_var_zeros = torch.zeros((traversal_size, nb_class, z_var_size))
    avg_struct_proto = torch.cat((z_var_zeros, avg_z_struct), dim=2)  # shape: traversal_size, nb_class, embedding_size

    base_prototype_avg_struct = torch.cat((torch.zeros(1, nb_class, z_var_size), avg_z_struct_base), dim=2)

    if index is not None:
        # Sweep over linearly spaced coordinates transformed through the
        # inverse CDF (ppf) of a gaussian since the prior of the latent
        # space is gaussian
        # cdf_traversal = np.linspace(0.05, 0.95, traversal_size)
        # cont_traversal = stats.norm.ppf(cdf_traversal)
        max_value = np.abs(sigma_var.mean())
        cont_traversal = np.linspace(-max_value, max_value, traversal_size)
        for class_id in range(nb_class):
            for column in range(traversal_size):
                avg_struct_proto[column][class_id][index] = cont_traversal[column]

    avg_struct_proto = torch.cat((base_prototype_avg_struct, avg_struct_proto), dim=0)
    avg_struct_proto = avg_struct_proto.permute(1, 0, 2)
    latent_samples = avg_struct_proto
    # Map samples through decoder
    generated_avg_struct = net.decoder(latent_samples)
    traversals_avg_struct = make_grid(generated_avg_struct.data, nrow=traversal_size + 1)
    traversals_avg_struct = traversals_avg_struct.permute(1, 2, 0)

    # figure:
    axs[3, 0].set(title=('Latent traversal with z_struct maj uc prototype: {}'.format(model_name)))
    axs[3, 0].imshow(traversals_maj_uc.numpy())
    axs[3, 0].axvline(32, linewidth=5, color='orange')

    axs[3, 1].set(title=('Latent traversal with avg z_struct prototype: {}'.format(model_name)))
    axs[3, 1].imshow(traversals_avg_struct.numpy())
    axs[3, 1].axvline(32, linewidth=5, color='orange')

    # Generation random:________________________________________________________________________________________________
    # generate random sample from real distribution:
    generation_size = (8, 8)
    nb_samples = generation_size[0] * generation_size[1]

    # maj uc:
    maj_uc = torch.tensor(np.load('binary_encoder_struct_results/uniq_code/uc_maj_class_' + model_name + '_' \
                                  + train_test + '.npy', allow_pickle=True))

    # z_var random sample:
    sample_var = std_var * torch.randn(nb_samples, z_var_size) + mu_var

    # z_struct random maj uc:
    sample_struct_maj_uc = torch.zeros(nb_samples, z_struct_size)
    for i in range(len(sample_struct_maj_uc)):
        sample_struct_maj_uc[i] = maj_uc[np.random.randint(len(maj_uc))]
    # z random:
    sample_maj_uc = torch.cat((sample_var, sample_struct_maj_uc), dim=1)

    # z_struct random:
    mu_struct = torch.tensor(1 - (mu_struct / 100))
    proba_struct = mu_struct.repeat(nb_samples, 1)
    sample_struct_avg_struct = torch.Tensor.float(torch.bernoulli(proba_struct))
    # z random:
    sample_avg_struct = torch.cat((sample_var, sample_struct_avg_struct), dim=1)

    # generate:
    generated_maj_uc = net.decoder(sample_maj_uc)
    grid_generation_maj_uc = make_grid(generated_maj_uc.data, nrow=generation_size[1])

    generated_avg_struct = net.decoder(sample_avg_struct)
    grid_generation_avg_struct = make_grid(generated_avg_struct.data, nrow=generation_size[1])

    # compute scores:
    FID_score_maj_uc = 0
    FID_score_avg_struct = 0
    IS_score_maj_uc = 0
    IS_score_avg_struct = 0
    LPIPS_score_alex_maj_uc = 0
    LPIPS_score_vgg_maj_uc = 0
    LPIPS_score_alex_avg_struct = 0
    LPIPS_score_vgg_avg_struct = 0

    generation_batch_maj_uc = generated_maj_uc[:nb_samples]
    generation_batch_avg_struct = generated_avg_struct[:nb_samples]

    """
    # FID scores:
    FID_score_maj_uc = calculate_fid_given_paths(original_batch,
                                                 generation_batch_maj_uc,
                                                 batch_size=32,
                                                 cuda='',
                                                 dims=2048)
    FID_score_avg_struct = calculate_fid_given_paths(original_batch,
                                                     generation_batch_avg_struct,
                                                     batch_size=32,
                                                     cuda='',
                                                     dims=2048)
    FID_score_maj_uc = np.around(FID_score_maj_uc, 3)
    FID_score_avg_struct = np.around(FID_score_avg_struct, 3)

    # IS scores:
    IS_score_maj_uc = inception_score(generation_batch_maj_uc,
                                      batch_size=32,
                                      resize=True)
    IS_score_avg_struct = inception_score(generation_batch_avg_struct,
                                          batch_size=32,
                                          resize=True)
    IS_score_maj_uc = np.around(IS_score_maj_uc[0], 3)
    IS_score_avg_struct = np.around(IS_score_avg_struct[0], 3)

    # LPIPS scores:
    loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
    loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization

    # image should be RGB, IMPORTANT: normalized to [-1,1]
    LPIPS_score_alex_maj_uc = torch.mean(loss_fn_alex(original_batch, generation_batch_maj_uc)).item()
    LPIPS_score_vgg_maj_uc = torch.mean(loss_fn_vgg(original_batch, generation_batch_maj_uc)).item()
    LPIPS_score_alex_maj_uc = np.around(LPIPS_score_alex_maj_uc, 3)
    LPIPS_score_vgg_maj_uc = np.around(LPIPS_score_vgg_maj_uc, 3)

    LPIPS_score_alex_avg_struct = torch.mean(loss_fn_alex(original_batch, generation_batch_avg_struct)).item()
    LPIPS_score_vgg_avg_struct = torch.mean(loss_fn_vgg(original_batch, generation_batch_avg_struct)).item()
    LPIPS_score_alex_avg_struct = np.around(LPIPS_score_alex_avg_struct, 3)
    LPIPS_score_vgg_avg_struct = np.around(LPIPS_score_vgg_avg_struct, 3)
    """

    # plot:
    samples_maj_uc = grid_generation_maj_uc.permute(1, 2, 0)
    samples_avg_struct = grid_generation_avg_struct.permute(1, 2, 0)

    axs[2, 0].set(title=('Maj uc random. Scores: FID(\u2193): {}, IS(\u2191): {},'
                  ' LPIPS (alex) (\u2191): {}, LPIPS (vgg) (\u2191): {}'.format(FID_score_maj_uc,
                                                                                IS_score_maj_uc,
                                                                                LPIPS_score_alex_maj_uc,
                                                                                LPIPS_score_vgg_maj_uc)))
    axs[2, 1].set(title=('Avg struct random. Scores: FID(\u2193): {}, IS(\u2191): {},'
                         ' LPIPS (alex) (\u2191): {}, LPIPS (vgg) (\u2191): {}'.format(FID_score_avg_struct,
                                                                                       IS_score_avg_struct,
                                                                                       LPIPS_score_alex_avg_struct,
                                                                                       LPIPS_score_vgg_avg_struct)))
    axs[2, 0].imshow(samples_maj_uc.numpy())
    axs[2, 1].imshow(samples_avg_struct.numpy())
    # End plot:________________________________________________________________________________________________

    plt.show()

    path_save = 'fig_results/resume/plot_resume_VAE_' + model_name + '_' + train_test + '.png'
    if save:
        fig.savefig(path_save)

    return


def viz_decoder_multi_label(net, loader, exp_name, nb_img=8, nb_class=10, save=True):
    """
    plot multi data for the same label for each line.
    nb_class row and for each label they are one row with original data and one with reconstructed data.
    :param loader:
    :param net:
    :return:
    """

    net.eval()
    size = (nb_img, nb_class * 2)

    # get n data per classes:
    first = True
    for data, label in loader:
        # print('loader ', len(label))
        for lab in range(nb_class):
            batch_lab = data[torch.where(label == lab)[0][:nb_img]]
            if first:
                batch = batch_lab
                first = False
            else:
                batch = torch.cat((batch, batch_lab), dim=0)

    # get reconstruction
    with torch.no_grad():
        input_data = batch
    if torch.cuda.is_available():
        input_data = input_data.cuda()

    x_recon, _, _ = net(input_data)

    net.train()

    first = True
    for class_id in range(nb_class):
        start_cl_id = class_id * nb_img
        end_cl_id = start_cl_id + nb_img
        original_batch = input_data[start_cl_id:end_cl_id].cpu()
        reconstruction_batch = x_recon[start_cl_id:end_cl_id].cpu()
        comparaison_batch = torch.cat([original_batch, reconstruction_batch])
        if first:
            comparison = comparaison_batch
            first = False
        else:
            comparison = torch.cat((comparison, comparaison_batch), dim=0)

    reconstructions = make_grid(comparison.data, nrow=size[0])

    # grid with originals data
    recon_grid = reconstructions.permute(1, 2, 0)
    fig, ax = plt.subplots(figsize=(10, 20), facecolor='w', edgecolor='k')
    ax.set(title=('model: {}:'.format(exp_name)))

    ax.imshow(recon_grid.numpy())
    # ax.axhline(y=size[0] // 2, linewidth=4, color='r')
    plt.show()

    if save:
        fig.savefig("fig_results/reconstructions/fig_reconstructions_multi_label_" + exp_name + ".png")

    return


def reconstruction_local(nb_class, nb_img, input_data, x_recon, exp_name, size, score, save=True):
    first = True
    for class_id in range(nb_class):
        start_cl_id = class_id * nb_img
        end_cl_id = start_cl_id + nb_img
        original_batch = input_data[start_cl_id:end_cl_id].cpu()
        reconstruction_batch = x_recon[start_cl_id:end_cl_id].cpu()
        comparaison_batch = torch.cat([original_batch, reconstruction_batch])
        if first:
            comparison = comparaison_batch
            first = False
        else:
            comparison = torch.cat((comparison, comparaison_batch), dim=0)

    reconstructions = make_grid(comparison.data, nrow=size[0])

    # grid with originals data
    recon_grid = reconstructions.permute(1, 2, 0)
    fig, ax = plt.subplots(figsize=(10, 20), facecolor='w', edgecolor='k')
    ax.set(title=('model: {}, BCE score: {}'.format(exp_name, score)))

    ax.imshow(recon_grid.numpy())
    plt.show()

    if save:
        fig.savefig("fig_results/reconstructions/fig_reconstructions_multi_label_" + exp_name + ".png")

    return


def viz_reconstruction_VAE(net, loader, exp_name, z_var_size, z_struct_size, nb_img=8, nb_class=10, save=True,
                           z_reconstruction=True, z_struct_reconstruction=False, z_var_reconstruction=False,
                           return_scores=False, real_distribution=True, mu_var=None, std_var=None, mu_struct=None,
                           std_struct=None, is_vae_var=False):
    """
    plot multi data for the same label for each line.
    nb_class row and for each label they are one row with original data and one with reconstructed data.
    :param loader:
    :param net:
    :return:
    """
    size = (nb_img, nb_class * 2)

    # get n data per classes:
    first = True
    for data, label in loader:
        # print('loader ', len(label))
        for lab in range(nb_class):
            batch_lab = data[torch.where(label == lab)[0][:nb_img]]
            if first:
                batch = batch_lab
                first = False
            else:
                batch = torch.cat((batch, batch_lab), dim=0)

    # get reconstruction
    with torch.no_grad():
        input_data = batch
    if torch.cuda.is_available():
        input_data = input_data.cuda()

    # z reconstruction:
    if is_vae_var:
        x_recon, _, _ = net(input_data)
    else:
        x_recon, z_struct, z_var, z_var_sample, _, z, _ = net(input_data)

    # compute score reconstruction on dataset test:
    score_reconstruction = F.mse_loss(x_recon, input_data)
    score_reconstruction = np.around(score_reconstruction.detach().numpy(), 3)

    score_reconstruction_zstruct = 0
    score_reconstruction_zvar = 0
    if z_struct_reconstruction:
        # z_struct reconstruction:
        if real_distribution:
            random_var = std_var * torch.randn(x_recon.shape[0], z_var_size) + mu_var
            # random_var = torch.zeros((x_recon.shape[0], z_var_size))
        else:
            random_var = torch.randn((x_recon.shape[0], z_var_size))
        z_struct_random = torch.cat((random_var, z_struct), dim=1)
        x_recon_struct = net.decoder(z_struct_random)
        score_reconstruction_zstruct = F.mse_loss(x_recon_struct, input_data)
        score_reconstruction_zstruct = np.around(score_reconstruction_zstruct.detach().numpy(), 3)

    if z_var_reconstruction:
        # z_var reconstruction:
        if real_distribution:
            mu_struct = torch.tensor(1-(mu_struct/100))
            proba_struct = mu_struct.repeat(x_recon.shape[0], 1)
            random_struct = torch.Tensor.float(torch.bernoulli(proba_struct))
            # random_struct = torch.zeros((x_recon.shape[0], z_var_size))
        else:
            random_struct = torch.randn((x_recon.shape[0], z_struct_size))
        z_var_random = torch.cat((z_var_sample, random_struct), dim=1)
        x_recon_var = net.decoder(z_var_random)
        score_reconstruction_zvar = F.mse_loss(x_recon_var, input_data)
        score_reconstruction_zvar = np.around(score_reconstruction_zvar.detach().numpy(), 3)

    """
    for i, (_data_, _label_) in enumerate(loader):
        print('computing scores.... {}/{}'.format(i, len(loader)))
        with torch.no_grad():
            _input_data_ = _data_
        if torch.cuda.is_available():
            _input_data_ = _input_data_.cuda()

        # z reconstruction:
        if is_vae_var:
            _x_recon_, _, _ = net(_input_data_)
        else:
            _x_recon_, _z_struct_, _, _z_var_sample_, _, _, _, _ = net(_input_data_)

        score_reconstruction_iter = F.mse_loss(_x_recon_, _input_data_)
        score_reconstruction += score_reconstruction_iter.detach().numpy()

        if z_var_reconstruction:
            # z_var + rand struct reconstruction:
            _random_struct_ = torch.abs(std_struct * torch.randn(_input_data_.shape[0], z_struct_size) + mu_struct)
            _z_var_random_ = torch.cat((_z_var_sample_, _random_struct_), dim=1)
            _x_recon_var_, _ = net.decoder(_z_var_random_)
            score_reconstruction_zvar_iter = F.mse_loss(_x_recon_var_, _input_data_)
            score_reconstruction_zvar += score_reconstruction_zvar_iter.detach().numpy()

        if z_struct_reconstruction:
            # z_struct + rand var reconstruction:
            _random_var_ = std_var * torch.randn(_input_data_.shape[0], z_var_size) + mu_var
            _z_struct_random_ = torch.cat((_random_var_, _z_struct_), dim=1)
            _x_recon_struct_, _ = net.decoder(_z_struct_random_)
            score_reconstruction_zstruct_iter = F.mse_loss(_x_recon_struct_, _input_data_)
            score_reconstruction_zstruct += score_reconstruction_zstruct_iter.detach().numpy()
    
    score_reconstruction /= len(loader)
    score_reconstruction_zvar /= len(loader)
    score_reconstruction_zstruct /= len(loader)
    """

    print('reconstructions scores on dataset test: z: {}, z_struct: {}, z_var: {}'.format(score_reconstruction,
                                                                                          score_reconstruction_zstruct,
                                                                                          score_reconstruction_zvar))

    if z_reconstruction:
        reconstruction_local(nb_class, nb_img, input_data, x_recon, exp_name, size, score_reconstruction, save=save)

    if z_struct_reconstruction:
        exp_name_struct = exp_name + "_z_struct"
        reconstruction_local(nb_class, nb_img, input_data, x_recon_struct, exp_name_struct, size,
                             score_reconstruction_zstruct, save=save)

    if z_var_reconstruction:
        exp_name_var = exp_name + "_z_var"
        reconstruction_local(nb_class, nb_img, input_data, x_recon_var, exp_name_var, size, score_reconstruction_zvar,
                             save=save)
    return


def traversal_values(size):
    cdf_traversal = np.linspace(0.05, 0.95, size)
    return stats.norm.ppf(cdf_traversal)


def traversal_values_min_max(min, max, size):
    return np.linspace(min, max, size)


def traversal_latent_prototype(net, model_name, train_test, device, z_var_size, sigma_var=None, nb_class=10, size=8,
                               avg_prototype=True, maj_uc_prototype=True, random=False, batch=None, index=None,
                               save=True):

    """
    Plot traversal latent z_var for one index with z_struct prototype: either with mean prototype or with maj uc.
    :param net:
    :param model_name:
    :return:
    """
    # load both prototype:
    if maj_uc_prototype:
        maj_uc_base = torch.tensor(np.load('binary_encoder_struct_results/uniq_code/uc_maj_class_' + model_name + '_' \
                                           + train_test + '.npy', allow_pickle=True)).unsqueeze(dim=0)
        maj_uc = torch.repeat_interleave(maj_uc_base, size, dim=0)
        z_var_zeros = torch.zeros((size, nb_class, z_var_size))
        maj_uc_proto = torch.cat((z_var_zeros, maj_uc), dim=2)  # shape: size, nb_class, embedding_size

        base_prototype_maj_uc = torch.cat((torch.zeros(1, nb_class, z_var_size), maj_uc_base), dim=2)

        if index is not None:
            # Sweep over linearly spaced coordinates transformed through the
            # inverse CDF (ppf) of a gaussian since the prior of the latent
            # space is gaussian
            # cdf_traversal = np.linspace(0.05, 0.95, size)
            # cont_traversal = stats.norm.ppf(cdf_traversal)
            max_value = np.abs(sigma_var.mean())
            cont_traversal = np.linspace(-max_value, max_value, size)
            for class_id in range(nb_class):
                for column in range(size):
                    maj_uc_proto[column][class_id][index] = cont_traversal[column]

        maj_uc_proto = torch.cat((base_prototype_maj_uc, maj_uc_proto), dim=0)
        maj_uc_proto = maj_uc_proto.permute(1, 0, 2)
        latent_samples = maj_uc_proto
        # Map samples through decoder
        generated_maj_uc = net.decoder(latent_samples)
        traversals_maj_uc = make_grid(generated_maj_uc.data, nrow=size+1)
        traversals_maj_uc = traversals_maj_uc.permute(1, 2, 0)

    if avg_prototype:
        _, avg_z_struct_base = get_z_struct_per_class_VAE(model_name, train_test='test', nb_class=nb_class)
        avg_z_struct_base = torch.tensor(avg_z_struct_base).unsqueeze(dim=0)
        avg_z_struct = torch.repeat_interleave(avg_z_struct_base, size, dim=0)
        z_var_zeros = torch.zeros((size, nb_class, z_var_size))
        avg_struct_proto = torch.cat((z_var_zeros, avg_z_struct), dim=2)    # shape: size, nb_class, embedding_size

        base_prototype_avg_struct = torch.cat((torch.zeros(1, nb_class, z_var_size), avg_z_struct_base), dim=2)

        if index is not None:
            # Sweep over linearly spaced coordinates transformed through the
            # inverse CDF (ppf) of a gaussian since the prior of the latent
            # space is gaussian
            # cdf_traversal = np.linspace(0.05, 0.95, size)
            # cont_traversal = stats.norm.ppf(cdf_traversal)
            max_value = np.abs(sigma_var.mean())
            cont_traversal = np.linspace(-max_value, max_value, size)
            for class_id in range(nb_class):
                for column in range(size):
                    avg_struct_proto[column][class_id][index] = cont_traversal[column]

        avg_struct_proto = torch.cat((base_prototype_avg_struct, avg_struct_proto), dim=0)
        avg_struct_proto = avg_struct_proto.permute(1, 0, 2)
        latent_samples = avg_struct_proto
        # Map samples through decoder
        generated_avg_struct = net.decoder(latent_samples)
        traversals_avg_struct = make_grid(generated_avg_struct.data, nrow=size+1)
        traversals_avg_struct = traversals_avg_struct.permute(1, 2, 0)

    # figure:
    fig, axs = plt.subplots(1, 2, figsize=(30, 15), facecolor='w', edgecolor='k')

    axs[0].set(title=('Latent traversal with z_struct maj uc prototype: {}'.format(model_name)))
    axs[0].imshow(traversals_maj_uc.numpy())
    axs[0].axvline(32, linewidth=5, color='orange')

    axs[1].set(title=('Latent traversal with avg z_struct prototype: {}'.format(model_name)))
    axs[1].imshow(traversals_avg_struct.numpy())
    axs[1].axvline(32, linewidth=5, color='orange')

    plt.show()

    if save:
        fig.savefig("fig_results/traversal_latent/fig_traversal_latent_" + "_" + model_name + ".png")

    return


def real_distribution_model(net, expe_name, z_struct_size, z_var_size, loader, train_test, plot_gaussian=False,
                            save=False, VAE_struct=False, is_vae_var=False):
    path = 'Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test + '_mu_var.npy'

    if not os.path.exists(path):

        net.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        first = True
        with torch.no_grad():
            for x, label in loader:

                data = x
                data = data.to(device)  # Variable(data.to(device))

                # compute loss:
                if is_vae_var:
                    _, latent_representation, _ = net(data)
                    mu_var_iter = latent_representation['mu']
                    sigma_var_iter = latent_representation['log_var']
                else:
                    _, z_struct, z_var, z_var_sample, latent_representation, z, _ = net(data)

                    z_struct_distribution_iter = z_struct
                    mu_var_iter = z_var[:, :z_var_size]
                    sigma_var_iter = z_var[:, z_var_size:]

                if first:
                    mu_var = mu_var_iter
                    sigma_var = sigma_var_iter
                else:
                    mu_var = torch.cat((mu_var, mu_var_iter), 0)
                    sigma_var = torch.cat((sigma_var, sigma_var_iter), 0)

                if VAE_struct:
                    if first:
                        z_struct_distribution = z_struct_distribution_iter
                        first = False
                    else:
                        z_struct_distribution = torch.cat((z_struct_distribution, z_struct_distribution_iter), 0)
        net.train()

        mu_var = torch.mean(mu_var, axis=0)
        sigma_var = torch.mean(sigma_var, axis=0)

        if VAE_struct:
            zeros_proportion = (np.count_nonzero(z_struct_distribution.detach().cpu() == 0, axis=0) * 100.) / len(z_struct_distribution)
            # mu_struct = torch.mean(z_struct_distribution, axis=0)
            # sigma_struct = torch.std(z_struct_distribution, axis=0)
        else:
            # mu_struct = 0
            # sigma_struct = 0
            zeros_proportion = 0

        np.save('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
                '_mu_var.npy', mu_var.detach().cpu())
        np.save('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
                'sigma_var.npy', sigma_var.detach().cpu())
        np.save('Other_results/real_distribution/binary_zeros_proportion_' + expe_name + '_' + train_test +
                'sigma_var.npy', zeros_proportion)
        # np.save('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
        #         'mu_struct.npy', mu_struct)
        # np.save('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
        #         'sigma_struct.npy', sigma_struct)

    mu_var = np.load('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
                     '_mu_var.npy', allow_pickle=True)
    sigma_var = np.load('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
                        'sigma_var.npy', allow_pickle=True)
    encoder_struct_zeros_proportion = np.load('Other_results/real_distribution/binary_zeros_proportion_' + expe_name + '_' + train_test +
                'sigma_var.npy', allow_pickle=True)
    # mu_struct = np.load(
    #     'Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
    #     'mu_struct.npy', allow_pickle=True)
    # sigma_struct = np.load(
    #     'Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
    #     'sigma_struct.npy', allow_pickle=True)

    if plot_gaussian:
        if VAE_struct:
            plt.bar(np.arange(len(encoder_struct_zeros_proportion)), encoder_struct_zeros_proportion,
                              label='Propotion of zeros for each encoder struct component',
                              color='blue')
            plt.show()

        mu = 0
        variance = 1
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)

        # plot figure:
        fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')
        ax.set(title=('Gaussian: ' + expe_name + "_" + train_test))
        ax.plot(x, stats.norm.pdf(x, mu, sigma), label='Gaussian (0, I)', color='red')

        for i in range(len(mu_var)):
            mu_var_iter = mu_var[i]
            variance_var_iter = np.abs(sigma_var[i])
            sigma_var_iter = math.sqrt(variance_var_iter)
            x_var = np.linspace(mu_var_iter - 3 * sigma_var_iter, mu_var_iter + 3 * sigma_var_iter, 100)
            ax.plot(x_var, stats.norm.pdf(x_var, mu_var_iter, sigma_var_iter), label='real data gaussian ' + str(i),
                    color='blue')

        ax.legend(loc=1)
        plt.show()

        if save:
            fig.savefig("fig_results/plot_distribution/fig_plot_distribution_" + expe_name + "_" + train_test + ".png")

    return torch.tensor(mu_var), torch.tensor(sigma_var), encoder_struct_zeros_proportion


def switch_img(net, exp_name, loader, z_var_size):
    with torch.no_grad():
        for x in loader:
            data = x[0]
            data = data.to(device)  # Variable(data.to(device))

            # compute loss:
            x_recons, z_struct, z_var, z_var_sample, latent_representation, z_latent = net(data)

            z_var = z_var_sample
            z_struct = z_struct

    # select two imges:
    ind_1 = np.random.randint(len(z_var))
    ind_2 = np.random.randint(len(z_var))

    assert ind_1 != ind_2, "not lucky ! two random int are same, try again !"

    # original data:
    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
    fig.suptitle('Switch images:')
    ax[0, 0].set_title('Original image 1)')
    ax[0, 0].imshow(data[ind_1].squeeze(dim=0), cmap='gray')
    ax[0, 1].set_title('Original image 2)')
    ax[0, 1].imshow(data[ind_2].squeeze(dim=0), cmap='gray')

    # switch_data:
    z_var_1 = z_var[ind_1]
    z_var_2 = z_var[ind_2]
    z_struct_1 = z_struct[ind_1]
    z_struct_2 = z_struct[ind_2]

    img_switch_1 = torch.cat((z_var_1, z_struct_2))
    img_switch_1 = net.decoder(img_switch_1).detach().numpy().squeeze().squeeze()
    img_switch_2 = torch.cat((z_var_2, z_struct_1))
    img_switch_2 = net.decoder(img_switch_2).detach().numpy().squeeze().squeeze()

    ax[1, 0].imshow(img_switch_1, cmap='gray')
    ax[1, 0].set_title('Swith: var1 + struct2')
    ax[1, 1].imshow(img_switch_2, cmap='gray')
    ax[1, 1].set_title('Swith: var2 + struct1')

    # test if original data:
    img_ori_1 = torch.cat((z_var_1, z_struct_1))
    img_ori_1 = net.decoder(img_ori_1).detach().numpy().squeeze().squeeze()
    img_ori_2 = torch.cat((z_var_2, z_struct_2))
    img_ori_2 = net.decoder(img_ori_2).detach().numpy().squeeze().squeeze()

    ax[2, 0].set_title('Reconstruction: var1 + struct1')
    ax[2, 0].imshow(img_ori_1, cmap='gray')
    ax[2, 1].set_title('Reconstruction: var2 + struct2')
    ax[2, 1].imshow(img_ori_2, cmap='gray')

    plt.show()

    return


def images_generation(net, model_name, batch, train_test, size=(8, 8), mu_var=None, es_distribution=None, std_var=None,
                      z_var_size=None, z_struct_size=None, FID=False, IS=False, LPIPS=False, real_distribution=True,
                      save=True, use_maj_uc=False):
    """
    plot image generation and compute three popular scores.
    :param net:
    :param model_name:
    :param FID:
    :param IS:
    :param LPIPS:
    :return:
    """
    nb_samples = size[0] * size[1]

    if real_distribution:
        # z_var random sample:
        sample_var = std_var * torch.randn(nb_samples, z_var_size) + mu_var
        if use_maj_uc:
            maj_uc = torch.tensor(np.load('binary_encoder_struct_results/uniq_code/uc_maj_class_' + model_name + '_' \
                              + train_test + '.npy', allow_pickle=True))
            # z_struct random:
            sample_struct = torch.zeros(nb_samples, z_struct_size)
            for i in range(len(sample_struct)):
                sample_struct[i] = maj_uc[np.random.randint(len(maj_uc))]
        else:
            # z_struct random:
            mu_struct = torch.tensor(1 - (es_distribution / 100))
            proba_struct = mu_struct.repeat(nb_samples, 1)
            sample_struct = torch.Tensor.float(torch.bernoulli(proba_struct))
        # z random:
        sample = torch.cat((sample_var, sample_struct), dim=1)
    else:
        embedding_size = z_var_size + z_struct_size
        sample = torch.randn(size=(size[0] * size[1], embedding_size))

    # generate:
    generated = net.decoder(sample)
    grid_generation = make_grid(generated.data, nrow=size[1])

    # compute scores:
    FID_score = 0
    IS_score = 0
    LPIPS_score_alex = 0
    LPIPS_score_vgg = 0

    # image should be RGB, IMPORTANT: normalized to [-1,1] for IS and LPIPS score
    # batch = (batch - 0.5)/0.5
    # generated = (generated - 0.5) / 0.5

    original_batch = batch
    generation_batch = generated[:len(batch)]

    if FID:
        FID_score = calculate_fid_given_paths(original_batch,
                                              generation_batch,
                                              batch_size=32,
                                              cuda='',
                                              dims=2048)
        FID_score = np.around(FID_score, 3)
        print('FID score: {}'.format(FID_score))
    if IS:
        IS_score = inception_score(generated,
                                   batch_size=32,
                                   resize=True)
        IS_score = np.around(IS_score[0], 3)
        print('IS score: {}'.format(IS_score))
    if LPIPS:
        loss_fn_alex = lpips.LPIPS(net='alex')  # best forward scores
        loss_fn_vgg = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization

        # image should be RGB, IMPORTANT: normalized to [-1,1]
        LPIPS_score_alex = torch.mean(loss_fn_alex(original_batch, generation_batch)).item()
        LPIPS_score_vgg = torch.mean(loss_fn_vgg(original_batch, generation_batch)).item()
        LPIPS_score_alex = np.around(LPIPS_score_alex, 3)
        LPIPS_score_vgg = np.around(LPIPS_score_vgg, 3)
        print('LPIPS score: alex: {}, vgg: {}'.format(LPIPS_score_alex, LPIPS_score_vgg))

    # plot:
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='w', edgecolor='k')

    samples = grid_generation.permute(1, 2, 0)
    ax.set(title=('Random Images generation from real distribution: {}. Scores: FID(\u2193): {}, IS(\u2191): {},'
                  ' LPIPS (alex) (\u2191): {}, LPIPS (vgg) (\u2191): {}'.format(model_name.split('1_2_1_1_')[-1],
                                                                                FID_score,
                                                                                IS_score,
                                                                                LPIPS_score_alex,
                                                                                LPIPS_score_vgg)))
    ax.imshow(samples.numpy())
    plt.show()

    if save:
        fig.savefig("fig_results/sample/random_sample_" + model_name + ".png")

    return


def manifold_digit(net, model_name, device, size=(20, 20), component_var=0, component_struct=0, random=False,
                   loader=None,
                   mu_var=None, std_var=None, mu_struct=None, std_struct=None, z_var_size=None, z_struct_size=None,
                   img_choice=None):
    """
    display 2D manifold of the digits for one component.
    :param net:
    :param model_name:
    :param size:
    :return:
    """
    # create z latent code:
    if random:
        sample_var = std_var * torch.randn(1, z_var_size) + mu_var
        sample_struct = std_struct * torch.randn(1, z_struct_size) + mu_struct
        sample = torch.cat((sample_var, sample_struct), dim=1)
    else:
        with torch.no_grad():
            for x, label in loader:
                data = x
                data = data.to(device)  # Variable(data.to(device))

                # compute loss:
                _, _, _, _, _, z, _, _ = net(data)
                break

        if img_choice is None:
            img_choice = torch.randint(len(x), (1,))
        sample = z[img_choice].detach()

    # parameters:
    digit_size = 32
    assert component_var < z_var_size, 'Please choose component var in zvar size no more !'
    assert component_struct < z_struct_size, 'Please choose component struct in zstruct size no more !'
    figure = np.zeros((digit_size * size[0], digit_size * size[0]))

    # Construct grid of latent variable values
    grid_x = norm.ppf(np.linspace(0.05, 0.95, size[0]))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, size[0]))

    # decode for each square in the grid
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = sample[0]
            z_sample[component_var] = yi
            z_sample[z_var_size + component_struct] = xi
            z_sample = torch.Tensor(z_sample).to(device)
            x_hat = net.decoder(z_sample)
            x_hat = x_hat.reshape(digit_size, digit_size).to('cpu').detach().numpy()
            figure[(size[0] - 1 - i) * digit_size:(size[0] - 1 - i + 1) * digit_size,
            j * digit_size:(j + 1) * digit_size] = x_hat

    plt.figure(figsize=(10, 10))
    plt.imshow(figure, cmap='gray')
    plt.show()

    return


def same_binary_code(net, model_name, loader, nb_class, train_test=None, save=True, Hmg_dist=True, is_VAE=False,
                     bin_after_GMP=False):
    """
    Compute, save and return uniqs code use for each classes, their percentage.
    If Hmg dst compute, save and return Hmg distance for each class.
    :param net:
    :param model_name:
    :param loader:
    :param nb_class:
    :param train_test:
    :param save:
    :param Hmg_dist:
    :return:
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z_struct_layer_num = get_layer_zstruct_num(net)

    path_model_uniq_code = 'binary_encoder_struct_results/uniq_code/' + model_name + '_' + train_test + '.npy'
    path_model_uniq_code_percent = 'binary_encoder_struct_results/uniq_code/percent_' + model_name + '_' + train_test\
                                   + '.npy'
    path_model_Hmg_dst = 'binary_encoder_struct_results/Hamming_dst/' + model_name + '_' + train_test + '.npy'
    path_model_encoder_struct_embedding = 'binary_encoder_struct_results/encoder_struct_embedding/' + model_name +\
                                          '_' + train_test + '.npy'
    path_model_list_labels_encoder_struct_embedding = 'binary_encoder_struct_results/encoder_struct_embedding' \
                                                      '/labels_list_' + model_name + '_' + train_test + '.npy'

    if os.path.exists(path_model_uniq_code) and \
       os.path.exists(path_model_uniq_code_percent) and \
       os.path.exists(path_model_Hmg_dst) and \
       os.path.exists(path_model_encoder_struct_embedding) and \
       os.path.exists(path_model_list_labels_encoder_struct_embedding):

        print('Load all binary encoder struct values:')
        uniq_code = np.load(path_model_uniq_code, allow_pickle=True)
        Hmg_dst = np.load(path_model_Hmg_dst, allow_pickle=True)
        embedding_struct = np.load(path_model_encoder_struct_embedding, allow_pickle=True)
        labels_list = np.load(path_model_list_labels_encoder_struct_embedding, allow_pickle=True)
        percentage_uniq_code = np.load(path_model_uniq_code_percent, allow_pickle=True)

        embedding_class = []
        for class_id in range(nb_class):
            embed_cl = embedding_struct[np.where(labels_list == class_id)]
            embedding_class.append(embed_cl)

        embedding_class = np.array(embedding_class)  # shape: nb_class, nb_z, z_size

        print('__________------Uniq code:------__________')
        for class_id in range(nb_class):
            print('class {}: unique code: {}/{}. In percent: {}'.format(class_id,
                                                                        len(uniq_code[class_id]),
                                                                        len(embedding_class[class_id]),
                                                                        percentage_uniq_code[class_id]))

        print('__________------Hamming distance:------__________')
        for class_id in range(nb_class):
            print('class {}: Average distance Hamming: {}'.format(class_id, Hmg_dst[class_id]))
        print('average distance: {}'.format(torch.mean(torch.tensor(Hmg_dst))))

    else:
        print('Compute all binary encoder struct values:')
        first = True
        for x, label in loader:
            data = x
            data = data.to(device)  # Variable(data.to(device))
            label = label.to(device)

            # compute loss:
            if is_VAE:
                _, embedding, _, _, _, z, _ = net(data)
            else:
                _, embedding, _, _, _, _, _, _, _, _, _, _ = net(data,
                                                                 z_struct_out=True,
                                                                 z_struct_layer_num=z_struct_layer_num)

            if first:
                labels_list = label.detach()
                embedding_struct = embedding.detach()
                first = False
            else:
                embedding_struct = torch.cat((embedding_struct, embedding.detach()), 0)
                labels_list = torch.cat((labels_list, label.detach()), 0)

        if is_VAE:
            embedding_struct = embedding_struct.cpu().numpy()
        else:
            embedding_struct = embedding_struct.cpu().numpy()
        labels_list = labels_list.cpu().numpy()

        if save:
            np.save(path_model_encoder_struct_embedding, embedding_struct)
            np.save(path_model_list_labels_encoder_struct_embedding, labels_list)

        embedding_class = []
        for class_id in range(nb_class):
            embed_cl = embedding_struct[np.where(labels_list == class_id)]
            embedding_class.append(embed_cl)

        embedding_class = np.array(embedding_class)  # shape: nb_class, nb_z, z_size

        # compute percentage same binary code:
        # first: unique code:
        uniq_code = []
        percentage_uniq_code = []
        for class_id in range(nb_class):
            uniq_code.append(np.unique(embedding_class[class_id], axis=0))
            percentage_uniq_code.append(len(uniq_code[class_id])/len(embedding_class[class_id]))
            print('class {}: unique code: {}/{}. In percent: {}'.format(class_id,
                                                                        len(uniq_code[class_id]),
                                                                        len(embedding_class[class_id]),
                                                                        percentage_uniq_code[class_id]))

        uniq_code = np.array(uniq_code)  # shape: nb_class, nb_different_code, z_size
        percentage_uniq_code = np.array(percentage_uniq_code)
        if save:
            np.save(path_model_uniq_code, uniq_code)
            np.save(path_model_uniq_code_percent, percentage_uniq_code)

        # compute hamming distance: differentiable hamming loss distance.
        if Hmg_dist:
            hamming_distance = nn.PairwiseDistance(p=0, eps=0.0)
            Hmg_dst = []
            for class_id in range(nb_class):
                nb_vector = len(embedding_class[class_id])
                dist = 0
                nb_distance = 0
                for i in range(nb_vector):
                    for j in range(nb_vector):
                        if i == j:
                            pass
                        else:
                            nb_distance += 1
                            distance_hamming_class = hamming_distance(torch.tensor(embedding_class[class_id][i]).unsqueeze(0),
                                                                      torch.tensor(embedding_class[class_id][j]).unsqueeze(0))
                            dist += distance_hamming_class
                Hmg_dst.append((dist/nb_distance))
                print('class {}: Average distance Hamming: {}'.format(class_id, Hmg_dst[class_id]))
        else:
            Hmg_dst = np.zeros(nb_class)
        print('average distance: {}'.format(torch.mean(torch.tensor(Hmg_dst))))

        if save:
            np.save(path_model_Hmg_dst, Hmg_dst)

    return uniq_code, Hmg_dst, percentage_uniq_code


def get_receptive_field_size(net, images):
    """
    compute receptive field of each component for each conv layer.
    :param net:
    :param images:
    :return:
    """

    image = images[0].unsqueeze(dim=0)
    # get activations:
    # a dictionary that keeps saving the activations as they come
    activations = collections.defaultdict(list)

    def save_activations(name, mod, inp, out):
        activations[name].append(out.cpu())

    for name, m in net.named_modules():
        if type(m) == nn.Conv2d:
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activations, name))

    out, _, _, _, _, _, _, _, _, _ = net(image)
     # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
    activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}

    jump = [1]
    receptive_field_size = [1]
    i = 0
    for name, fm in activations.items():
        i += 1
        layer_num = int(name.split('.')[-1])
        stride_size = net.encoder_struct[layer_num].stride[0]
        filter_size = net.encoder_struct[layer_num].kernel_size[0]

        jump.append(jump[-1] * stride_size)  # represent the cumulative stride
        receptive_field_size.append(receptive_field_size[-1] + (filter_size - 1) * jump[-2])  # size of receptive field

    print('receptive field: {}'.format(receptive_field_size))
    return receptive_field_size


def z_struct_code_classes(model_name, nb_class, train_test=None):
    """
    Save embedding class for each class.
    From embedding struct, uniq code and label list for a specific model, compute number of each uniq code used in each
    class and save it.
    :param model_name:
    :param nb_class:
    :param train_test:
    :return:
    """

    path_model_uniq_code = 'binary_encoder_struct_results/uniq_code/' + model_name + '_' + train_test + '.npy'
    path_model_encoder_struct_embedding = 'binary_encoder_struct_results/encoder_struct_embedding/' + model_name +\
                                          '_' + train_test + '.npy'
    path_model_list_labels_encoder_struct_embedding = 'binary_encoder_struct_results/encoder_struct_embedding' \
                                                      '/labels_list_' + model_name + '_' + train_test + '.npy'

    embedding_struct = np.load(path_model_encoder_struct_embedding, allow_pickle=True)
    uniq_code = np.load(path_model_uniq_code, allow_pickle=True)
    labels_list = np.load(path_model_list_labels_encoder_struct_embedding, allow_pickle=True)

    embedding_class = []
    for class_id in range(nb_class):
        embed_cl = embedding_struct[np.where(labels_list == class_id)]
        embedding_class.append(embed_cl)

    embedding_class = np.array(embedding_class)  # shape: nb_class, nb_z, z_size
    np.save('binary_encoder_struct_results/uniq_code/embedding_class_' + model_name + '_' + train_test + '.npy',
            embedding_class)

    global_count = []
    for class_id in range(nb_class):
        count_class = []
        for code in uniq_code[class_id]:
            count = 0
            for z_struct in embedding_class[class_id]:
                if (code == z_struct).all():
                    count += 1
                else:
                    pass
            count_class.append(count)
        global_count.append(np.array(count_class))

    global_count = np.array(global_count)
    np.save('binary_encoder_struct_results/uniq_code/global_count_' + model_name + '_' + train_test + '.npy',
            global_count)

    return


def score_with_best_code_uniq(net, model_name, train_test, loader, z_struct_size, loader_size, bin_after_GMP=False):
    """
    Use majoritary uniq binary code for each class and compute classification score with this uniq code for each class.
    :param net:
    :param model_name:
    :param train_test:
    :param loader:
    :param z_struct_size:
    :param loader_size:
    :return:
    """

    global_count = np.load('binary_encoder_struct_results/uniq_code/global_count_' + model_name + '_'
                           + train_test + '.npy', allow_pickle=True)
    embedding_class = np.load('binary_encoder_struct_results/uniq_code/embedding_class_' + model_name + '_'
                              + train_test + '.npy', allow_pickle=True)
    uniq_code = np.load('binary_encoder_struct_results/uniq_code/' + model_name + '_' + train_test + '.npy',
                        allow_pickle=True)

    more_represented_class_code = []
    for i in range(len(global_count)):
        index_max = np.argmax(global_count[i])
        nb_code = global_count[i][index_max]
        percent_code = np.round(nb_code/len(embedding_class[i])*100, 3)
        print("for class {}, we have {} z_struct with code {}. It represent {}% "
              "of total z_struct class".format(i, nb_code, uniq_code[i][index_max], percent_code))
        more_represented_class_code.append(uniq_code[i][index_max])

    # score prediction with uniq best binary code:
    z_struct_layer_num = get_layer_zstruct_num(net)

    net.eval()
    print('Compute all binary encoder struct values:')
    first = True
    for x, label in loader:

        data = x
        data = data.to(device)  # Variable(data.to(device))
        label = label.to(device)

        data_input = torch.zeros((len(data), z_struct_size))
        for i in range(len(label)):
            data_input[i] = torch.tensor(more_represented_class_code[label[i]])

        data_input = data_input.to(device)

        # compute loss:
        prediction = net.encoder_struct[z_struct_layer_num:](data_input)

        if first:
            prediction_total = prediction.detach()
            labels = label
            first = False
        else:
            prediction_total = torch.cat((prediction_total, prediction.detach()), 0)
            labels = torch.cat((labels, label.detach()), 0)

    net.train()

    predicted = prediction_total.argmax(dim=1, keepdim=True)
    correct = predicted.eq(labels.view_as(predicted)).sum().item()
    scores = correct
    scores = 100. * scores / loader_size

    print("prediction with same best uniq code for test set: {} %".format(scores))

    return scores


def histo_count_uniq_code(model_name, train_test, plot_histo=True, return_percent=False):
    """
    plot histogram of utilization of each uniq code for each class.
    :param net:
    :param model:
    :param name:
    :param loader:
    :return:
    """
    global_count = np.load('binary_encoder_struct_results/uniq_code/global_count_' + model_name + '_'
                           + train_test + '.npy', allow_pickle=True)

    path_global_count_each_code = 'binary_encoder_struct_results/uniq_code/global_count_each_class' + model_name + '_' \
                                  + train_test + '.npy'
    path_code_maj_each_class = 'binary_encoder_struct_results/uniq_code/uc_maj_class_' + model_name + '_' \
                               + train_test + '.npy'

    if os.path.exists(path_global_count_each_code):
        code_class_global = np.array(np.load(path_global_count_each_code, allow_pickle=True))
    else:
        code_class_global = []
        for i in range(len(global_count)):
            code_class_iter = []
            for index in range(len(global_count[i])):
                nb_code = global_count[i][index]
                code_class_iter.append(nb_code)
            code_class_global.append(code_class_iter)
        np.save(path_global_count_each_code, code_class_global)

    # compute percent:
    embedding_class = np.load('binary_encoder_struct_results/uniq_code/embedding_class_' + model_name + '_'
                              + train_test + '.npy', allow_pickle=True)
    uniq_code = np.load('binary_encoder_struct_results/uniq_code/' + model_name + '_' + train_test + '.npy',
                        allow_pickle=True)

    percent_max = []
    code_maj_each_class = []
    for i in range(len(global_count)):
        index_max = np.argmax(global_count[i])
        nb_code = global_count[i][index_max]
        percent_code = np.round(nb_code / len(embedding_class[i]) * 100, 3)
        percent_max.append(percent_code)
        code_maj = uniq_code[i][index_max]
        code_maj_each_class.append(code_maj)

    code_maj_each_class = np.array(code_maj_each_class)
    np.save(path_code_maj_each_class, code_maj_each_class)

    # histo:
    if plot_histo:
        # plot histos:
        nrows = 4
        ncols = 3
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 15), facecolor='w', edgecolor='k')
        fig.suptitle("Histograms uniq binary code each class: {}".format(model_name), fontsize=16)
        for i in range(len(code_class_global)):
            row = i // ncols
            col = i % ncols
            axs[row, col].hist(code_class_global[i], bins=100, alpha=0.75, label="class "+str(i))
            axs[row, col].set_title("Class: {}: [{} %] ".format(i, percent_max[i]), fontsize=12)
        plt.show()

    if return_percent:
        return percent_max
    else:
        return


def score_uniq_code(net, loader, device, z_struct_layer_num, nb_class, z_struct_size, loader_size):
    """
    Function used in solver to get score classification with majoritary uniq binary code and uniq code for each class.
    :param net:
    :param loader:
    :param device:
    :param z_struct_layer_num:
    :param nb_class:
    :param z_struct_size:
    :param loader_size:
    :return:
    """

    # get z_struct binary code:
    net.eval()
    first = True
    for x, label in loader:

        data = x
        data = data.to(device)  # Variable(data.to(device))
        label = label.to(device)

        # compute loss:
        _, embedding, _, _, _, _, _, _, _, _ = net(data, z_struct_out=True, z_struct_layer_num=z_struct_layer_num)

        if first:
            labels_list = label.detach()
            embedding_struct = embedding.detach().squeeze(2).squeeze(2)
            first = False
        else:
            embedding_struct = torch.cat((embedding_struct, embedding.detach().squeeze(2).squeeze(2)), 0)
            labels_list = torch.cat((labels_list, label.detach()), 0)

    # print('zstruct: ', embedding_struct.shape, labels_list.shape)

    # get z_struct per class:
    first = True
    for class_id in range(nb_class):
        # print("class ", class_id)
        embed_cl = embedding_struct[torch.where(labels_list == class_id)]
        # print('embed_cl ', embed_cl.shape)
        uniq_code = torch.unique(embed_cl, dim=0)
        # print('uniq_code ', uniq_code.shape, )

        percentage_uniq_code = len(uniq_code) / len(embed_cl)
        # print('percentage_uniq_code ', percentage_uniq_code)
        # print('class {}: unique code: {}/{}. In percent: {}'.format(class_id,
        #                                                             len(uniq_code),
        #                                                             len(embed_cl),
        #                                                             percentage_uniq_code))
        first_class = True
        for code in uniq_code:
            count = torch.tensor(0)
            for z_struct in embed_cl:
                if (code == z_struct).all():
                    count += 1
                else:
                    pass
            # print('count', count, code)
            if first_class:
                count_class = count.unsqueeze(0)
                first_class = False
            else:
                count_class = torch.cat((count_class, count.unsqueeze(0)), 0)

        index_max = torch.argmax(count_class)
        # print('count', count_class, index_max, uniq_code[index_max])

        if first:
            more_represented_class_code = uniq_code[index_max].unsqueeze(0)
            first = False
        else:
            more_represented_class_code = torch.cat((more_represented_class_code, uniq_code[index_max].unsqueeze(0)), 0)

        # print('more_represented_class_code', more_represented_class_code.shape, more_represented_class_code)

    # get prediction:
    first = True
    for x, label in loader:

        data = x
        data = data.to(device)  # Variable(data.to(device))
        label = label.to(device)

        data_input = torch.zeros((len(data), z_struct_size))
        for i in range(len(label)):
            data_input[i] = torch.tensor(more_represented_class_code[label[i]])

        data_input = data_input.to(device)

        # compute loss:
        prediction = net.encoder_struct[z_struct_layer_num:](data_input)

        if first:
            prediction_total = prediction.detach()
            labels = label
            first = False
        else:
            prediction_total = torch.cat((prediction_total, prediction.detach()), 0)
            labels = torch.cat((labels, label.detach()), 0)

    # compute score:
    predicted = prediction_total.argmax(dim=1, keepdim=True)
    correct = predicted.eq(labels.view_as(predicted)).sum().item()
    scores = correct
    scores = 100. * scores / loader_size

    net.train()

    return scores, more_represented_class_code


def maj_uc_prototype(net, model_name, train_test, z_var_size=None, ES_reconstruction=False):

    maj_uc = np.load('binary_encoder_struct_results/uniq_code/uc_maj_class_' + model_name + '_' \
                     + train_test + '.npy', allow_pickle=True)
    nb_class = 10
    if ES_reconstruction:
        z = torch.tensor(maj_uc)
        x_recon_prototype = net.decoder(z).detach().numpy()
    else:
        zeros_zvar = torch.zeros((nb_class, z_var_size))
        z = torch.cat((zeros_zvar, torch.tensor(maj_uc)), dim=1)
        x_recon_prototype = net.decoder(z).detach().numpy()

    fig = plt.figure(figsize=(10, 5))
    plt.title('Prototype ES (z_var=0) with maj uc: {}'.format(model_name))
    plt.axis('off')

    for k in range(nb_class):
        ax = fig.add_subplot(2, 5, k + 1, xticks=[], yticks=[])
        ax.imshow(x_recon_prototype[k][0], cmap='gray')
        ax.set_title("({})".format(str(k)))
    plt.show()

    return


def plot_prototype(net, model_name, train_test, z_var_size, nb_class=None, avg_zstruct=True, maj_uc_prototype=True,
                   save=True):

    # load both prototype:
    if maj_uc_prototype:
        maj_uc = torch.tensor(np.load('binary_encoder_struct_results/uniq_code/uc_maj_class_' + model_name + '_' \
                         + train_test + '.npy', allow_pickle=True))
        z_var_zeros = torch.zeros((nb_class, z_var_size))
        maj_uc_proto = torch.cat((z_var_zeros, maj_uc), dim=1)
        recons_maj_uc_proto = net.decoder(maj_uc_proto).detach().numpy()

        # figure:
        fig = plt.figure(figsize=(20, 10))
        plt.title('Prototype maj uc z_struct, model: {}'.format(model_name))
        plt.axis('off')

        for k in range(nb_class):
            ax = fig.add_subplot(2, 5, k + 1, xticks=[], yticks=[])
            ax.imshow(recons_maj_uc_proto[k][0], cmap='gray')
            ax.set_title(str(k))
        plt.show()

        if save:
            fig.savefig(
                "fig_results/prototype_z_struct_class/maj_uc_proto_struct_" + model_name + train_test + ".png")

    if avg_zstruct:
        _, avg_z_struct = get_z_struct_per_class_VAE(model_name, train_test='test', nb_class=nb_class)
        avg_z_struct = torch.tensor(avg_z_struct)
        z_var_zeros = torch.zeros((nb_class, z_var_size))
        avg_struct_proto = torch.cat((z_var_zeros, avg_z_struct), dim=1)
        recons_avg_struct_proto = net.decoder(avg_struct_proto).detach().numpy()

        # figure:
        fig = plt.figure(figsize=(20, 10))
        plt.title('Prototype avg z_struct, model: {}'.format(model_name))
        plt.axis('off')

        for k in range(nb_class):
            ax = fig.add_subplot(2, 5, k + 1, xticks=[], yticks=[])
            ax.imshow(recons_avg_struct_proto[k][0], cmap='gray')
            ax.set_title(str(k))
        plt.show()

        if save:
            fig.savefig(
                "fig_results/prototype_z_struct_class/avg_proto_struct_" + model_name + train_test + ".png")

    return


# __________Test for VAE var classifier:
def VAE_var_classifier_score(net, loader):

    nb_data = len(loader.dataset)
    classification_score = 0
    with torch.no_grad():
        for x in loader:

            data = x[0]
            data = data.to(device)  # Variable(data.to(device))
            labels = x[1]
            labels = labels.to(device)

            # compute loss:
            _, _, prediction_var = net(data)

            predicted = prediction_var.argmax(dim=1, keepdim=True)
            correct = predicted.eq(labels.view_as(predicted)).sum().item()
            classification_score_iter = float(correct)
            classification_score += classification_score_iter

    score = 100. * classification_score / nb_data

    return score


def get_activation_values_VAE_var(net, exp_name, image):

    activations_path = 'regions_of_interest/activations/activations_' + exp_name + '.npy'

    if os.path.exists(activations_path):
        print("path already exist, load it !")
    else:
        # a dictionary that keeps saving the activations as they come
        activations = collections.defaultdict(list)

        def save_activations_VAE_var(name, mod, inp, out):
            activations[name].append(out.cpu())

        # Registering hooks for all the Conv2d layers
        # Note: Hooks are called EVERY TIME the module performs a forward pass. For modules that are
        # called repeatedly at different stages of the forward pass (like RELUs), this will save different
        # activations. Editing the forward pass code to save activations is the way to go for these cases.
        for name, m in net.named_modules():
            m.register_forward_hook(partial(save_activations_VAE_var, name))

        out, _, _ = net(image)

        # concatenate all the outputs we saved to get the the activations for each layer for the whole dataset
        activations = {name: torch.cat(outputs, 0) for name, outputs in activations.items()}

        np.save(activations_path, activations)

    return


def load_plot_histo_activations(model_name):

    activations = np.load('regions_of_interest/activations/activations_' + model_name + '.npy', allow_pickle=True)
    print(activations.shape)

    return
