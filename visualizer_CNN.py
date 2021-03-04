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
from visualizer import get_checkpoints_scores_CNN

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


def get_layer_zstruct_num(net, net_type):
    if net_type == 'Custom_CNN':
        add_layer = 1
    elif net_type == 'Custom_CNN_BK':
        add_layer = 1

    # get layer num for GMP:
    for name, m in net.named_modules():
        if type(m) == nn.AdaptiveMaxPool2d:
            z_struct_layer_num = int(name.split('.')[-1])

    return z_struct_layer_num + add_layer


def compute_z_struct(net_trained, exp_name, loader, train_test=None, net_type=None, return_results=False):
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
        z_struct_layer_num = get_layer_zstruct_num(net_trained, net_type)

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

            _, z_struct, _, _, _ = net_trained(input_data,
                                               z_struct_out=True,
                                               z_struct_layer_num=z_struct_layer_num)
            pred, _, _, _, _ = net_trained(input_data)

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


def compute_z_struct_representation_noised(net, exp_name, train_test=None, nb_repeat=100, nb_class=10, net_type=None):
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
        z_struct_layer_num = get_layer_zstruct_num(net, net_type)
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
                pred, _, _, _, _ = net(z_struct_representation_noised, z_struct_prediction=True,
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


def get_receptive_field(net, img_size):
    """
    return size of receptive field for a specific neuron in the model.
    :param net:
    :param img_size:
    :return:
    """
    receptive_field_dict = 0
    # receptive_field_dict = receptive_field(net, img_size, return_zstruct_RF=True)

    return receptive_field_dict


def ratio(exp_name, train_test=None, cat=None, other_ratio=False):
    """
    compute the ratio between std of one class and std of all data.
    :param exp_name:
    :param train_test:
    :return:
    """
    z_struct_representation, label_list, _ = load_z_struct_representation(exp_name, train_test=train_test)
    representation_z_struct_class = load_z_struct_representation_per_class(exp_name, train_test=train_test)
    nb_class = len(representation_z_struct_class)

    z_struct_mean_global = np.mean(z_struct_representation, axis=0)
    z_struct_std_global = np.std(z_struct_representation, axis=0)

    z_struct_mean_global_per_class = []
    z_struct_std_global_per_class = []
    for class_id in range(nb_class):
        z_struct_mean_global_per_class.append(np.mean(representation_z_struct_class[class_id], axis=0))
        z_struct_std_global_per_class.append(np.std(representation_z_struct_class[class_id], axis=0))

    z_struct_mean_global_per_class = np.array(z_struct_mean_global_per_class)
    z_struct_std_global_per_class = np.array(z_struct_std_global_per_class)

    ratio_std_per_class = []
    for class_id in range(nb_class):
        ratio_std_per_class.append(z_struct_std_global_per_class[class_id] / z_struct_std_global)

    ratio_std_per_class = np.array(ratio_std_per_class)

    path_save = 'values_CNNs_to_compare/' + str(cat) + '/ratio_std_each_class_' + exp_name + train_test + '.npy'
    np.save(path_save, ratio_std_per_class)
    path_save = 'values_CNNs_to_compare/' + str(cat) + '/ratio_std_mean_' + exp_name + train_test + '.npy'
    np.save(path_save, np.mean(ratio_std_per_class))
    path_save = 'values_CNNs_to_compare/' + str(cat) + '/std_per_class_' + exp_name + train_test + '.npy'
    np.save(path_save, z_struct_std_global_per_class)
    path_save = 'values_CNNs_to_compare/' + str(cat) + '/std_global_' + exp_name + train_test + '.npy'
    np.save(path_save, np.mean(z_struct_std_global))

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

        # normalized z_struct:
        z_struct_representation_normalized = z_struct_representation / np.linalg.norm(z_struct_representation)
        representation_z_struct_class_normalized = []
        for class_id in range(nb_class):
            z_struct_class = z_struct_representation_normalized[np.where(label_list == class_id)]
            representation_z_struct_class_normalized.append(z_struct_class)

        representation_z_struct_class_normalized = np.array(representation_z_struct_class_normalized)

        z_struct_mean_global_per_class_normalized = []
        z_struct_std_global_per_class_normalized = []
        for class_id in range(nb_class):
            z_struct_mean_global_per_class_normalized.append(np.mean(representation_z_struct_class_normalized[class_id],
                                                                     axis=0))
            z_struct_std_global_per_class_normalized.append(np.std(representation_z_struct_class_normalized[class_id],
                                                                   axis=0))

        variance_intra_class_normalized = np.square(z_struct_std_global_per_class_normalized)
        variance_intra_class_normalized_mean_components = np.mean(variance_intra_class_normalized, axis=0)
        variance_inter_class_normalized = np.square(np.std(z_struct_mean_global_per_class_normalized, axis=0))
        ratio_variance_normalized = variance_intra_class_normalized_mean_components / \
                                    (variance_inter_class_normalized + EPS)
        ratio_variance_normalized_mean = np.mean(ratio_variance_normalized)

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


def plot_resume(net, exp_name, is_ratio, is_distance_loss, cat=None, train_test=None, path_scores=None, save=True):
    """
    plot interesting values to resume experiementation behavior: distance matrix, loss, acc, ratio, var intra, var inter
    :param net:
    :param exp_name:
    :param train_test:
    :return:
    """

    ratio_variance_mean, variance_intra_class, variance_inter_class = ratio(exp_name,
                                                                            train_test=train_test,
                                                                            cat=cat)

    variance_intra_class_mean_per_class = np.mean(variance_intra_class, axis=1)
    variance_intra_class_mean = np.mean(variance_intra_class_mean_per_class)
    variance_inter_class_mean = np.mean(variance_inter_class)

    nb_class = len(variance_intra_class)

    # print('ratio:', ratio_variance_mean)
    # print('vari intra mean per class:', variance_intra_class_mean_per_class.shape)
    # print('var intra mean', variance_intra_class_mean)
    # print('var inter', variance_inter_class.shape)
    # print('var inter mean', variance_inter_class_mean)

    # define figure:____________________________________________________________________________________________________
    fig, axs = plt.subplots(nrows=2, ncols=2, gridspec_kw={'width_ratios': [2, 1.5], 'height_ratios': [1, 1]},
                            figsize=(30, 20), facecolor='w', edgecolor='k')
    fig.suptitle('Resume results for model: {}'.format(exp_name), fontsize=16)

    # loss and acc:____________________________________________________________________________________________________
    _, epochs, train_score, test_score, total_loss_train, total_loss_test, ratio_train_loss, \
    ratio_test_loss, class_loss_train, class_loss_test, \
    var_distance_classes_train, var_distance_classes_test = get_checkpoints_scores_CNN(net,
                                                                                       path_scores,
                                                                                       exp_name,
                                                                                       is_ratio=is_ratio,
                                                                                       is_distance_loss=is_distance_loss)
    # get accuracy train and test:
    acc_train_last_epoch = train_score[-1]
    acc_test_last_epoch = test_score[-1]

    axs[0, 0].set(xlabel='nb_iter', ylabel='loss', title=('Losses: ' + exp_name))
    axs[0, 0].plot(epochs, total_loss_train, label='train')
    axs[0, 0].plot(epochs, total_loss_test, label='test')
    if is_ratio:
        axs[0, 0].plot(epochs, ratio_train_loss, label='ratio loss train')
        axs[0, 0].plot(epochs, ratio_test_loss, label='ratio loss test')
        axs[0, 0].plot(epochs, class_loss_train, label='classification loss train')
        axs[0, 0].plot(epochs, class_loss_test, label='classification loss test')
    if is_distance_loss:
        axs[0, 0].plot(epochs, var_distance_classes_train, label='distance between class train')
        axs[0, 0].plot(epochs, var_distance_classes_test, label='distance between class train')
    axs[0, 0].legend(loc=1)

    # 2d projection:____________________________________________________________________________________________________
    z_struct_representation, label_list, _ = load_z_struct_representation(exp_name, train_test=train_test)

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(z_struct_representation)
    t = reduced.transpose()

    lim_min = np.min(t)
    lim_max = np.max(t)

    axs[1, 0].set(xlabel='nb_iter', ylabel='loss', title='PCA of z_struct for each images')
    axs[1, 0].set(xlim=(lim_min, lim_max), ylim=(lim_min, lim_max))

    x = np.arange(nb_class)
    ys = [i + x + (i * x) ** 2 for i in range(10)]
    colors = cm.rainbow(np.linspace(0, 1, len(ys)))

    for c, color in zip(range(nb_class), colors):
        x = np.array(t[0])[np.concatenate(np.argwhere(np.array(label_list) == c)).ravel()]
        y = np.array(t[1])[np.concatenate(np.argwhere(np.array(label_list) == c)).ravel()]
        axs[1, 0].scatter(x, y, alpha=0.6, color=color, label='class ' + str(c))
    axs[1, 0].legend(loc=1)

    # distance matrix: _________________________________________________________________________________________________
    distance_inter_class = distance_matrix(net, exp_name, train_test=train_test, plot_fig=False)

    axs[0, 1].set(title='Distances matrix between mean classes')
    img = axs[0, 1].matshow(distance_inter_class, cmap=plt.cm.rainbow)
    fig.colorbar(img, ticks=[-1, 0, 1], fraction=0.045, ax=axs[0, 1])
    for x in range(distance_inter_class.shape[0]):
        for y in range(distance_inter_class.shape[1]):
            axs[0, 1].text(x, y, "%0.2f" % distance_inter_class[x, y], size=12, color='black', ha="center", va="center")

    # values in array: _________________________________________________________________________________________________
    axs[1, 1].set(title='Statistics')

    text_titles = ["Ratio: ",
                   "Variance intra class mean: ",
                   # "Variance intra class mean per class: ",
                   "Variance inter class mean: ",
                   "Accuracy train: ",
                   "Accuracy test: "]
    n_lines = len(text_titles)

    text_values = [ratio_variance_mean,
                   variance_intra_class_mean,
                   # variance_intra_class_mean_per_class,
                   variance_inter_class_mean,
                   acc_train_last_epoch,
                   acc_test_last_epoch]

    for y in range(n_lines):
        axs[1, 1].text(.1,
                       1 - (y + 0.4) / n_lines,
                       text_titles[y] + str(text_values[y]),
                       size=20,
                       color='black',
                       va="center")

    fig.tight_layout()
    plt.show()

    path_save = 'fig_results/resume/plot_resume_' + exp_name + '_' + train_test + '.png'
    if save and not os.path.exists(path_save):
        fig.savefig(path_save)

    return
