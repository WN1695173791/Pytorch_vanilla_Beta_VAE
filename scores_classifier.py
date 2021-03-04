import torch
import torch.nn.functional as F


def compute_scores(net, loader, device, loader_size, loss_min_distance_cl, nb_class, z_struct_out, z_struct_layer_num):

    classification_loss = 0
    classification_score = 0
    mean_distance_intra_class = 0
    nb_data = loader_size

    with torch.no_grad():
        for x in loader:

            data = x[0]
            labels = x[1]
            data = data.to(device)  # Variable(data.to(device))
            labels = labels.to(device)  # Variable(labels.to(device))

            # compute loss:
            prediction, _, _, _, _, mean_distance_intra_class = net(data,
                                                                    z_struct_out=z_struct_out,
                                                                    z_struct_layer_num=z_struct_layer_num,
                                                                    labels=labels,
                                                                    nb_class=nb_class,
                                                                    loss_min_distance_cl=loss_min_distance_cl)

            # classification loss:
            classification_loss_iter = F.nll_loss(prediction, labels)
            classification_loss += classification_loss_iter

            # classification score:
            classification_score_iter = compute_scores_prediction(prediction, labels)
            classification_score += classification_score_iter

            # mean_distance_intra_class loss:
            mean_distance_intra_class_iter = mean_distance_intra_class
            mean_distance_intra_class += mean_distance_intra_class

    # loss
    loss = classification_loss / len(loader)
    mean_distance_intra_class = mean_distance_intra_class / len(loader)
    # scores:
    score = 100. * classification_score / nb_data

    return score, loss, mean_distance_intra_class


def compute_scores_prediction(prediction, labels):
    """
    return nb of correct prediction for the current batch
    :param prediction:
    :param labels:
    :return:
    """
    predicted = prediction.argmax(dim=1, keepdim=True)
    correct = predicted.eq(labels.view_as(predicted)).sum().item()
    scores = correct
    return float(scores)
