import torch
import torch.nn.functional as F

EPS = 1e-12


def compute_scores(net, loader, device, loader_size, nb_class, ratio_reg, z_struct_layer_num,
                   other_ratio, loss_min_distance_cl, contrastive_criterion, without_acc, lambda_classification,
                   lambda_contrastive, lambda_ratio_reg, diff_var, lambda_var_intra, lambda_var_inter,
                   lambda_var_distance, lambda_distance_mean, z_struct_out, Hmg_dst_loss, lambda_hmg_dst):

    classification_loss = 0
    classification_score = 0
    nb_data = loader_size
    ratio_loss = 0
    diff_var_loss = 0
    loss_distance_cl = 0
    loss_distance_mean = 0
    contrastive_loss = 0
    variance_intra = 0
    variance_inter = 0
    total_loss = 0

    with torch.no_grad():
        for x in loader:

            data = x[0]
            labels = x[1]
            data = data.to(device)  # Variable(data.to(device))
            labels = labels.to(device)  # Variable(labels.to(device))

            # compute loss:
            prediction, embedding, ratio, var_distance_classes_train, \
            intra_var, mean_distance_intra_class, inter_var,\
            global_avg_Hmg_dst, _, _, _, _ = net(data,
                                                 labels=labels,
                                                 nb_class=nb_class,
                                                 use_ratio=ratio_reg,
                                                 z_struct_out=z_struct_out,
                                                 z_struct_layer_num=z_struct_layer_num,
                                                 loss_min_distance_cl=loss_min_distance_cl,
                                                 Hmg_dst_loss=Hmg_dst_loss)

            # classification loss:
            classification_loss_iter = F.nll_loss(prediction, labels)
            classification_loss += classification_loss_iter

            # classification score:
            classification_score_iter = compute_scores_prediction(prediction, labels)
            classification_score += classification_score_iter

            # ratio loss:
            ratio_loss_iter = ratio
            ratio_loss += ratio_loss_iter

            # contrastive loss:
            if contrastive_loss:
                # loss take embedding, not prediction
                embedding = embedding.squeeze(axis=-1).squeeze(axis=-1)
                contrastive_loss = contrastive_criterion(embedding, labels)
                contrastive_loss_iter = contrastive_loss
                contrastive_loss += contrastive_loss_iter

            # difference variances loss:
            diff_var_loss_iter = variance_intra - variance_inter
            diff_var_loss += diff_var_loss_iter

            # variance losses:
            variance_intra += intra_var
            variance_inter += inter_var

            # min distance classes loss:
            loss_distance_cl_iter = var_distance_classes_train
            loss_distance_cl += loss_distance_cl_iter

            # distance mean loss:
            target_mean = torch.tensor(10)
            target_mean = target_mean.to(device)
            loss_distance_mean_iter = -(torch.abs(1 / (target_mean - mean_distance_intra_class + EPS)))
            loss_distance_mean += loss_distance_mean_iter

            # hamming distance loss:
            hmg_dst_loss_iter = global_avg_Hmg_dst

            # Total loss:
            total_loss_iter = 0
            if not without_acc:
                total_loss_iter += classification_loss_iter * lambda_classification

            if contrastive_loss:
                total_loss_iter += contrastive_loss_iter * lambda_contrastive

            if ratio_reg:
                # ratio loss:
                if other_ratio:
                    ratio = -(ratio_loss_iter * lambda_ratio_reg)
                else:
                    ratio = ratio_loss_iter * lambda_ratio_reg
                total_loss_iter += ratio

            if diff_var:
                loss_diff_var_iter_lambda = (lambda_var_intra * intra_var) - (lambda_var_inter * inter_var)
                total_loss_iter += loss_diff_var_iter_lambda

            if loss_min_distance_cl:
                total_loss_iter += loss_distance_cl_iter * lambda_var_distance

            if loss_distance_mean:
                total_loss_iter += loss_distance_mean_iter * lambda_distance_mean

            if Hmg_dst_loss:
                total_loss_iter += hmg_dst_loss_iter * lambda_hmg_dst

            total_loss += total_loss_iter

    # loss
    classification_loss = classification_loss / len(loader)
    total_loss_iter = total_loss_iter / len(loader)
    ratio_loss = ratio_loss / len(loader)
    contrastive_loss = contrastive_loss / len(loader)
    diff_var_loss = diff_var_loss / len(loader)
    variance_intra = variance_intra / len(loader)
    variance_inter = variance_inter / len(loader)
    loss_distance_cl = loss_distance_cl / len(loader)
    loss_distance_mean = loss_distance_mean / len(loader)
    total_loss = total_loss / len(loader)

    # scores:
    score = 100. * classification_score / nb_data

    return score, classification_loss, total_loss_iter, ratio_loss, contrastive_loss, diff_var_loss, variance_intra, \
           variance_inter, loss_distance_cl, loss_distance_mean, total_loss


def compute_scores_VAE(net, loader, loader_size, device, lambda_BCE, beta, is_vae_var=False, ES_reconstruction=False,
                       is_prediction_var=False, loss_struct_recons_class=False, size_average=False, is_VAE=False,
                       lambda_EV_class=None, lambda_struct_recons_class=None, lambda_recons=None,
                       loss_z_struct_class=False):

    classification_loss_EV = 0
    classification_loss_z_struct = 0
    classification_ES_score = 0
    BCE_loss = 0
    BCE_loss_ES = 0
    KLD_loss = 0
    total_loss = 0
    classification_score_EV = 0
    nb_data = loader_size

    with torch.no_grad():
        for x in loader:

            data = x[0]
            data = data.to(device)  # Variable(data.to(device))
            labels = x[1]
            labels = labels.to(device)

            # total loss:
            total_loss_iter = 0

            # compute loss:
            if is_vae_var:
                x_recons, latent_representation, prediction_var = net(data)
            else:
                x_recons, z_struct, z_var, z_var_sample, latent_representation, z, \
                prediction_var, prediction_struct, \
                global_avg_Hmg_dst = net(data,
                                         loss_struct_recons_class=loss_struct_recons_class,
                                         device=device,
                                         loss_z_struct_class=loss_z_struct_class)

            if ES_reconstruction:
                # BCE loss:
                BCE_loss_iter_ES = F.mse_loss(x_recons, data, size_average=size_average)
                BCE_loss_ES += BCE_loss_iter_ES

                total_loss_iter += BCE_loss_ES

            if is_prediction_var:
                classification_loss_EV_iter = F.nll_loss(prediction_var, labels, size_average=size_average)
                classification_loss_EV += classification_loss_EV_iter

                total_loss_iter += classification_loss_EV * lambda_EV_class

                # classification score:
                classification_score_EV_iter = compute_scores_prediction(prediction_var, labels)
                classification_score_EV += classification_score_EV_iter

            if loss_struct_recons_class or loss_z_struct_class:
                classification_loss_z_struct_iter = F.nll_loss(prediction_struct, labels, size_average=size_average)
                classification_loss_z_struct += classification_loss_z_struct_iter

                total_loss_iter += classification_loss_z_struct * lambda_struct_recons_class

                # classification score:
                classification_score_ES_iter = compute_scores_prediction(prediction_struct, labels)
                classification_ES_score += classification_score_ES_iter

            if (is_VAE and not ES_reconstruction) or is_vae_var:
                mu = latent_representation['mu']
                logvar = latent_representation['log_var']

                # BCE loss:
                BCE_loss_iter = F.mse_loss(x_recons, data, size_average=size_average)
                BCE_loss += BCE_loss_iter
                # KL divergence loss:
                if size_average:
                    KLD_loss_iter = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                else:
                    KLD_loss_iter = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                KLD_loss += KLD_loss_iter

                total_loss_iter += ((lambda_BCE * BCE_loss_iter) + (beta * KLD_loss_iter)) * lambda_recons

            total_loss += total_loss_iter

    # losses:
    BCE_loss_ES = BCE_loss_ES / len(loader)
    classification_loss_EV = classification_loss_EV / len(loader)
    classification_loss_z_struct = classification_loss_z_struct / len(loader)

    BCE_loss = BCE_loss / len(loader)
    KLD_loss = KLD_loss / len(loader)
    total_loss = total_loss / len(loader)

    # scores:
    classification_score_EV = 100. * classification_score_EV / nb_data
    classification_ES_score = 100. * classification_ES_score / nb_data

    print("score EV: {}. Score ES: {}".format(classification_score_EV, classification_ES_score))

    return total_loss, BCE_loss, KLD_loss, BCE_loss_ES, classification_loss_EV, classification_loss_z_struct, \
           classification_score_EV, classification_ES_score


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
