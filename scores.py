import torch
import torch.nn.functional as F
import numpy as np

EPS = 1e-12


def compute_scores(net, loader, device, latent_spec, nb_data, is_partial_rand_class, random_percentage, is_E1,
                   is_zvar_sim_loss, is_C, is_noise_stats, is_perturbed_score):
    """
    compute all sample_scores
    :param
    :return:
    """

    is_continuous = 'cont' in latent_spec
    is_discrete = 'disc' in latent_spec
    is_both_continue = 'cont_var' in latent_spec
    is_both_discrete = 'disc_var' in latent_spec

    # Calculate dimensions of latent distribution
    latent_disc_dim = 0
    latent_disc_dim_var = 0
    latent_disc_dim_class = 0
    if is_discrete:
        latent_disc_dim += sum([dim for dim in latent_spec['disc']])
    if is_both_continue:
        latent_cont_class_dim = latent_spec['cont_class']
    if is_both_discrete:
        latent_disc_dim_var += sum([dim for dim in latent_spec['disc_var']])
        latent_disc_dim_class += sum([dim for dim in latent_spec['disc_class']])

    scores = {}
    losses = {}

    recons_loss = 0
    classification_loss = 0
    classification_partial_rand_loss = 0
    kl_var_loss = 0
    kl_class_loss = 0
    total_kld = 0
    beta_vae_loss = 0
    zvar_sim_loss = 0

    scores_Zc_Zd = 0
    score_Zc_random_Zd = 0
    score_Zc_Zd_random = 0
    score_Zc_pert_Zd = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    score_Zc_Zd_pert = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    score_zc_zd_partial = 0

    nb_data = nb_data

    # to compute proba pred:
    stats_prediction = {'label': [], 'proba': [], 'proba_noised': [[] for i in range(latent_cont_class_dim)]}
    nb_class = 10

    if not net.training:
        is_prediction = True

    i = 0
    with torch.no_grad():
        for x in loader:
            i += 1
            # print("Dataset : compute sample_scores and loss: batch {}/{}".format(batch, len(loader)))
            batch_size = len(x[0])

            data = x[0]
            labels = x[1]
            data = data.to(device)  # Variable(data.to(device))
            labels = labels.to(device)  # Variable(labels.to(device))

            # compute loss:
            x_recon, _, _, latent_representation, latent_sample, \
            latent_sample_variability, latent_sample_class, latent_sample_random_continue, prediction, pred_noised, \
            prediction_partial_rand_class, prediction_random_variability, prediction_random_class, \
            prediction_zc_pert_zd, prediction_zc_zd_pert, z_var, \
            z_var_reconstructed = net(data,
                                      is_perturbed_score=is_perturbed_score,
                                      is_noise_stats=is_noise_stats,
                                      is_prediction=is_prediction,
                                      both_continue=is_both_continue,
                                      both_discrete=is_both_discrete,
                                      is_partial_rand_class=is_partial_rand_class,
                                      random_percentage=random_percentage,
                                      is_E1=is_E1,
                                      is_zvar_sim_loss=is_zvar_sim_loss)

            # reconstruction loss
            recons_loss_iter = (F.mse_loss(x_recon, data, size_average=False).div(
                batch_size))

            classification_loss_iter = 0
            if is_C:
                # classification loss:
                classification_loss_iter = F.nll_loss(prediction, labels, size_average=False).div(
                    batch_size)

            # zvar_sim_loss loss:
            zvar_sim_loss_iter = 0
            if is_zvar_sim_loss:
                zvar_sim_loss_iter = F.mse_loss(z_var, z_var_reconstructed, size_average=False).div(
                    batch_size)

            # partial classification:
            classification_partial_rand_loss_iter = 0
            if is_partial_rand_class:
                classification_partial_rand_loss_iter = (F.nll_loss(prediction_partial_rand_class, labels,
                                                                    size_average=False).div(
                    batch_size))
            if is_both_continue:
                mu_var, logvar_var = latent_representation['cont_var']
                kl_cont_loss_var = kl_divergence(mu_var, logvar_var)
                mu_class, logvar_class = latent_representation['cont_class']
                kl_cont_loss_class = kl_divergence(mu_class, logvar_class)
                kl_var_loss_iter = kl_cont_loss_var
                kl_class_loss_iter = kl_cont_loss_class
            elif is_both_discrete:
                kl_disc_loss_var = _kl_multiple_discrete_loss(latent_representation['disc_var'])
                kl_disc_loss_class = _kl_multiple_discrete_loss(latent_representation['disc_class'])
                kl_var_loss_iter = kl_disc_loss_var
                kl_class_loss_iter = kl_disc_loss_class
            else:
                if is_continuous:
                    mu, logvar = latent_representation['cont']
                    kl_cont_loss = kl_divergence(mu, logvar)
                    kl_var_loss_iter = kl_cont_loss
                if is_discrete:
                    kl_disc_loss = _kl_multiple_discrete_loss(latent_representation['disc'])
                    kl_class_loss_iter = kl_disc_loss

            total_kld_iter = (kl_var_loss_iter + kl_class_loss_iter)

            beta_vae_loss_iter = recons_loss_iter + total_kld_iter + classification_loss_iter + \
                                 classification_partial_rand_loss_iter + zvar_sim_loss_iter

            recons_loss += recons_loss_iter.item()
            classification_loss += classification_loss_iter
            classification_partial_rand_loss += classification_partial_rand_loss_iter
            kl_var_loss += kl_var_loss_iter
            kl_class_loss += kl_class_loss_iter
            total_kld += total_kld_iter
            beta_vae_loss += beta_vae_loss_iter.item()
            zvar_sim_loss += zvar_sim_loss_iter

            if is_C:
                # compute sample_scores:
                # create different representation of data:
                prediction_zc_zd = prediction
                prediction_zc_random_zd = prediction_random_variability
                prediction_zc_zd_random = prediction_random_class
                prediction_zc_pert_zd = prediction_zc_pert_zd
                prediction_zc_zd_pert = prediction_zc_zd_pert
               
                if is_noise_stats:
                    # compute score with one bit random
                    stats_prediction = compute_one_bit_random_score(prediction_zc_random_zd, labels, stats_prediction,
                                                                    pred_noised)

                # compute classification score with original data representation latent:
                # return nb of correct prediction for the current batch
                scores_Zc_Zd_iter = compute_scores_pred(prediction_zc_zd, labels)
                if is_partial_rand_class:
                    score_zc_zd_partial_iter = compute_scores_pred(prediction_partial_rand_class, labels)
                else:
                    score_zc_zd_partial_iter = 0

                score_Zc_random_Zd_iter = compute_scores_pred(prediction_zc_random_zd, labels)
                score_Zc_Zd_random_iter = compute_scores_pred(prediction_zc_zd_random, labels)

                if not is_both_discrete and is_perturbed_score:
                    score_Zc_pert_Zd_iter = compute_Zc_pert_Zd_scores(prediction_zc_pert_zd, labels)
                    score_Zc_Zd_pert_iter = compute_Zc_pert_Zd_scores(prediction_zc_zd_pert, labels)
                else:
                    score_Zc_pert_Zd_iter = 0
                    score_Zc_Zd_pert_iter = 0

                scores_Zc_Zd += scores_Zc_Zd_iter
                score_zc_zd_partial += score_zc_zd_partial_iter
                score_Zc_random_Zd += score_Zc_random_Zd_iter
                score_Zc_Zd_random += score_Zc_Zd_random_iter
                if not is_both_discrete and is_perturbed_score:
                    score_Zc_pert_Zd = [sum(x) for x in zip(score_Zc_pert_Zd_iter, score_Zc_pert_Zd)]
                    score_Zc_Zd_pert = [sum(x) for x in zip(score_Zc_Zd_pert_iter, score_Zc_Zd_pert)]
                else:
                    score_Zc_pert_Zd = 0
                    score_Zc_Zd_pert = 0
    # loss
    losses['recon_loss'] = recons_loss / nb_data
    losses['classification_loss'] = classification_loss / nb_data
    losses['classification_partial_rand_loss'] = classification_partial_rand_loss / nb_data
    losses['kl_var_loss'] = kl_var_loss / nb_data
    losses['kl_class_loss'] = kl_class_loss / nb_data
    losses['total_kld'] = total_kld / nb_data
    losses['beta_vae_loss'] = beta_vae_loss / nb_data
    losses['zvar_sim_loss'] = zvar_sim_loss / nb_data

    mean_proba_per_class = 0
    std_proba_per_class = 0
    mean_proba_per_class_noised = 0
    std_proba_per_class_noised = 0

    if is_C:
        if is_noise_stats:
            # compute proba:
            mean_proba_per_class, std_proba_per_class, mean_proba_per_class_noised, \
            std_proba_per_class_noised = compute_proba(stats_prediction,
                                                       nb_class,
                                                       pred_noised)
        # Accuracy
        scores_Zc_Zd = 100. * scores_Zc_Zd / nb_data
        score_zc_zd_partial = 100 * score_zc_zd_partial / nb_data
        score_Zc_random_Zd = 100. * score_Zc_random_Zd / nb_data
        score_Zc_Zd_random = 100. * score_Zc_Zd_random / nb_data
        if not is_both_discrete and is_perturbed_score:
            score_Zc_pert_Zd = [(i * 100. / nb_data) for i in score_Zc_pert_Zd]
            score_Zc_Zd_pert = [(i * 100. / nb_data) for i in score_Zc_Zd_pert]
        else:
            score_Zc_pert_Zd = 0
            score_Zc_Zd_pert = 0

        scores['Zc_Zd'] = scores_Zc_Zd
        scores['score_zc_zd_partial'] = score_zc_zd_partial
        scores['Zc_random_Zd'] = score_Zc_random_Zd
        scores['Zc_Zd_random'] = score_Zc_Zd_random
        scores['Zc_pert_Zd'] = score_Zc_pert_Zd
        scores['Zc_Zd_pert'] = score_Zc_Zd_pert

    return scores, losses, mean_proba_per_class, std_proba_per_class, mean_proba_per_class_noised, \
           std_proba_per_class_noised


def compute_one_bit_random_score(prediction_zc_random_zd, labels, stats_prediction, pred_noised):
    # compute proba prediciton:
    pred_proba = prediction_zc_random_zd.exp()  # to get a probability (between 0 and 1)
    pred_proba_class = [pred_proba[i][lab].item() for i, lab in
                        enumerate(labels)]  # for each data get the pred score for class i of labels
    # save proba et label associated:
    stats_prediction['label'].extend(labels.cpu().detach().numpy())
    stats_prediction['proba'].extend(pred_proba_class)

    # pred for z_struct_noised: shape: (nb_repeat, nb_bits_noised, batch_size, nb_class)
    for i in range(pred_noised.shape[1]):  # for each z noised (one bit by one bit)
        pred_proba_noised = np.exp(pred_noised[:, i])  # get proba (between 0 and 1)
        pred_proba_class_noised = [pred_proba_noised[:, j, lab] for j, lab in enumerate(labels)]
        # add in 'proba_noised', pred for each class: list of len: nb_labels
        stats_prediction['proba_noised'][i].extend(pred_proba_class_noised)  # shape: (struct_dim, nb_data, nb_repeat)

    return stats_prediction


def compute_proba(stats_prediction, nb_class, pred_noised):
    mean_proba_per_class = []
    std_proba_per_class = []
    mean_proba_per_class_noised = []
    std_proba_per_class_noised = []

    # sort all sample_scores prediction per label
    for i in range(nb_class):
        # get average score prediction of predict[i] for label i
        all_predicted_scores = np.array(stats_prediction['proba'])[
            np.concatenate(np.argwhere(np.array(stats_prediction['label']) == i)).ravel()]
        mean_proba_per_class.append(sum(all_predicted_scores) / len(all_predicted_scores))
        std_proba_per_class.append(np.std(all_predicted_scores, axis=0))

    # for z_noised:
    # for each bit noised:
    for j in range(pred_noised.shape[1]):
        mean = []
        std = []
        for i in range(nb_class):
            # get average score prediction of predict[i] for label i
            all_predicted_scores_noised = np.array(stats_prediction['proba_noised'][j])[
                np.concatenate(np.argwhere(np.array(stats_prediction['label']) == i)).ravel()]
            all_predicted_scores_noised_avarage = np.mean(all_predicted_scores_noised, axis=0)
            mean.append(np.mean(all_predicted_scores_noised_avarage, axis=0))
            std.append(np.std(all_predicted_scores_noised_avarage, axis=0))
        mean_proba_per_class_noised.append(mean)
        std_proba_per_class_noised.append(std)

    return mean_proba_per_class, std_proba_per_class, mean_proba_per_class_noised, std_proba_per_class_noised


def compute_scores_pred(prediction, labels):
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


def compute_Zc_pert_Zd_scores(prediction_pert, labels):
    """
    return nb of correct prediction for the current batch
    :param prediction_pert:
    :param labels:
    :return:
    """
    scores_pert = []

    for i in range(len(prediction_pert)):
        pred_iter = prediction_pert[i]
        pred_iter = pred_iter.argmax(dim=1, keepdim=True)
        correct_iter = pred_iter.eq(labels.view_as(pred_iter)).sum().item()
        scores_iter = correct_iter
        scores_pert.append(float(scores_iter))
    return scores_pert


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0

    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)

    return total_kld


def _kl_multiple_discrete_loss(alphas):
    """
    Calculates the KL divergence between a set of categorical distributions
    and a set of uniform categorical distributions.
    Parameters
    ----------
    alphas : list
        List of the alpha parameters of a categorical (or gumbel-softmax)
        distribution. For example, if the categorical atent distribution of
        the model has dimensions [2, 5, 10] then alphas will contain 3
        torch.Tensor instances with the parameters for each of
        the distributions. Each of these will have shape (N, D).
    """
    # Calculate kl losses for each discrete latent
    kl_losses = [_kl_discrete_loss(alpha) for alpha in alphas]

    # Total loss is sum of kl loss for each discrete latent
    kl_loss = torch.sum(torch.cat(kl_losses))
    return float(kl_loss)


def _kl_discrete_loss(alpha):
    """
    Calculates the KL divergence between a categorical distribution and a
    uniform categorical distribution.
    Parameters
    ----------
    alpha : torch.Tensor
        Parameters of the categorical or gumbel-softmax distribution.
        Shape (N, D)
    """
    disc_dim = int(alpha.size()[-1])
    log_dim = torch.Tensor([np.log(disc_dim)])
    if torch.cuda.is_available():
        log_dim = log_dim.cuda()
    # Calculate negative entropy of each row
    neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
    # Take mean of negative entropy across batch
    mean_neg_entropy = torch.mean(neg_entropy, dim=0)
    # KL loss of alpha with uniform categorical variable
    kl_loss = log_dim + mean_neg_entropy
    return kl_loss
