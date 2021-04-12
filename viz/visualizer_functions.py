from models.encoder_struct import Encoder_struct
from models.vae_var import VAE_var
from models.VAE import VAE

import matplotlib.pyplot as plt
import torch
import csv
import os


def get_checkpoints(net, path, model_name, device):
    """
    Load saved weights after model training.
    :param net:
    :param path:
    :param model_name:
    :param device:
    :return:
    """
    file_path = os.path.join(path, model_name, 'last')
    checkpoint = torch.load(file_path, map_location=torch.device(device))
    net.load_state_dict(checkpoint['net'])
    nb_iter = checkpoint['iter']
    nb_epochs = checkpoint['epochs']

    return net, nb_iter, nb_epochs


def str2bool(string):
    """
    Convert string to bool.
    :param string:
    :return:
    """
    return string.lower() in ("yes", "true", "t", "1")


def model_load(model_name, model_path=None, device=None, verbose=True):
    """
    We load model parameters from experimentation name. With this parameters we create model architecture and load
    associate weighs.
    :param verbose:
    :param model_path:
    :param device:
    :param model_name: model name to load
    :return:
    """

    # load csv parameters:
    exp_csv_name = '../args_parser/' + model_name + '.csv'
    with open(exp_csv_name, newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            parameters_dict = row

    # model:
    model_name = parameters_dict['exp_name']

    # Parameters:
    stride_size = int(parameters_dict['stride_size'])

    # losses:

    # VAE parameters:
    is_VAE = str2bool(parameters_dict['is_VAE'])
    lambda_BCE = float(parameters_dict['lambda_BCE'])
    beta = float(parameters_dict['beta'])
    z_struct_size = int(parameters_dict['z_struct_size'])

    # Encoder struct parameters:
    big_kernel_size = int(parameters_dict['big_kernel_size'])
    BK_in_first_layer = str2bool(parameters_dict['BK_in_first_layer'])
    BK_in_second_layer = str2bool(parameters_dict['BK_in_second_layer'])
    BK_in_third_layer = str2bool(parameters_dict['BK_in_third_layer'])
    two_conv_layer = str2bool(parameters_dict['two_conv_layer'])
    three_conv_layer = str2bool(parameters_dict['three_conv_layer'])

    # VAE var parameters:
    is_VAE_var = str2bool(parameters_dict['is_VAE_var'])
    z_var_size = int(parameters_dict['z_var_size'])
    var_second_cnn_block = str2bool(parameters_dict['var_second_cnn_block'])
    var_third_cnn_block = str2bool(parameters_dict['var_third_cnn_block'])

    use_structural_encoder = str2bool(parameters_dict['use_structural_encoder'])

    # encoder struct:
    is_encoder_struct = str2bool(parameters_dict['is_encoder_struct'])
    kernel_size_1 = int(parameters_dict['kernel_size_1'])
    kernel_size_2 = int(parameters_dict['kernel_size_2'])
    kernel_size_3 = int(parameters_dict['kernel_size_3'])
    binary_z = str2bool(parameters_dict['binary_z'])
    binary_first_conv = str2bool(parameters_dict['binary_first_conv'])
    binary_second_conv = str2bool(parameters_dict['binary_second_conv'])
    binary_third_conv = str2bool(parameters_dict['binary_third_conv'])

    # other default parameters:
    hidden_filters_1 = int(parameters_dict['hidden_filters_layer1'])
    hidden_filters_2 = int(parameters_dict['hidden_filters_layer2'])
    hidden_filters_3 = int(parameters_dict['hidden_filters_layer3'])
    other_architecture = str2bool(parameters_dict['other_architecture'])

    if is_VAE:
        model_type = 'VAE'
        net = VAE(z_var_size=z_var_size,
                  var_second_cnn_block=var_second_cnn_block,
                  var_third_cnn_block=var_third_cnn_block,
                  other_architecture=other_architecture,
                  z_struct_size=z_struct_size,
                  big_kernel_size=big_kernel_size,
                  stride_size=stride_size,
                  kernel_size_1=kernel_size_1,
                  kernel_size_2=kernel_size_2,
                  kernel_size_3=kernel_size_3,
                  hidden_filters_1=hidden_filters_1,
                  hidden_filters_2=hidden_filters_2,
                  hidden_filters_3=hidden_filters_3,
                  BK_in_first_layer=BK_in_first_layer,
                  BK_in_second_layer=BK_in_second_layer,
                  BK_in_third_layer=BK_in_third_layer,
                  two_conv_layer=two_conv_layer,
                  three_conv_layer=three_conv_layer,
                  Binary_z=binary_z,
                  binary_first_conv=binary_first_conv,
                  binary_second_conv=binary_second_conv,
                  binary_third_conv=binary_third_conv)
    elif is_encoder_struct:
        model_type = 'Encoder struct'
        net = Encoder_struct(z_struct_size=z_struct_size,
                             big_kernel_size=big_kernel_size,
                             stride_size=stride_size,
                             kernel_size_1=kernel_size_1,
                             kernel_size_2=kernel_size_2,
                             kernel_size_3=kernel_size_3,
                             hidden_filters_1=hidden_filters_1,
                             hidden_filters_2=hidden_filters_2,
                             hidden_filters_3=hidden_filters_3,
                             BK_in_first_layer=BK_in_first_layer,
                             BK_in_second_layer=BK_in_second_layer,
                             BK_in_third_layer=BK_in_third_layer,
                             two_conv_layer=two_conv_layer,
                             three_conv_layer=three_conv_layer,
                             Binary_z=binary_z,
                             binary_first_conv=binary_first_conv,
                             binary_second_conv=binary_second_conv,
                             binary_third_conv=binary_third_conv)
    elif is_VAE_var:
        model_type = 'VAE vae'
        net = VAE_var(z_var_size=z_var_size,
                      var_second_cnn_block=var_second_cnn_block,
                      var_third_cnn_block=var_third_cnn_block,
                      other_architecture=other_architecture)

    net, _, nb_epochs = get_checkpoints(net, model_path, model_name, device)
    net = net.to(device)
    if verbose:
        print('{}: _________--------------{}-------------________________, nb_epochs: {}'.format(model_type,
                                                                                                 model_name,
                                                                                                 nb_epochs))
        print(net)
    return net, z_struct_size, z_var_size


def get_checkpoints_scores_VAE(model_name, device):
    """
    Load scores save during training.
    :param model_name:
    :param device:
    :return:
    """
    path_scores = '../checkpoint_scores_CNN/'
    file_path = os.path.join(path_scores, model_name, 'last')
    checkpoints_scores = torch.load(file_path, map_location=torch.device(device))

    global_iter = checkpoints_scores['iter']
    epochs = checkpoints_scores['epochs']
    train_score = checkpoints_scores['Total_loss_train']
    test_score = checkpoints_scores['Total_loss_test']
    BCE_train = checkpoints_scores['BCE_train']
    BCE_test = checkpoints_scores['BCE_test']
    KLD_train = checkpoints_scores['KLD_train']
    KLD_test = checkpoints_scores['KLD_test']

    return global_iter, epochs, train_score, test_score, BCE_train, BCE_test, KLD_train, KLD_test


def plot_loss(model_name, device, save=True):
    """
    Load and plot loss values saved during training.
    :param model_name:
    :param device:
    :param save:
    :return:
    """
    global_iter, epochs, train_score, test_score, BCE_train, BCE_test, KLD_train, \
    KLD_test = get_checkpoints_scores_VAE(model_name, device)

    fig, ax = plt.subplots(figsize=(15, 10), facecolor='w', edgecolor='k')

    ax.set(xlabel='nb_iter', ylabel='loss',
           title=('Losses: {}'.format(model_name)))

    ax.plot(epochs, train_score, label='Total train')
    ax.plot(epochs, test_score, label='Total test')
    ax.plot(epochs, BCE_train, label='BCE train')
    ax.plot(epochs, BCE_test, label='BCE test')
    ax.plot(epochs, KLD_train, label='KLD train')
    ax.plot(epochs, KLD_test, label='KLD test')

    ax.legend(loc=1)
    plt.show()

    if save:
        fig.savefig("../fig_results/losses/fig_losses_Test_Mnist_VAE_" + model_name + ".png")

    return


def real_distribution_model(net, model_name, z_var_size, loader, train_test, plot_gaussian=False,
                            save=False, VAE_struct=False, is_vae_var=False):
    """
    Compute real distribution for loader set and specific net.
    Extract all data in loader and compute average mean and std value for each component of z_var, and save percentage
    of zero for each component of z_struct.
    :param net:
    :param model_name:
    :param z_var_size:
    :param loader:
    :param train_test:
    :param plot_gaussian:
    :param save:
    :param VAE_struct:
    :param is_vae_var:
    :return:
    """

    path = 'Other_results/real_distribution/gaussian_real_distribution_' + model_name + '_' + train_test + '_mu_var.npy'

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
                    _, latent_representation = net(data)
                    mu_var_iter = latent_representation['mu']
                    sigma_var_iter = latent_representation['log_var']
                else:
                    _, z_struct, z_var, z_var_sample, latent_representation, z = net(data)

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
            zeros_proportion = (np.count_nonzero(z_struct_distribution == 0, axis=0) * 100.) / len(z_struct_distribution)
            # mu_struct = torch.mean(z_struct_distribution, axis=0)
            # sigma_struct = torch.std(z_struct_distribution, axis=0)
        else:
            mu_struct = 0
            sigma_struct = 0

        np.save('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
                '_mu_var.npy', mu_var)
        np.save('Other_results/real_distribution/gaussian_real_distribution_' + expe_name + '_' + train_test +
                'sigma_var.npy', sigma_var)
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

