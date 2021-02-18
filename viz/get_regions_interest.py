import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def get_regions_interest_fct(regions, labels, activation, activations_normalized, details=False, best=True, worst=False,
                             viz_mean_img=True, viz_grid=True, percentage=None, list_filter=None, nrow=8,
                             plot_histogram=False, save_mean_regions=False, path_save_mean_region=None, bins=20):
    """
  get regions of interest
  """
    nb_filter = activation.shape[1]

    if save_mean_regions:
        assert path_save_mean_region is not None, 'you must choice path_save_mean_region if you want save mean regions'

    if best == False and worst == False:
        assert percentage is not None, "if don't choice best or worst value, you didn't choice a percentage value"
    if best == True and worst == True:
        raise TypeError('choice only one value at True between best an worst')

    # consider only regions of all images of list_filter or all filters:
    if list_filter == None:
        if details:
            print('Interest of all filters')
        regions_interest_filter = regions
        activations_values_interest = activation
        activations_values_interest_normalized = activations_normalized
        list_filter = range(nb_filter)
    else:
        # assert max(list_filter) < nb_filter and min(list_filter) >= 0, 'filter choisen out of range'
        if details:
            print('Interest of filters:', list_filter)
        regions_interest_filter = get_index_filter_interest(regions, list_filter)
        activations_values_interest = activation[:, list_filter]
        activations_values_interest_normalized = activations_normalized[:, list_filter]
        nb_filter = len(list_filter)
        list_filter = list_filter

    # consider a percent of best or worst activations:
    if percentage is None:
        if details:
            print('Consider all image regions')
        nb_regions = activation.shape[0]
        selected_regions = regions_interest_filter
        activation_values = activation[:, list_filter]
        activation_values_normalized = activations_normalized[:, list_filter]
    else:
        assert 100 >= percentage >= 0, 'percentage value must be in 0 and 100'
        n = int((len(activation) * percentage) / 100)
        if details:
            print('Consider {}% image regions = {} images'.format(percentage, n))
        nb_regions = n
        selected_regions, activation_values = get_n_first_regions_index(best,
                                                                        worst,
                                                                        n,
                                                                        activations_values_interest,
                                                                        nb_filter,
                                                                        regions_interest_filter)
        selected_regions_normalized, activation_values_normalized = get_n_first_regions_index(best,
                                                                                              worst,
                                                                                              n,
                                                                                              activations_values_interest_normalized,
                                                                                              nb_filter,
                                                                                              regions_interest_filter)

    # get labels for plot histogram:
    labels_all, labels_selected, labels_selected_normalized, labels_selected_all, labels_selected_all_normalized = get_labels_histogram(
        labels, activation, activations_normalized, list_filter=list_filter, best=best, worst=worst,
        percentage=percentage, plot=False, return_values=True)

    # visualization: one mean image or grid image:
    if viz_mean_img:
        nb_image = 1
        means_images = []
        if details:
            print('mean image:')
        if plot_histogram:
            print('histogram of images repartition:')
            plt.hist(labels_all, bins=bins)
            plt.show()

        if details:
            for i, ind_filter in enumerate(list_filter):

                print('mean regions of {} regions more={} or worst={} active for filter number: {} :'.format(n,
                                                                                                             best,
                                                                                                             worst,
                                                                                                             ind_filter))
                mean_img = np.mean(selected_regions[i], 0)
                viz_regions(nb_image, mean_img, nrow)
                plt.show()

                print('normalized region:')
                mean_img_normalized = np.mean(selected_regions_normalized[i], 0)
                viz_regions(nb_image, mean_img_normalized, nrow)
                plt.show()

                if plot_histogram:
                    print('histogram for filter {}:'.format(ind_filter))
                    x = labels_selected[i]
                    y = labels_selected_normalized[i]
                    plt.hist([x, y], bins, alpha=0.75, label=['regular', 'normalized'])
                    plt.show()

            if plot_histogram:
                print('Histogram for all filters combined:')
                x = labels_selected_all
                y = labels_selected_all_normalized
                plt.hist([x, y], bins, alpha=0.75, label=['regular', 'normalized'])
                plt.show()

        else:
            to_visualize = []
            to_visualize_normalized = []
            for i, ind_filter in enumerate(list_filter):
                mean_img = np.mean(selected_regions[i], 0)
                mean_img_normalized = np.mean(selected_regions_normalized[i], 0)
                to_visualize.append(mean_img)
                to_visualize_normalized.append(mean_img_normalized)
            print('mean regions of {} regions who activate most filters:'.format(n))
            if save_mean_regions:
                path_save = path_save_mean_region + '.png'
                viz_regions(len(to_visualize), to_visualize, nrow=nrow, save=True, path_save=path_save)
            else:
                viz_regions(len(to_visualize), to_visualize, 10)
            plt.show()

            print('With normalized activations:')
            if save_mean_regions:
                path_save = path_save_mean_region + '_normalized.png'
                viz_regions(len(to_visualize_normalized), to_visualize_normalized, nrow=nrow, save=True,
                            path_save=path_save)
            else:
                viz_regions(len(to_visualize_normalized), to_visualize_normalized, nrow=nrow)
            plt.show()

    if viz_grid:
        nb_image = nb_regions
        print('grid image')
        for i, ind_filter in enumerate(list_filter):
            region_to_print = []
            print('grid regions of {} regions more={} or worst={} active for filter number: {} :'.format(n, best, worst,
                                                                                                         ind_filter))
            for j in range(nb_regions):
                region_to_print.append(selected_regions[i][j])
            viz_regions(nb_image, region_to_print, nrow)
            plt.show()

            print('normalized regions:')
            region_to_print_normalized = []
            for j in range(nb_regions):
                region_to_print_normalized.append(selected_regions_normalized[i][j])
            viz_regions(nb_image, region_to_print_normalized, nrow)
            plt.show()

    return selected_regions, activation_values, activation_values_normalized


def get_index_filter_interest(regions, list_filter):
    """
  extract only regions of the filter interest
  """
    return regions[:, list_filter]


def get_n_first_regions_index(best, worst, n, activation, nb_filter, regions):
    """
  select only regions that we want with associated label
  """
    regions_selected = []
    activation_values = []
    if best:
        for i in range(nb_filter):
            ind_filter = (-activation[:, i]).argsort()[:n]
            regions_selected.append(regions[ind_filter, i])
            activation_values.append(activation[ind_filter, i])
        return regions_selected, activation_values

    elif worst:
        for i in range(nb_filter):
            ind_filter = activation[:, i].argsort()[:n]
            regions_selected.append(regions[ind_filter, i])
            activation_values.append(activation[ind_filter, i])
        return regions_selected, activation_values

    else:
        print('choice worst or best with bool True or False')


def viz_regions(nb_image, regions, nrow, save=False, path_save=None):
    """
  visualize region of interest
  """
    regions = torch.tensor(regions)
    regions = regions.reshape((nb_image, 1, regions.shape[-2], regions.shape[-1]))
    visTensor(regions, ch=0, allkernels=False, nrow=nrow, save=save, path_save=path_save)
    plt.ioff()
    plt.show()


def get_labels_histogram(labels, activation, activations_normalized, list_filter=None, best=True, worst=False,
                         percentage=None, plot=True, return_values=False, n_bins=20):
    """
    labels_all: contain all labels to plot histo
    labels_selected: contains labels of regions image who activated the n first (or last) filters sort by filter
    labels_selected_normalized: idem for activation normalized
    labels_selected_all: contains labels of regions image who activated the n first (or last) filters for all filter combined histo
    labels_selected_all_normalized: idem for activation normalized
    """
    if best == False and worst == False:
        assert percentage != None, "if don't choice best or worst value, you didn't choice a percentage value"
    if best == True and worst == True:
        raise TypeError('choice only one value at True between best an worst')

    if list_filter is None:
        activations_values_interest = activation
        activations_values_interest_normalized = activations_normalized
        list_filter = range(activations_values_interest.shape[1])
    else:
        activations_values_interest = activation[:, list_filter]
        activations_values_interest_normalized = activations_normalized[:, list_filter]

    # plot histogram for all images:
    labels_all = labels
    if plot:
        print('Histogram of labels for all images:')
        plt.hist(labels_all, bins=n_bins)
        plt.show()

    # plot histogram for n first activations:
    if percentage is not None:
        assert 100 >= percentage >= 0, 'percentage value must be in 0 and 100'
        n = int((len(activation) * percentage) / 100)
        print('Consider {}% image regions = {} images'.format(percentage, n))
        labels_selected = []
        labels_selected_normalized = []
        if best:
            for i, filt in enumerate(list_filter):
                ind_filter = (-activations_values_interest[:, i]).argsort()[:n]
                labels_selected.append(labels[ind_filter])
                if plot:
                    print('Histogram for filter {} with the {} first regions who actived it:'.format(filt, n))
                    plt.hist(labels_selected[i], bins=n_bins)
                    plt.show()
                ind_filter_normalized = (-activations_values_interest_normalized[:, i]).argsort()[:n]
                labels_selected_normalized.append(labels[ind_filter_normalized])
                if plot:
                    print('Histo for activations normalized:')
                    plt.hist(labels_selected_normalized[i], bins=n_bins)
                    plt.show()
        elif worst:
            for i, filt in enumerate(list_filter):
                ind_filter = activations_values_interest[:, i].argsort()[:n]
                labels_selected.append(labels[ind_filter])
                if plot:
                    print('Histogram for filter {} with the {} first regions who actived it:'.format(filt, n))
                    plt.hist(labels_selected[i], bins=n_bins)
                    plt.show()

                ind_filter_normalized = activations_values_interest_normalized[:, i].argsort()[:n]
                labels_selected_normalized.append(labels[ind_filter_normalized])
                if plot:
                    print('Histo for activations normalized:')
                    plt.hist(labels_selected_normalized[i], bins=n_bins)
                    plt.show()
        else:
            print('choice worst or best with bool True or False')

        # plot global histo: with all filters of list_filter
        labels_selected_all = [val for sublist in labels_selected for val in sublist]
        if plot:
            print('Histogram for all filters in list_filter with the {} first regions for all filters combined:'.format(
                n))
            plt.hist(labels_selected_all, bins=n_bins)
            plt.show()

        labels_selected_all_normalized = [val for sublist in labels_selected_normalized for val in sublist]
        if plot:
            print('Histo for activations normalized:')
            plt.hist(labels_selected_all_normalized, bins=n_bins)
            plt.show()

    if return_values:
        return labels_all, labels_selected, labels_selected_normalized, labels_selected_all, labels_selected_all_normalized


def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1, save=False, path_save=None):
    n, c, w, h = tensor.shape

    if save:
        assert path_save is not None, 'you must choice path if you want save figure'
    if allkernels:
        tensor = tensor.view(n * c, -1, w, h)
    elif c != 3:
        tensor = tensor[:, ch, :, :].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))
    grid = make_grid(tensor, nrow=nrow, normalize=True, padding=padding, pad_value=1)
    plt.figure(figsize=(nrow, rows))
    fig = plt.gcf()
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    if save:
        fig.savefig(path_save, dpi=100)
        print('image saved')
