import numpy as np
from numpy import linalg as LA
import cv2
import math
import matplotlib.pyplot as plt
import torch


def get_all_regions_max(images, activations, len_img_h, len_img_w, net):
    print('begin extraction regions')

    # declare dict:
    region_final = {}
    activation_final = {}
    activation_final_normalized = {}

    # parameters for receptive field:
    # https://rubikscode.net/2020/05/18/receptive-field-arithmetic-for-convolutional-neural-networks/
    jump = [1]
    receptive_field_size = [1]
    i = 0
    for name, fm in activations.item().items():
        i += 1
        layer_num = int(name.split('.')[-1])
        stride_size = net.net[layer_num].stride[0]
        filter_size = net.net[layer_num].kernel_size[0]

        jump.append(jump[-1] * stride_size)  # represent the cumulative stride
        receptive_field_size.append(receptive_field_size[-1] + (filter_size - 1) * jump[-2])  # size of receptive field

    for name, fm in activations.item().items():
        layer_num = int(name.split('.')[-1])
        filter_size = net.net[layer_num].kernel_size[0]
        padding = net.net[layer_num].padding[0]
        stride_size = net.net[layer_num].stride[0]

        # print('layer {}: stride={}, filter size={}, padding={} and fm of shape={}'.format(name,
        #                                                                                   stride_size,
        #                                                                                   filter_size,
        #                                                                                   padding,
        #                                                                                   fm.size()))
        if 'net.0' in name:
            n_layer = 1
            regions_layer = np.zeros((fm.shape[0], fm.shape[1], receptive_field_size[1], receptive_field_size[1]))
            activation_layer = np.zeros((fm.shape[0], fm.shape[1]))
            activation_layer_normalized = np.zeros((fm.shape[0], fm.shape[1]))
        if 'net.3' in name:
            n_layer = 2
            regions_layer = np.zeros((fm.shape[0], fm.shape[1], receptive_field_size[2], receptive_field_size[2]))
            activation_layer = np.zeros((fm.shape[0], fm.shape[1]))
            activation_layer_normalized = np.zeros((fm.shape[0], fm.shape[1]))
        if 'net.6' in name:
            n_layer = 3
            regions_layer = np.zeros((fm.shape[0], fm.shape[1], receptive_field_size[3], receptive_field_size[3]))
            activation_layer = np.zeros((fm.shape[0], fm.shape[1]))
            activation_layer_normalized = np.zeros((fm.shape[0], fm.shape[1]))
        if 'net.9' in name:
            n_layer = 4
            regions_layer = np.zeros((fm.shape[0], fm.shape[1], receptive_field_size[4], receptive_field_size[4]))
            activation_layer = np.zeros((fm.shape[0], fm.shape[1]))
            activation_layer_normalized = np.zeros((fm.shape[0], fm.shape[1]))
        for j in range(fm.shape[0]):
            # for each images of my loader:
            print('treating image n {}/{}, for layer: {}'.format(j, fm.shape[0], name))
            im = images[j].unsqueeze(0).numpy().squeeze()  # im shape: (len_img_h, len_img_w)
            # plt.imshow(im, cmap='gray')
            # plt.show()
            # image i of batch batch: numpy array: (len_img_h, len_img_w)
            if 'net.0' in name:
                regions_im_j = np.zeros((fm.shape[1], receptive_field_size[1], receptive_field_size[1]))
                # initialise empty list of regions for batch
                activation_im_j = np.zeros((fm.shape[1]))
                activation_im_j_normalized = np.zeros((fm.shape[1]))
            if 'net.3' in name:
                regions_im_j = np.zeros((fm.shape[1], receptive_field_size[2], receptive_field_size[2]))
                activation_im_j = np.zeros((fm.shape[1]))
                activation_im_j_normalized = np.zeros((fm.shape[1]))
            if 'net.6' in name:
                regions_im_j = np.zeros((fm.shape[1], receptive_field_size[3], receptive_field_size[3]))
                activation_im_j = np.zeros((fm.shape[1]))
                activation_im_j_normalized = np.zeros((fm.shape[1]))
            if 'net.9' in name:
                regions_im_j = np.zeros((fm.shape[1], receptive_field_size[4], receptive_field_size[4]))
                activation_im_j = np.zeros((fm.shape[1]))
                activation_im_j_normalized = np.zeros((fm.shape[1]))
            for i in range(fm.shape[1]):  # for each fm in image j
                # act_max = torch.max(fm[j][i].abs())
                act_max = fm[j][i].max()

                ind_x = int((np.where(fm[j][i] == act_max)[0])[0])  # get index (x,y) of act_max
                ind_y = int((np.where(fm[j][i] == act_max)[1])[0])

                """
                start_x = [ind_x]
                start_y = [ind_y]
            
                # get index start region:
                for n in range(1, n_layer + 1):
                    start_x.append(math.ceil(start_x[n - 1] - ((filter_size - 1 / 2) - padding) * jump[n]))
                    start_y.append(math.ceil(start_y[n - 1] - ((filter_size - 1 / 2) - padding) * jump[n]))
                start_x = np.array(start_x)
                start_y = np.array(start_y)
                start_x[start_x < 0] = 0
                start_y[start_y < 0] = 0

                # parameters for extract region:
                ind_x = start_x[-1]
                ind_y = start_y[-1]

                if ind_x < 0:
                    reduice_region_col_size = ind_x
                    ind_x = 0
                else:
                    reduice_region_col_size = 0
                if ind_y < 0:
                    reduice_region_raw_size = ind_y
                    ind_y = 0
                else:
                    reduice_region_raw_size = 0

                begin_col = ind_x
                end_col = ind_x + receptive_field_size[n_layer] + reduice_region_col_size
                begin_raw = ind_y
                end_raw = ind_y + receptive_field_size[n_layer] + reduice_region_raw_size

                if end_col > len_img_w:
                    end_col = len_img_w
                if end_raw > len_img_h:
                    end_raw = len_img_h

                if end_col == 0:
                    end_col = 2
                if end_raw == 0:
                    end_raw = 2
                if begin_col == len_img_w:
                    begin_col = begin_col - 1
                if begin_raw == len_img_h:
                    begin_raw = begin_raw - 1

                region = im[begin_col:end_col, begin_raw:end_raw]
                
                if region.shape != (receptive_field_size[n_layer], receptive_field_size[n_layer]):
                    print(region.shape)
                    if type(region) == np.ndarray:
                        region = cv2.resize(region,
                                            (receptive_field_size[n_layer], receptive_field_size[n_layer]),
                                            interpolation=cv2.INTER_AREA)
                    else:
                        region = cv2.resize(region.numpy(),
                                            (receptive_field_size[n_layer], receptive_field_size[n_layer]),
                                            interpolation=cv2.INTER_AREA)
                                            
                """

                if 'net.0' in name:
                    region, begin_col, end_col, begin_raw, end_raw = get_region_layer1(receptive_field_size[n_layer],
                                                                                       im,
                                                                                       ind_x,
                                                                                       ind_y,
                                                                                       stride_size,
                                                                                       padding,
                                                                                       len_img_h,
                                                                                       len_img_w,
                                                                                       return_all=True)
                if 'net.3' in name:
                    region, begin_col, end_col, begin_raw, end_raw = get_region_layer2(receptive_field_size[n_layer],
                                                                                       im,
                                                                                       ind_x,
                                                                                       ind_y,
                                                                                       stride_size,
                                                                                       padding,
                                                                                       len_img_h,
                                                                                       len_img_w,
                                                                                       return_all=True)
                if 'net.6' in name:
                    region, begin_col, end_col, begin_raw, end_raw = get_region_layer3(receptive_field_size[n_layer],
                                                                                       im,
                                                                                       ind_x,
                                                                                       ind_y,
                                                                                       stride_size,
                                                                                       padding,
                                                                                       len_img_h,
                                                                                       len_img_w,
                                                                                       return_all=True)
                if 'net.9' in name:
                    region, begin_col, end_col, begin_raw, end_raw = get_region_layer4(receptive_field_size[n_layer],
                                                                                       im,
                                                                                       ind_x,
                                                                                       ind_y,
                                                                                       stride_size,
                                                                                       padding,
                                                                                       len_img_h,
                                                                                       len_img_w,
                                                                                       return_all=True)
                regions_im_j[i] = region
                activation_im_j[i] = act_max.detach().numpy()
                norme = LA.norm(region, 1)
                if norme == 0.0:
                    print('norm null for filter: {}, image: {} with activation:{}, index({},{})'.format(i, j, act_max,
                                                                                                        ind_x, ind_y))
                    activation_im_j_normalized[i] = act_max.detach().numpy()
                else:
                    activation_im_j_normalized[i] = act_max.detach().numpy() / norme
            regions_layer[j] = regions_im_j
            activation_layer[j] = activation_im_j
            activation_layer_normalized[j] = activation_im_j_normalized
        region_final[name] = regions_layer
        activation_final[name] = activation_layer
        activation_final_normalized[name] = activation_layer_normalized
    return region_final, activation_final, activation_final_normalized


def get_region_layer1(region_shape, image, ind_x, ind_y, stride, padding, len_img_h, len_img_w, return_all=False):

    # determine pixel high left of region of interest:
    index_col_hl = (ind_x * stride) - padding
    index_raw_hl = (ind_y * stride) - padding

    if index_col_hl < 0:
        reduice_region_col_size = index_col_hl
        index_col_hl = 0
    else:
        reduice_region_col_size = 0
    if index_raw_hl < 0:
        reduice_region_raw_size = index_raw_hl
        index_raw_hl = 0
    else:
        reduice_region_raw_size = 0

    begin_col = index_col_hl
    end_col = index_col_hl + region_shape + reduice_region_col_size
    begin_raw = index_raw_hl
    end_raw = index_raw_hl + region_shape + reduice_region_raw_size

    if end_col > len_img_w:
        end_col = len_img_w
    if end_raw > len_img_h:
        end_raw = len_img_h

    region = image[begin_col:end_col, begin_raw:end_raw]
    if region.shape != (region_shape, region_shape):
        print('reshape region', region.shape)
        if type(region) == np.ndarray:
            region = cv2.resize(region, (region_shape, region_shape), interpolation=cv2.INTER_AREA)
        else:
            region = cv2.resize(region.numpy(), (region_shape, region_shape), interpolation=cv2.INTER_AREA)

    if return_all:
        return region, begin_col, end_col, begin_raw, end_raw
    else:
        return region


def get_region_layer2(region_shape, image, ind_x, ind_y, stride, padding, len_img_h, len_img_w, return_all=False):

    # determine pixel high left of region of interest:
    index_col_hl = (ind_x * stride) - padding
    index_raw_hl = (ind_y * stride) - padding

    if index_col_hl < 0:
        index_col_hl = 0
    if index_raw_hl < 0:
        index_raw_hl = 0

    index_col_hl_2 = (index_col_hl * stride) - padding
    index_raw_hl_2 = (index_raw_hl * stride) - padding

    if index_col_hl_2 < 0:
        reduice_region_col_size = index_col_hl_2
        index_col_hl_2 = 0
    else:
        reduice_region_col_size = 0
    if index_raw_hl_2 < 0:
        reduice_region_raw_size = index_raw_hl_2
        index_raw_hl_2 = 0
    else:
        reduice_region_raw_size = 0

    begin_col = index_col_hl_2
    end_col = index_col_hl_2 + region_shape + reduice_region_col_size
    begin_raw = index_raw_hl_2
    end_raw = index_raw_hl_2 + region_shape + reduice_region_raw_size

    if end_col > len_img_w:
        end_col = len_img_w
    if end_raw > len_img_h:
        end_raw = len_img_h

    region = image[begin_col:end_col, begin_raw:end_raw]
    if region.shape != (region_shape, region_shape):
        print('reshape region', region.shape)
        if type(region) == np.ndarray:
            region = cv2.resize(region, (region_shape, region_shape), interpolation=cv2.INTER_AREA)
        else:
            region = cv2.resize(region.numpy(), (region_shape, region_shape), interpolation=cv2.INTER_AREA)

    if return_all:
        return region, begin_col, end_col, begin_raw, end_raw
    else:
        return region


def get_region_layer3(region_shape, image, ind_x, ind_y, stride, padding, len_img_h, len_img_w, return_all=False):

    # determine pixel high left of region of interest:
    index_col_hl = (ind_x * stride) - padding
    index_raw_hl = (ind_y * stride) - padding

    if index_col_hl < 0:
        index_col_hl = 0
    if index_raw_hl < 0:
        index_raw_hl = 0

    index_col_hl_2 = (index_col_hl * stride) - padding
    index_raw_hl_2 = (index_raw_hl * stride) - padding

    if index_col_hl_2 < 0:
        index_col_hl_2 = 0
    if index_raw_hl_2 < 0:
        index_raw_hl_2 = 0

    index_col_hl_3 = (index_col_hl_2 * stride) - padding
    index_raw_hl_3 = (index_raw_hl_2 * stride) - padding

    if index_col_hl_3 < 0:
        reduice_region_col_size = index_col_hl_3
        index_col_hl_3 = 0
    else:
        reduice_region_col_size = 0
    if index_raw_hl_3 < 0:
        reduice_region_raw_size = index_raw_hl_3
        index_raw_hl_3 = 0
    else:
        reduice_region_raw_size = 0

    begin_col = index_col_hl_3
    end_col = index_col_hl_3 + region_shape + reduice_region_col_size
    begin_raw = index_raw_hl_3
    end_raw = index_raw_hl_3 + region_shape + reduice_region_raw_size

    if end_col > len_img_w:
        end_col = len_img_w
    if end_raw > len_img_h:
        end_raw = len_img_h

    region = image[begin_col:end_col, begin_raw:end_raw]
    if region.shape != (region_shape, region_shape):
        print('reshape region', region.shape)
        if type(region) == np.ndarray:
            region = cv2.resize(region, (region_shape, region_shape), interpolation=cv2.INTER_AREA)
        else:
            region = cv2.resize(region.numpy(), (region_shape, region_shape), interpolation=cv2.INTER_AREA)

    if return_all:
        return region, begin_col, end_col, begin_raw, end_raw
    else:
        return region


def get_region_layer4(region_shape, image, ind_x, ind_y, stride, padding, len_img_h, len_img_w, return_all=False):

    # determine pixel high left of region of interest:
    index_col_hl = (ind_x * stride) - padding
    index_raw_hl = (ind_y * stride) - padding

    if index_col_hl < 0:
        index_col_hl = 0
    if index_raw_hl < 0:
        index_raw_hl = 0

    index_col_hl_2 = (index_col_hl * stride) - padding
    index_raw_hl_2 = (index_raw_hl * stride) - padding

    if index_col_hl_2 < 0:
        index_col_hl_2 = 0
    if index_raw_hl_2 < 0:
        index_raw_hl_2 = 0

    index_col_hl_3 = (index_col_hl_2 * stride) - padding
    index_raw_hl_3 = (index_raw_hl_2 * stride) - padding

    if index_col_hl_3 < 0:
        index_col_hl_3 = 0
    if index_raw_hl_3 < 0:
        index_raw_hl_3 = 0

    index_col_hl_4 = (index_col_hl_3 * stride) - padding
    index_raw_hl_4 = (index_raw_hl_3 * stride) - padding

    if index_col_hl_4 < 0:
        reduice_region_col_size = index_col_hl_4
        index_col_hl_4 = 0
    else:
        reduice_region_col_size = 0
    if index_raw_hl_4 < 0:
        reduice_region_raw_size = index_raw_hl_4
        index_raw_hl_4 = 0
    else:
        reduice_region_raw_size = 0

    begin_col = index_col_hl_4
    end_col = index_col_hl_4 + region_shape + reduice_region_col_size
    begin_raw = index_raw_hl_4
    end_raw = index_raw_hl_4 + region_shape + reduice_region_raw_size

    if end_col > len_img_w:
        end_col = len_img_w
    if end_raw > len_img_h:
        end_raw = len_img_h

    region = image[begin_col:end_col, begin_raw:end_raw]
    if region.shape != (region_shape, region_shape):
        print('reshape region', region.shape)
        if type(region) == np.ndarray:
            region = cv2.resize(region, (region_shape, region_shape), interpolation=cv2.INTER_AREA)
        else:
            region = cv2.resize(region.numpy(), (region_shape, region_shape), interpolation=cv2.INTER_AREA)

    if return_all:
        return region, begin_col, end_col, begin_raw, end_raw
    else:
        return region
