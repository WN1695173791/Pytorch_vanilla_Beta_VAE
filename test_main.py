import logging
import os

from visualizer import *

import torch
import torch.nn.functional as F
import torch.optim as optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm

# import losses
from dataset import sampler
from dataset.dataset_2 import get_dataloaders, get_mnist_dataset
from models.custom_CNN import Custom_CNN
from models.custom_CNN_BK import Custom_CNN_BK
from models.custom_CNN_BK import compute_ratio_batch_test, compute_var_distance_class_test
from models.default_CNN import DefaultCNN
from pytorchtools import EarlyStopping
from scores_classifier import compute_scores
from solver import gpu_config
from visualizer_CNN import get_layer_zstruct_num
import numpy as np
from dataset.sampler import BalancedBatchSampler
import random
from models.Encoder_decoder import Encoder_decoder
import torch.nn as nn
import matplotlib.pyplot as plt


img_size = (1, 32, 32)


def build_compare_reconstruction(size, data, input_data, x_recon):
    # reconstructions
    num_images = int(size[0] * size[1] / 2)
    if data.shape[1] == 3:
        originals = input_data[:num_images].cpu()
    else:
        originals = input_data[:num_images].cpu()
    reconstructions = x_recon.view(-1, *img_size)[:num_images].cpu()
    # If there are fewer examples given than spaces available in grid,
    # augment with blank images
    num_examples = originals.size()[0]
    if num_images > num_examples:
        blank_images = torch.zeros((num_images - num_examples,) + originals.size()[1:])
        originals = torch.cat([originals, blank_images])
        reconstructions = torch.cat([reconstructions, blank_images])

    # Concatenate images and reconstructions
    comparison = torch.cat([originals, reconstructions])

    return comparison


train_loader, test_loader = get_mnist_dataset(batch_size=64)

net = Encoder_decoder(z_struct_size=32,
                      big_kernel_size=8,
                      stride_size=1,
                      classif_layer_size=30,
                      add_classification_layer=False,
                      hidden_filters_1=32,
                      hidden_filters_2=32,
                      hidden_filters_3=32,
                      BK_in_first_layer=True,
                      two_conv_layer=True,
                      three_conv_layer=False,
                      BK_in_second_layer=False,
                      BK_in_third_layer=False,
                      Binary_z=False,
                      add_linear_after_GMP=False)

net, device = gpu_config(net)

print(net)
num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('The number of parameters of model is', num_params)

optimizer = optimizer.Adam(net.parameters(), lr=1e-4)
out = False

epoch = 10


for i in range(epoch):
    for batch_idx, (data, labels) in enumerate(train_loader):
        print(batch_idx, len(train_loader))

        data = data.to(device)  # Variable(data.to(self.device))
        labels = labels.to(device)  # Variable(labels.to(self.device))

        x_recons, z_struct = net(data)
        mse_loss = F.mse_loss(x_recons, data)
        loss = mse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for data, labels in test_loader:
        batch = data.to(device)[:32]

    net.eval()
    size = (8, 8)

    with torch.no_grad():
        input_data = batch
    if torch.cuda.is_available():
        input_data = input_data.cuda()

    x_recon, _ = net(input_data)

    net.train()

    comparison = build_compare_reconstruction(size, batch, input_data, x_recon)
    reconstructions = make_grid(comparison.data, nrow=size[0])

    # grid with originals data
    recon_grid = reconstructions.permute(1, 2, 0)
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='w', edgecolor='k')
    ax.set(title=('model: test decoder'))

    ax.imshow(recon_grid.numpy())
    ax.axhline(y=size[0] // 2, linewidth=4, color='r')
    plt.show()

    fig.savefig("fig_results/reconstructions/fig_reconstructions_z_" + 'test_decoder_only_' + str(epoch) + ".png")
