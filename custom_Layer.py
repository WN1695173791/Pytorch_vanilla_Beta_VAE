from torch.autograd import Variable
from abc import ABC
import torch.nn as nn


class Flatten(nn.Module, ABC):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class View(nn.Module, ABC):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class PrintLayer(nn.Module, ABC):

    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


def reparameterization_trick(mu, logvar):
    """
    Samples from a normal distribution using the reparameterization trick.
    :param mu: torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)
    :param logvar: torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
    :return:
    """
    """
    if mu.training:
        std = logvar.div(2).exp()
        eps = Variable(std.data.new(std.size()).normal_())
        return mu + std * eps
    else:
        # Reconstruction mode
        return mu
    """
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps
