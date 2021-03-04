import numpy as np
from torch.utils.data.sampler import Sampler
import torch


class BalancedSampler(Sampler):
    def __init__(self, data_source, batch_size, images_per_class=3):
        self.data_source = data_source
        self.ys = self.data_source.train_labels
        self.num_groups = batch_size // images_per_class
        self.batch_size = batch_size
        self.num_instances = images_per_class
        self.num_samples = len(self.ys)
        self.num_classes = len(torch.unique(self.ys))
        if self.num_groups > self.num_classes:
            self.num_groups = self.num_classes

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        num_batches = len(self.data_source) // self.batch_size
        ret = []
        while num_batches > 0:
            sampled_classes = np.random.choice(self.num_classes, self.num_groups, replace=False)
            for i in range(len(sampled_classes)):
                ith_class_idxs = np.nonzero(np.array(self.ys) == sampled_classes[i])[0]
                class_sel = np.random.choice(ith_class_idxs, size=self.num_instances, replace=True)
                ret.extend(np.random.permutation(class_sel))
            num_batches -= 1
        return iter(ret)


class BalanceSamplesPerClass(Sampler):
    def __init__(self, data_source, batch_size, images_per_class=10):
        self.data_source = data_source
        self.batch_size = batch_size
        self.labels = self.data_source.train_labels
        self.num_classes = len(torch.unique(self.labels))
        self.num_groups = batch_size // images_per_class

    def __iter__(self):
        num_batches = len(self.data_source) // self.batch_size
        ret = []
        while num_batches > 0:
            classes = np.arange(self.num_classes)
            sampled_classes = classes.repeat(self.num_groups, axis=0)
            np.random.shuffle(sampled_classes)
            for i in range(len(sampled_classes)):
                ith_class_idxs = np.nonzero(np.array(self.labels) == sampled_classes[i])[0]
                class_sel = np.random.choice(ith_class_idxs, 1, replace=True)
                ret.extend(np.random.permutation(class_sel))
            num_batches -= 1
        return iter(ret)

    def __len__(self):
        return len(self.data_source)
