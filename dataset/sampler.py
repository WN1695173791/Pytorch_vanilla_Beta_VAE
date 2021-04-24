import numpy as np
from torch.utils.data.sampler import Sampler, BatchSampler
from torch.utils.data import DataLoader
import torch
import random

is_torchvision_installed = True
try:
    import torchvision
except:
    is_torchvision_installed = False


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


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def _get_label(self, dataset, idx, labels=None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if is_torchvision_installed and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif is_torchvision_installed and dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max * len(self.keys)


class OneDataClassSampler(BatchSampler):
    """
        BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
        Returns batches of size n_classes * n_samples
        """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

            break  # add break only for have one data per class i.e. n_sample=1.

    def __len__(self):
        return len(self.dataset) // self.batch_size
