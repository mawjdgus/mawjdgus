import csv, torchvision, numpy as np, random, os
from PIL import Image
import torch
from torch.utils.data import Sampler, Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler, Subset
from torchvision import transforms, datasets
from collections import defaultdict


class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                pair_indices.append(random.choice(self.dataset.classwise_indices[y]))

            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset)+self.batch_size-1) // self.batch_size
        else:
            return self.num_iterations


class DatasetWrapper(Dataset):
    # Additinoal attributes
    # - indices
    # - classwise_indices
    # - num_classes
    # - get_class

    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        # torchvision 0.2.0 compatibility
        if torchvision.__version__.startswith('0.2'):
            if isinstance(self.base_dataset, datasets.ImageFolder):
                self.base_dataset.targets = [s[1] for s in self.base_dataset.imgs]
            else:
                if self.base_dataset.train:
                    self.base_dataset.targets = self.base_dataset.train_labels
                else:
                    self.base_dataset.targets = self.base_dataset.test_labels

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys())+1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]


class ConcatWrapper(Dataset): # TODO: Naming
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    @staticmethod
    def numcls(sequence):
        s = 0
        for e in sequence:
            l = e.num_classes
            s += l
        return s

    @staticmethod
    def clsidx(sequence):
        r, s, n = defaultdict(list), 0, 0
        for e in sequence:
            l = e.classwise_indices
            for c in range(s, s + e.num_classes):
                t = np.asarray(l[c-s]) + n
                r[c] = t.tolist()
            s += e.num_classes
            n += len(e)
        return r

    def __init__(self, datasets):
        super(ConcatWrapper, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        # for d in self.datasets:
        #     assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

        self.num_classes = self.numcls(self.datasets)
        self.classwise_indices = self.clsidx(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def get_class(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        true_class = self.datasets[dataset_idx].base_dataset.targets[self.datasets[dataset_idx].indices[sample_idx]]
        return self.datasets[dataset_idx].base_dataset.target_transform(true_class)

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes



def load_dataset(name, root, sample='default', **kwargs):
    # Dataset
    if name in ['imagenet','tinyimagenet', 'CUB200', 'STANFORD120', 'MIT67']:
        # TODO
        if name == 'tinyimagenet':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            train_val_dataset_dir = os.path.join(root, "train")
            test_dataset_dir = os.path.join(root, "val")

            trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))
            valset   = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))

        elif name == 'imagenet':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            train_val_dataset_dir = os.path.join(root, "train")
            test_dataset_dir = os.path.join(root, "val")

            trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))
            valset   = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))

        else:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            train_val_dataset_dir = os.path.join(root, name, "train")
            test_dataset_dir = os.path.join(root, name, "test")

            trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))
            valset   = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))

    elif name.startswith('cifar'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if name == 'cifar10':
            CIFAR = datasets.CIFAR10
        else:
            CIFAR = datasets.CIFAR100

        trainset = DatasetWrapper(CIFAR(root, train=True,  download=True, transform=transform_train))
        valset   = DatasetWrapper(CIFAR(root, train=False, download=True, transform=transform_test))
    else:
        raise Exception('Unknown dataset: {}'.format(name))

    # Sampler
    if sample == 'default':
        get_train_sampler = lambda d: BatchSampler(RandomSampler(d), kwargs['batch_size'], False)
        get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), kwargs['batch_size'], False)

    elif sample == 'pair':
        get_train_sampler = lambda d: PairBatchSampler(d, kwargs['batch_size'])
        get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), kwargs['batch_size'], False)

    else:
        raise Exception('Unknown sampling: {}'.format(sampling))

    trainloader = DataLoader(trainset, batch_sampler=get_train_sampler(trainset), num_workers=4)
    valloader   = DataLoader(valset,   batch_sampler=get_test_sampler(valset), num_workers=4)

    return trainloader, valloader

class CustomDataset(Dataset):
    
    def __init__(self, root, train=True):

        self.num_classes = 1000
        
        self.transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        self.transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        
        self.train_val_dataset_dir = os.path.join(root, "train")
        self.test_dataset_dir = os.path.join(root, "val")
        
        if train:
            self.base_dataset = DatasetWrapper(datasets.ImageFolder(root=self.train_val_dataset_dir, transform=self.transform_train))
        else:
            self.base_dataset   = DatasetWrapper(datasets.ImageFolder(root=self.test_dataset_dir, transform=self.transform_test))
                
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        batch_index = idx
        pair_index = 0
        y = self.base_dataset.get_class(idx)
        pair_index = (random.choice(self.base_dataset.classwise_indices[y]))
        pair_1 = self.base_dataset[batch_index][0]
        pair_2 = self.base_dataset[pair_index][0]
        label_1 = torch.tensor(self.base_dataset[batch_index][1])
        label_2 = torch.tensor(self.base_dataset[pair_index][1])
        # pair = torch.cat((pair_1, pair_2))
        # label = torch.cat((label_1.unsqueeze(0),label_2.unsqueeze(0)))
        return  pair_1,pair_2, label_1, label_2

# train_dataloader = DataLoader(trainset, batch_sampler=torch.dist.DistributedSampler)



# import math
# from typing import TypeVar, Optional, Iterator

# import torch
# from . import Sampler, Dataset
# import torch.distributed as dist


# T_co = TypeVar('T_co', covariant=True)


# class DistributedSampler(Sampler[T_co]):
#     r"""Sampler that restricts data loading to a subset of the dataset.

#     It is especially useful in conjunction with
#     :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
#     process can pass a :class:`~torch.utils.data.DistributedSampler` instance as a
#     :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
#     original dataset that is exclusive to it.

#     .. note::
#         Dataset is assumed to be of constant size.

#     Args:
#         dataset: Dataset used for sampling.
#         num_replicas (int, optional): `Number` of processes participating in
#             distributed training. By default, :attr:`world_size` is retrieved from the
#             current distributed group.
#         rank (int, optional): Rank of the current process within :attr:`num_replicas`.
#             By default, :attr:`rank` is retrieved from the current distributed
#             group.
#         shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
#             indices.
#         seed (int, optional): random seed used to shuffle the sampler if
#             :attr:`shuffle=True`. This number should be identical across all
#             processes in the distributed group. Default: ``0``.
#         drop_last (bool, optional): if ``True``, then the sampler will drop the
#             tail of the data to make it evenly divisible across the number of
#             replicas. If ``False``, the sampler will add extra indices to make
#             the data evenly divisible across the replicas. Default: ``False``.

#     .. warning::
#         In distributed mode, calling the :meth:`set_epoch` method at
#         the beginning of each epoch **before** creating the :class:`DataLoader` iterator
#         is necessary to make shuffling work properly across multiple epochs. Otherwise,
#         the same ordering will be always used.

#     Example::

#         >>> sampler = DistributedSampler(dataset) if is_distributed else None
#         >>> loader = DataLoader(dataset, shuffle=(sampler is None),
#         ...                     sampler=sampler)
#         >>> for epoch in range(start_epoch, n_epochs):
#         ...     if is_distributed:
#         ...         sampler.set_epoch(epoch)
#         ...     train(loader)
#     """

#     def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
#                  rank: Optional[int] = None, shuffle: bool = True,
#                  seed: int = 0, drop_last: bool = False) -> None:
#         if num_replicas is None:
#             if not dist.is_available():
#                 raise RuntimeError("Requires distributed package to be available")
#             num_replicas = dist.get_world_size()
#         if rank is None:
#             if not dist.is_available():
#                 raise RuntimeError("Requires distributed package to be available")
#             rank = dist.get_rank()
#         if rank >= num_replicas or rank < 0:
#             raise ValueError(
#                 "Invalid rank {}, rank should be in the interval"
#                 " [0, {}]".format(rank, num_replicas - 1))
#         self.dataset = dataset
#         self.num_replicas = num_replicas
#         self.rank = rank
#         self.epoch = 0
#         self.drop_last = drop_last
#         # If the dataset length is evenly divisible by # of replicas, then there
#         # is no need to drop any data, since the dataset will be split equally.
#         if self.drop_last and len(self.dataset) % self.num_replicas != 0:
#             # Split to nearest available length that is evenly divisible.
#             # This is to ensure each rank receives the same amount of data when
#             # using this Sampler.
#             self.num_samples = math.ceil(
#                 (len(self.dataset) - self.num_replicas) / self.num_replicas
#             )
#         else:
#             self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
#         self.total_size = self.num_samples * self.num_replicas
#         self.shuffle = shuffle
#         self.seed = seed

#     def __iter__(self) -> Iterator[T_co]:
#         if self.shuffle:
#             # deterministically shuffle based on epoch and seed
#             g = torch.Generator()
#             g.manual_seed(self.seed + self.epoch)
#             indices = torch.randperm(len(self.dataset), generator=g).tolist()
#         else:
#             indices = list(range(len(self.dataset)))

#         if not self.drop_last:
#             # add extra samples to make it evenly divisible
#             padding_size = self.total_size - len(indices)
#             if padding_size <= len(indices):
#                 indices += indices[:padding_size]
#             else:
#                 indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
#         else:
#             # remove tail of data to make it evenly divisible.
#             indices = indices[:self.total_size]
#         assert len(indices) == self.total_size

#         # subsample
#         indices = indices[self.rank:self.total_size:self.num_replicas]
#         assert len(indices) == self.num_samples

#         return iter(indices)

#     def __len__(self) -> int:
#         return self.num_samples

#     def set_epoch(self, epoch: int) -> None:
#         r"""
#         Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
#         use a different random ordering for each epoch. Otherwise, the next iteration of this
#         sampler will yield the same ordering.

#         Args:
#             epoch (int): Epoch number.
#         """
#         self.epoch = epoch