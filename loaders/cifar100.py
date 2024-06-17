import torchvision.transforms as tf
from datasets import CIFAR100Coarse2Fine, CIFAR68Coarse2Fine
from transforms import MultiViewGenerator
from datasets.utils.wrappers import NeighborsWrapper
from torch.utils.data import Subset
import torch

def construct_train_transforms(cfg):
    return tf.Compose([
        tf.RandomResizedCrop(size=32, scale=(0.08, 1.0)),
        tf.RandomHorizontalFlip(),
        tf.ToTensor(),
        tf.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

def construct_val_transforms(cfg):
    return tf.Compose([
        tf.ToTensor(),
        tf.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

def construct_cifar100_c2f_splits(cfg):
    train_transform = construct_train_transforms(cfg)

    if cfg.DATASET.MULTIVIEW:
        train_transform = MultiViewGenerator(train_transform, n_views=cfg.DATASET.MULTIVIEW_NUM)

    train_ds = CIFAR100Coarse2Fine(cfg.DATASET.DATAROOT, split='train', transforms=train_transform, sample_imbalance=cfg.DATASET.SAMPLE_IMBALANCED)

    val_transform = construct_val_transforms(cfg)

    val_ds = CIFAR100Coarse2Fine(cfg.DATASET.DATAROOT, split='val', transforms=val_transform, sample_imbalance=cfg.DATASET.SAMPLE_IMBALANCED)
    test_ds = CIFAR100Coarse2Fine(cfg.DATASET.DATAROOT, split='test', transforms=val_transform, sample_imbalance=cfg.DATASET.SAMPLE_IMBALANCED)

    if cfg.NEIGHBORS is not None:
        train_ds = NeighborsWrapper(train_ds, cfg.NEIGHBORS, 5)


    fine_classes = test_ds.num_fine
    coarse_classes = test_ds.num_coarse
    return train_ds, val_ds, test_ds, fine_classes, coarse_classes


def construct_cifar68_c2f_splits(cfg):
    train_transform = construct_train_transforms(cfg)

    if cfg.DATASET.MULTIVIEW:
        train_transform = MultiViewGenerator(train_transform, n_views=cfg.DATASET.MULTIVIEW_NUM)

    train_ds = CIFAR68Coarse2Fine(cfg.DATASET.DATAROOT, split='train', transforms=train_transform)

    val_transform = construct_val_transforms(cfg)

    val_ds = CIFAR68Coarse2Fine(cfg.DATASET.DATAROOT, split='val', transforms=val_transform)
    test_ds = CIFAR68Coarse2Fine(cfg.DATASET.DATAROOT, split='test', transforms=val_transform)

    if cfg.NEIGHBORS is not None:
        train_ds = NeighborsWrapper(train_ds, cfg.NEIGHBORS, 5)

    fine_classes = test_ds.num_fine
    coarse_classes = test_ds.num_coarse
    return train_ds, val_ds, test_ds, fine_classes, coarse_classes


def construct_cifar100_twosource_split(cfg):
    train_transform = construct_train_transforms(cfg)

    if cfg.DATASET.MULTIVIEW:
        train_transform = MultiViewGenerator(train_transform, n_views=cfg.DATASET.MULTIVIEW_NUM)

    train_ds = CIFAR100Coarse2Fine(cfg.DATASET.DATAROOT, split='train', transforms=train_transform)

    val_transform = construct_val_transforms(cfg)

    val_ds = CIFAR100Coarse2Fine(cfg.DATASET.DATAROOT, split='val', transforms=val_transform)
    test_ds = CIFAR100Coarse2Fine(cfg.DATASET.DATAROOT, split='test', transforms=val_transform)

    fine_classes = test_ds.num_fine
    coarse_classes = test_ds.num_coarse
    supercoarse_classes = train_ds.num_supercoarse

    def label_transform_supercoarse(x):
        x['coarse_label'] = x['supercoarse_label']
        x['source'] = 1
        return x

    def label_transform_alternative(x):
        x['coarse_label'] = x['alternative_coarse_label']
        x['source'] = 2
        return x

    def label_transform_coarse(x):
        x['source'] = 0
        return x

    indices = torch.arange(len(train_ds))
    if cfg.DATASET.SOURCES == [0]:
        train_ds = SubsetWithLabelTransform(train_ds, indices[indices % 2 == 0], label_transform_coarse)
    elif cfg.DATASET.SOURCES == [1]:
        train_ds = SubsetWithLabelTransform(train_ds, indices, label_transform_supercoarse)
    elif cfg.DATASET.SOURCES == [2]:
        train_ds = SubsetWithLabelTransform(train_ds, indices[indices % 2 == 1], label_transform_alternative)
    elif cfg.DATASET.SOURCES == [0, 1]:
        train_ds_coarse = SubsetWithLabelTransform(train_ds, indices[indices % 2 == 0], label_transform_coarse)
        train_ds_supercoarse = SubsetWithLabelTransform(train_ds, indices[indices % 2 == 1], label_transform_supercoarse)
        train_ds = torch.utils.data.ConcatDataset([train_ds_coarse, train_ds_supercoarse])
    elif cfg.DATASET.SOURCES == [0, 2]:
        train_ds_coarse = SubsetWithLabelTransform(train_ds, indices[indices % 2 == 0], label_transform_coarse)
        train_ds_alternative_coarse = SubsetWithLabelTransform(train_ds, indices[indices % 2 == 1], label_transform_alternative)
        train_ds = torch.utils.data.ConcatDataset([train_ds_coarse, train_ds_alternative_coarse])
    else:
        raise NotImplementedError

    if cfg.NEIGHBORS is not None:
        train_ds = NeighborsWrapper(train_ds, cfg.NEIGHBORS, 5)


    return train_ds, val_ds, test_ds, fine_classes, coarse_classes, supercoarse_classes

def construct_cifar100_imbalanced_twosource_split(cfg):
    train_transform = construct_train_transforms(cfg)

    if cfg.DATASET.MULTIVIEW:
        train_transform = MultiViewGenerator(train_transform, n_views=cfg.DATASET.MULTIVIEW_NUM)

    train_ds = CIFAR68Coarse2Fine(cfg.DATASET.DATAROOT, split='train', transforms=train_transform)

    val_transform = construct_val_transforms(cfg)

    val_ds = CIFAR68Coarse2Fine(cfg.DATASET.DATAROOT, split='val', transforms=val_transform)
    test_ds = CIFAR68Coarse2Fine(cfg.DATASET.DATAROOT, split='test', transforms=val_transform)

    fine_classes = test_ds.num_fine
    coarse_classes = test_ds.num_coarse
    supercoarse_classes = train_ds.num_supercoarse

    def label_transform_supercoarse(x):
        x['coarse_label'] = x['supercoarse_label']
        x['source'] = 1
        return x

    def label_transform_coarse(x):
        x['source'] = 0
        return x

    indices = torch.arange(len(train_ds))
    if cfg.DATASET.SOURCES == [0]:
        train_ds = SubsetWithLabelTransform(train_ds, indices, label_transform_coarse)
    elif cfg.DATASET.SOURCES == [1]:
        train_ds = SubsetWithLabelTransform(train_ds, indices, label_transform_supercoarse)
    else:
        train_ds_coarse = SubsetWithLabelTransform(train_ds, indices[indices % 2 == 0], label_transform_coarse)
        train_ds_supercoarse = SubsetWithLabelTransform(train_ds, indices[indices % 2 == 1], label_transform_supercoarse)
        train_ds = torch.utils.data.ConcatDataset([train_ds_coarse, train_ds_supercoarse])

    if cfg.NEIGHBORS is not None:
        train_ds = NeighborsWrapper(train_ds, cfg.NEIGHBORS, 5)

    return train_ds, val_ds, test_ds, fine_classes, coarse_classes, supercoarse_classes