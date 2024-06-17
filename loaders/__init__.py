import torch
from path import dataset_paths

BREEDS_DATASETS = ['living17', 'entity30', 'entity13', 'nonliving26']

def construct_coarse2fine_loader(cfg):
    if cfg.DATASET.NAME in BREEDS_DATASETS:
        cfg.DATASET.DATAROOT = dataset_paths['ImageNet']
    elif 'ImageNet' in cfg.DATASET.NAME:
        cfg.DATASET.DATAROOT = dataset_paths['ImageNet']
    elif 'CIFAR' in cfg.DATASET.NAME:
        cfg.DATASET.DATAROOT = dataset_paths['CIFAR100']
    else:
        cfg.DATASET.DATAROOT = dataset_paths[cfg.DATASET.NAME]

    if 'CIFAR100' in cfg.DATASET.NAME:
        from .cifar100 import construct_cifar100_c2f_splits
        train, val, test, fine_classes, coarse_classes = construct_cifar100_c2f_splits(cfg)
    elif cfg.DATASET.NAME == 'CIFAR68':
        from .cifar100 import construct_cifar68_c2f_splits
        train, val, test, fine_classes, coarse_classes = construct_cifar68_c2f_splits(cfg)
    elif cfg.DATASET.NAME in BREEDS_DATASETS:
        from .breeds import construct_breeds_c2f_splits
        train, val, test, fine_classes, coarse_classes = construct_breeds_c2f_splits(cfg)
    elif cfg.DATASET.NAME == 'tieredImageNet':
        from .tiered_imagenet import construct_tiered_imagenet_c2f_splits
        train, val, test, fine_classes, coarse_classes = construct_tiered_imagenet_c2f_splits(cfg)
    elif cfg.DATASET.NAME == 'PBMC':
        from .single_cell import construct_single_cell_c2f_splits
        train, val, test, fine_classes, coarse_classes = construct_single_cell_c2f_splits(cfg)
    elif cfg.DATASET.NAME == 'Nasal':
        from .single_cell import construct_single_cell_c2f_splits
        train, val, test, fine_classes, coarse_classes = construct_single_cell_c2f_splits(cfg)

    else:
        raise NotImplementedError('Dataset not implemented: {}'.format(cfg.DATASET.NAME))

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train,
        num_replicas=cfg.SOLVER.DEVICES,
        rank=cfg.RANK_ID,
        shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=cfg.SOLVER.BATCH_SIZE,
                                               shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
                                               pin_memory=cfg.DATALOADER.PIN_MEMORY, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(dataset=val, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                                             pin_memory=cfg.DATALOADER.PIN_MEMORY)

    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                                             pin_memory=cfg.DATALOADER.PIN_MEMORY)

    return train_loader, val_loader, test_loader, fine_classes, coarse_classes


def construct_two_source_loader(cfg):
    if cfg.DATASET.NAME in BREEDS_DATASETS:
        cfg.DATASET.DATAROOT = dataset_paths['ImageNet']
    elif 'ImageNet' in cfg.DATASET.NAME:
        cfg.DATASET.DATAROOT = dataset_paths['ImageNet']
    elif 'CIFAR100' in cfg.DATASET.NAME:
        cfg.DATASET.DATAROOT = dataset_paths['CIFAR100']
    else:
        cfg.DATASET.DATAROOT = dataset_paths[cfg.DATASET.NAME]

    if cfg.DATASET.NAME == 'CIFAR100':
        from .cifar100 import construct_cifar100_twosource_split
        train, val, test, fine_classes, coarse_classes, supercoarse_classes = construct_cifar100_twosource_split(cfg)
    elif cfg.DATASET.NAME == 'CIFAR100Imbalanced':
        from .cifar100 import construct_cifar100_imbalanced_twosource_split
        train, val, test, fine_classes, coarse_classes, supercoarse_classes = construct_cifar100_imbalanced_twosource_split(cfg)
    else:
        raise NotImplementedError('Dataset not implemented: {}'.format(cfg.DATASET.NAME))

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train,
        num_replicas=cfg.SOLVER.DEVICES,
        rank=cfg.RANK_ID,
        shuffle=True,
    )
    train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=cfg.SOLVER.BATCH_SIZE,
                                               shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS,
                                               pin_memory=cfg.DATALOADER.PIN_MEMORY, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(dataset=val, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                                             pin_memory=cfg.DATALOADER.PIN_MEMORY)

    test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
                                             num_workers=cfg.DATALOADER.NUM_WORKERS,
                                             pin_memory=cfg.DATALOADER.PIN_MEMORY)

    return train_loader, val_loader, test_loader, fine_classes, coarse_classes, supercoarse_classes
