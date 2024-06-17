import torchvision.transforms as tf
from datasets import TieredImagenetCoarse2Fine
from datasets.utils.wrappers import NeighborsWrapper
from transforms import MultiViewGeneratorWithDifferentAugmentations
from transforms.image import GaussianBlur

def construct_train_transforms(cfg):
    return tf.Compose([
        tf.RandomResizedCrop(224, scale=(0.2, 1.)),
        tf.RandomApply([
            tf.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        tf.RandomGrayscale(p=0.2),
        tf.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        tf.RandomHorizontalFlip(),
        tf.ToTensor(),
        tf.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    ])

def construct_val_transforms(cfg):
    return tf.Compose([
        tf.Resize(256),
        tf.CenterCrop(224),
        tf.ToTensor(),
        tf.Normalize(mean=[0.4717, 0.4499, 0.3837], std=[0.2600, 0.2516, 0.2575])
    ])


def construct_tiered_imagenet_c2f_splits(cfg):
    dataroot = cfg.DATASET.DATAROOT
    transform_train = construct_train_transforms(cfg)
    transform_validation = construct_val_transforms(cfg)

    if cfg.DATASET.MULTIVIEW:
        transform_train = MultiViewGeneratorWithDifferentAugmentations(
            strong_transform=transform_train, weak_transform=transform_validation)

    train_ds = TieredImagenetCoarse2Fine(dataroot, split='train', transforms=transform_train, allow_cache= not cfg.RANK_ID)

    val_ds = TieredImagenetCoarse2Fine(dataroot, split='val', transforms=transform_validation, allow_cache= not cfg.RANK_ID)

    test_ds = TieredImagenetCoarse2Fine(dataroot, split='val', transforms=transform_validation, allow_cache= not cfg.RANK_ID)

    if cfg.NEIGHBORS is not None:
        train_ds = NeighborsWrapper(train_ds, cfg.NEIGHBORS, 5)

    print(f'Number of training samples: {len(train_ds)}, number of validation samples: {len(val_ds)}, number of test samples: {len(test_ds)}')
    fine_classes = train_ds.num_fine if hasattr(train_ds, 'num_fine') else train_ds.dataset.num_fine
    coarse_classes = train_ds.num_coarse if hasattr(train_ds, 'num_coarse') else train_ds.dataset.num_coarse
    return train_ds, val_ds, test_ds, fine_classes, coarse_classes
