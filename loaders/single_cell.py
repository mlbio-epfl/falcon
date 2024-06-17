from datasets import SingleCellData
from datasets.utils.wrappers import NeighborsWrapper


# single cell data uses transductive setting
def construct_single_cell_c2f_splits(cfg):
    file = cfg.DATASET.DATAROOT.replace('.h5ad', '_preprocessed.pth')

    train_ds = SingleCellData(file, two_views=cfg.DATASET.MULTIVIEW)

    val_ds = SingleCellData(file)

    test_ds = SingleCellData(file)

    if cfg.NEIGHBORS is not None:
        train_ds = NeighborsWrapper(train_ds, cfg.NEIGHBORS, 5)

    print(f'Number of training samples: {len(train_ds)}, number of validation samples: {len(val_ds)}, number of test samples: {len(test_ds)}')
    fine_classes = train_ds.num_fine if hasattr(train_ds, 'num_fine') else train_ds.dataset.num_fine
    coarse_classes = train_ds.num_coarse if hasattr(train_ds, 'num_coarse') else train_ds.dataset.num_coarse
    return train_ds, val_ds, test_ds, fine_classes, coarse_classes
