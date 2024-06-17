import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

def dump_to_txt(images, path):
    with open(path, 'w') as f:
        for img in images:
            f.write(img + '\n')

def load_from_txt(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

def load_f2c_mapping():
    import csv
    fine2coarse = {}
    for split in ['train', 'val', 'test']:
        path = f'datasets/tiered_imagenet/hierarchies/{split}.csv'
        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                fine2coarse[row[0]] = row[1]
    return fine2coarse

# Not exacty tiered imagenet since we do not resize images and we collect all classes.
class TieredImagenetCoarse2Fine(Dataset):
    def __init__(self, dataroot, split='val', transforms=None, allow_cache=False):
        assert split in ['train', 'val']
        self.dataroot = os.path.join(dataroot, split)
        print(f'Loading ImageNet from {self.dataroot}')
        self.split = split
        self.transforms = transforms

        f2c_mapping = load_f2c_mapping()

        self.images, self.labels, self.coarse_labels = TieredImagenetCoarse2Fine.load_maybe_cached_data(dataroot, split, f2c_mapping, allow_cache)

        self.num_fine = len(set(self.labels))
        self.num_coarse = len(set(self.coarse_labels))

    @staticmethod
    def load_maybe_cached_data(dataroot, split, f2c_mapping, allow_cache):
        cache_folder = f'{dataroot}/c2f_dataset_cache/tiered_imagenet_{split}'
        if os.path.exists(os.path.join(cache_folder, 'fine.pt')):
            print('Loading cached data')
            labels = torch.load(os.path.join(cache_folder, 'fine.pt'))
            coarse_labels = torch.load(os.path.join(cache_folder, 'coarse.pt'))
            images = load_from_txt(os.path.join(cache_folder, 'images.txt'))
        else:
            os.makedirs(cache_folder, exist_ok=True)
            images, labels, coarse_labels = TieredImagenetCoarse2Fine.load_data(dataroot, split, f2c_mapping)
            if allow_cache:
                print('>> Caches filled')
                torch.save(labels, os.path.join(cache_folder, 'fine.pt'))
                torch.save(coarse_labels, os.path.join(cache_folder, 'coarse.pt'))
                dump_to_txt(images, os.path.join(cache_folder, 'images.txt'))
        return images, labels, coarse_labels

    @staticmethod
    def load_data(dataroot, split, f2c_mapping):
        images = sorted(glob.glob(os.path.join(dataroot, split) + '/*/*.JPEG'))
        assert len(images) > 0

        coarse_classes = list(set(f2c_mapping.values()))
        fine_classes = sorted(list(set(f2c_mapping.keys())))
        coarse_labels = []
        fine_labels = []
        filtered_images = []
        for img in images:
            lbl = img.split('/')[-2]
            if lbl not in fine_classes:
                continue
            fine_labels.append(fine_classes.index(lbl))
            coarse_labels.append(coarse_classes.index(f2c_mapping[lbl]))
            filtered_images.append(img)

        assert np.unique(fine_labels).shape[0] == len(fine_classes)
        assert np.unique(coarse_labels).shape[0] == len(coarse_classes)
        return filtered_images, fine_labels, coarse_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img = Image.open(self.images[item]).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        return {
            'index': item,
            'inputs': img,
            'fine_label': self.labels[item],
            'coarse_label': self.coarse_labels[item],
        }

    def get_graph(self):
        M = torch.zeros((self.num_fine, self.num_coarse))
        for coarse, fine in zip(self.coarse_labels, self.labels):
            M[fine, coarse] = 1
        assert torch.sum(M) == len(set(self.labels))
        return M.numpy()


