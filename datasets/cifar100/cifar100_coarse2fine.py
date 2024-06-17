import os
import pickle
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
from PIL import Image
from collections import defaultdict
from .labels import _FINE_LABEL_NAMES, DEFAULT_TAXONOMY, SUPERCLASS_TAXONOMY, SUPERSUPERCLAS_TAXONOMY, ALTERNATIVE_SUPERCLASS_TAXONOMY

portion_keep = np.array([0.57375399, 0.61887877, 0.49827284, 0.61363604, 0.70610113,
       0.84783802, 0.68711689, 0.9519748 , 0.88483212, 0.85602336,
       0.73676322, 0.5784213 , 0.34717017, 0.47788748, 0.42177804,
       0.61173724, 0.65665981, 0.5644105 , 0.79943416, 0.69019384,
       0.85861546, 0.84061074, 0.5655741 , 0.52854164, 0.87284635,
       0.44738893, 0.38830181, 0.63654859, 0.84592266, 0.45378019,
       0.59191851, 0.37450617, 0.97042908, 0.40045453, 0.50140883,
       0.82288002, 0.76198367, 0.65541795, 0.52453847, 0.47314545,
       0.96185752, 0.91658886, 0.49564212, 0.84868512, 0.91926797,
       0.70796942, 0.83786135, 0.33228417, 0.32755937, 0.75536667,
       0.3939889 , 0.72539474, 0.74988883, 0.40057496, 0.77408639,
       0.89928709, 0.3173045 , 0.63207938, 0.52763706, 0.4150968 ,
       0.97584858, 0.33739184, 0.88057277, 0.33795859, 0.93550844,
       0.69919175, 0.69305501, 0.33304447, 0.83976966, 0.48514858,
       0.49979666, 0.3912749 , 0.68442045, 0.43216038, 0.4289422 ,
       0.63583394, 0.59675707, 0.80642398, 0.42701694, 0.69398926,
       0.45437665, 0.98812622, 0.98534736, 0.54456234, 0.97894244,
       0.86929247, 0.7191352 , 0.70135236, 0.65428483, 0.82100301,
       0.89086774, 0.72972202, 0.64628091, 0.32348358, 0.76580364,
       0.80736886, 0.50139469, 0.65188098, 0.59059803, 0.36106552])

class _CIFAR100Raw(Dataset):

    def __init__(self, dataroot, split, sample_imbalance=False):
        assert split in ['train', 'val', 'test', 'trainval']
        self.dataroot = dataroot

        samples_loader = _CIFAR100Raw.load_images_imbalanced if sample_imbalance else _CIFAR100Raw.load_images
        self.dsid2img, self.ds_lbls = samples_loader(dataroot, split)
        print(len(self.ds_lbls), len(self.ds_lbls))
        self.id2dsid = _CIFAR100Raw.construct_split(self.ds_lbls, split)
        self.dsid2name = _FINE_LABEL_NAMES

    @staticmethod
    def load_images(dataroot, split):
        file = os.path.join(dataroot, split if 'val' not in split else 'train')
        with open(file, 'rb') as f:
            data = pickle.load(f, encoding="bytes")
            fine_labels = data[b"fine_labels"]
            images = data[b"data"]

            id2imgs = dict()
            for idx, _ in enumerate(images):
                img_reshaped = np.transpose(np.reshape(images[idx], (3, 32, 32)), (1, 2, 0))
                img = img_reshaped
                id2imgs[idx] = img
        return id2imgs, np.array(fine_labels)

    @staticmethod
    def load_images_imbalanced(dataroot, split):
        file = os.path.join(dataroot, split if 'val' not in split else 'train')
        with open(file, 'rb') as f:
            data = pickle.load(f, encoding="bytes")
            fine_labels = data[b"fine_labels"]
            images = data[b"data"]

            fine_labels = np.array(fine_labels)
            keep_image = np.zeros_like(fine_labels).astype(bool)
            for cls in np.unique(fine_labels):
                keep_percentage = portion_keep[cls]
                num_keep = int(len(np.where(fine_labels == cls)[0]) * keep_percentage)
                keep_image[np.where(fine_labels == cls)[0][:num_keep]] = True
            fine_labels = fine_labels[keep_image]

            id2imgs = dict()
            id = 0
            for idx, _ in enumerate(images):
                if not keep_image[idx]:
                    continue
                img_reshaped = np.transpose(np.reshape(images[idx], (3, 32, 32)), (1, 2, 0))
                img = img_reshaped
                id2imgs[id] = img
                id += 1
        return id2imgs, fine_labels

    @staticmethod
    def construct_split(lblbs, split):
        # train_imgs_per_class = 450
        indices = []
        indices_ = np.arange(len(lblbs))

        if split == 'test' or split == 'trainval':
            return indices_
        else:
            for lb in np.unique(lblbs):
                train_imgs_per_class = int(len(indices_[lb == lblbs]) * 0.9)
                if split == 'train':
                    indices.append(indices_[lb == lblbs][:train_imgs_per_class])
                elif split == 'val':
                    indices.append(indices_[lb == lblbs][train_imgs_per_class:])
            return np.hstack(indices)

    def __getitem__(self, idx):
        dsid = self.id2dsid[idx]
        image = Image.fromarray(self.dsid2img[dsid]).convert('RGB')
        label = self.ds_lbls[dsid]
        label_str = self.dsid2name[label]

        return image, label_str

    def get_label_str(self, idx):
        dsid = self.id2dsid[idx]
        label = self.ds_lbls[dsid]
        label_str = self.dsid2name[label]
        return label_str

    def __len__(self):
        return len(self.id2dsid)


class CIFAR100Coarse2Fine(_CIFAR100Raw):

    def __init__(self, dataroot, split, transforms, sample_imbalance=False):
        super().__init__(dataroot, split, sample_imbalance)
        self.transform = transforms
        self.coarse_taxonomy = SUPERCLASS_TAXONOMY
        self.fine_taxonomy = DEFAULT_TAXONOMY
        print(f"Dataset {split} size: {len(self)}")

        self.coarse_labels = self._gather_labels(self.coarse_taxonomy)
        self.fine_labels = self._gather_labels(self.fine_taxonomy)
        self.num_supercoarse = 10
        self.num_coarse = 20
        self.num_fine = 100

    def _gather_labels(self, taxonomy):
        return torch.tensor([taxonomy.get_id(self.get_label_str(idx)) for idx in range(len(self))]).long()

    def __getitem__(self, idx):
        img, lbl_name = super().__getitem__(idx)
        fine_lbl_id = self.fine_taxonomy.get_id(lbl_name)
        coarse_lbl_id = self.coarse_taxonomy.get_id(lbl_name)
        return {
            'index': idx,
            'inputs': self.transform(img) if self.transform is not None else img,
            'fine_label': fine_lbl_id,
            'coarse_label': coarse_lbl_id,
        }

    def get_graph(self):
        M = np.zeros((self.num_fine, self.num_coarse))
        for fine, coarse in zip(self.fine_labels, self.coarse_labels):
            M[fine, coarse] = 1
        return M


class CIFAR68Coarse2Fine(_CIFAR100Raw):

    def __init__(self, dataroot, split, transforms):
        super().__init__(dataroot, split)
        self.transform = transforms
        self.coarse_taxonomy = SUPERCLASS_TAXONOMY
        self.fine_taxonomy = DEFAULT_TAXONOMY

        self.coarse_labels = self._gather_labels(self.coarse_taxonomy)
        self.fine_labels = self._gather_labels(self.fine_taxonomy)

        self.num_supercoarse = 10
        self.num_coarse = 20
        self.num_fine = 100

        leftout_classes = self._classes_for_removal()
        kept_classes = [i for i in range(self.num_fine) if i not in leftout_classes]
        remove_element = torch.zeros(len(self.fine_labels)).bool()
        for cls in leftout_classes:
            remove_element[self.fine_labels == cls] = True

        fine_class_mapper = torch.zeros(self.num_fine).long()
        fine_class_mapper[leftout_classes] = -1
        fine_class_mapper[kept_classes] = torch.arange(len(kept_classes)).long()

        self.idx_mapper = torch.arange(len(self.fine_labels))[~remove_element].long()

        self.coarse_labels = self.coarse_labels[~remove_element]
        self.fine_labels = self.fine_labels[~remove_element]
        self.fine_labels = fine_class_mapper[self.fine_labels]

        self.num_fine = 100 - len(leftout_classes)
        print(f'Fine per coarse: {self.get_graph().sum(0).mean()} +/- {self.get_graph().sum(0).std()}')

        print(f"Dataset {split} size: {len(self)}")


    def _gather_labels(self, taxonomy):
        num_images = len(self.id2dsid)
        return torch.tensor([taxonomy.get_id(self.get_label_str(idx)) for idx in range(num_images)]).long()


    def __getitem__(self, idx_):
        idx = self.idx_mapper[idx_]
        img, lbl_name = super().__getitem__(idx)
        return {
            'index': idx_,
            'inputs': self.transform(img) if self.transform is not None else img,
            'fine_label': self.fine_labels[idx_],
            'coarse_label': self.coarse_labels[idx_],
        }


    def __len__(self):
        return len(self.idx_mapper)


    def get_graph(self):
        M = np.zeros((self.num_fine, self.num_coarse))
        for fine, coarse in zip(self.fine_labels, self.coarse_labels):
            M[fine, coarse] = 1
        return M

    def _classes_for_removal(self):
        return [0, 2,
                7, 9,
                11,
                15, 16,
                21, 22,
                25, 27, 26,
                30,
                #36,
                40, 42,
                45, 47, 49,
                50, 51,
                57,
                61,
                # 65, 66,
                70, 72,
                #75,
                81,  85,
                88, 89,
                90, 93, 94,
                95,
                ]
