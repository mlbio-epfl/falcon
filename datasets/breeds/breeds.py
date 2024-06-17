# cp from https://github.com/nijingchao/SCGM/blob/main/dataset/dataset_breeds.py with minimal adaptation
import numpy as np
import torch
import pickle as pkl
import os
from torch.utils.data import Dataset
from robustness.tools import folder
from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26
from robustness.tools.helpers import get_label_mapping
from sklearn.utils import shuffle

class BREEDSFactory:
    def __init__(self, info_dir, data_dir):
        self.info_dir = info_dir
        self.data_dir = data_dir

    def get_breeds(self, ds_name, partition, transforms=None, split=None):
        superclasses, subclass_split, label_map = self.get_classes(ds_name, split)
        partition = 'val' if partition == 'validation' else partition
        # print(f"==> Preparing dataset {ds_name}, split: {split}, partition: {partition}..")
        if split is not None:
            # split can be  'good','bad' or None. if not None, 'subclass_split' will have 2 items, for 'train' and 'test'. otherwise, just 1
            index = 0 if partition == 'train' else 1
            return self.create_dataset(partition, subclass_split[index], transforms)
        else:
            return self.create_dataset(partition, subclass_split[0], transforms)

    def create_dataset(self, partition, subclass_split, transforms):
        coarse_custom_label_mapping = get_label_mapping("custom_imagenet", subclass_split)
        fine_subclass_split = [[item] for sublist in subclass_split for item in sublist]
        fine_custom_label_mapping = get_label_mapping("custom_imagenet", fine_subclass_split)

        active_custom_label_mapping = fine_custom_label_mapping
        active_subclass_split = fine_subclass_split

        dataset = folder.ImageFolder(root=os.path.join(self.data_dir, partition), transform=transforms, label_mapping=active_custom_label_mapping)
        coarse2fine, coarse_labels, idx_to_fine_class = self.extract_c2f_from_dataset(dataset, coarse_custom_label_mapping, fine_custom_label_mapping, partition)
        setattr(dataset, 'num_classes', len(active_subclass_split))
        setattr(dataset, 'coarse2fine', coarse2fine)
        setattr(dataset, 'coarse_targets', coarse_labels)
        setattr(dataset, 'idx_to_fine_class', idx_to_fine_class)
        return dataset

    def extract_c2f_from_dataset(self, dataset, coarse_custom_label_mapping, fine_custom_label_mapping, partition):
        classes, original_classes_to_idx = dataset._find_classes(os.path.join(self.data_dir, partition))
        _, coarse_classes_to_idx = coarse_custom_label_mapping(classes, original_classes_to_idx)
        _, fine_classes_to_idx = fine_custom_label_mapping(classes, original_classes_to_idx)
        coarse2fine = {}
        for k, v in coarse_classes_to_idx.items():
            if v in coarse2fine:
                coarse2fine[v].append(fine_classes_to_idx[k])
            else:
                coarse2fine[v] = [fine_classes_to_idx[k]]

        idx_to_fine_class = {v: k for k, v in fine_classes_to_idx.items()}

        fine2coarse = {}
        for k in coarse2fine:
            fine_labels_k = coarse2fine[k]
            for i in range(len(fine_labels_k)):
                assert fine_labels_k[i] not in fine2coarse
                fine2coarse[fine_labels_k[i]] = k

        fine_labels = dataset.targets
        coarse_labels = []
        for i in range(len(fine_labels)):
            coarse_labels.append(fine2coarse[fine_labels[i]])

        return coarse2fine, coarse_labels, idx_to_fine_class

    def get_classes(self, ds_name, split=None):
        if ds_name == 'living17':
            return make_living17(self.info_dir, split)
        elif ds_name == 'entity30':
            return make_entity30(self.info_dir, split)
        elif ds_name == 'entity13':
            return make_entity13(self.info_dir, split)
        elif ds_name == 'nonliving26':
            return make_nonliving26(self.info_dir, split)
        else:
            raise NotImplementedError


    def get_class_count(self, ds_name):
        if ds_name == 'living17':
            return (17, 68)
        elif ds_name == 'entity30':
            return (30, 240)
        elif ds_name == 'entity13':
            return (13, 260)
        elif ds_name == 'nonliving26':
            return (26, 104)
        else:
            raise NotImplementedError


class BREEDS(Dataset):
    def __init__(self, info_dir, data_dir, ds_name, partition, split, transform, train=True, seed=1000, tr_ratio=0.9):
        super(Dataset, self).__init__()
        breeds_factory = BREEDSFactory(info_dir, data_dir)
        self.dataset = breeds_factory.get_breeds(ds_name=ds_name,
                                                 partition=partition,
                                                 transforms=None,
                                                 split=split)
        self.num_coarse, self.num_fine = breeds_factory.get_class_count(ds_name)

        self.transform = transform
        self.loader = self.dataset.loader
        self.is_train = False

        images = [s[0] for s in self.dataset.samples]
        labels = self.dataset.targets
        coarse_labels = self.dataset.coarse_targets

        if partition == 'train':
            images, labels, coarse_labels = shuffle(images, labels, coarse_labels, random_state=seed)
            num_tr = int(len(images) * tr_ratio)

            if train is True:
                self.images = images[:num_tr]
                self.labels = labels[:num_tr]
                self.coarse_labels = coarse_labels[:num_tr]
                self.is_train = True
            else:
                self.images = images[num_tr:]
                self.labels = labels[num_tr:]
                self.coarse_labels = coarse_labels[num_tr:]
        else:
            self.images = images
            self.labels = labels
            self.coarse_labels = coarse_labels

        self.id2fine = self.load_names(os.path.join(info_dir, 'node_names.txt'))


    def load_names(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
        id2fine_name = dict()
        for line in lines:
            line = line.strip()
            id, name = line.split('\t')
            id2fine_name[id] = name
        id2fine = dict()
        for id, name in self.dataset.idx_to_fine_class.items():
            id2fine[id] = id2fine_name[name]
        return id2fine

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, coarse_target = self.images[index], self.labels[index], self.coarse_labels[index]
        img = self.loader(img)
        if self.transform is not None:
            img = self.transform(img)

        # return img, coarse_target, target

        return {
            'index': index,
            'inputs': img,
            'fine_label': target,
            'coarse_label': coarse_target,
        }

    def __len__(self):
        return len(self.images)

    def get_graph(self):
        M = np.zeros((self.num_fine, self.num_coarse))
        for coarse, fine_set in self.dataset.coarse2fine.items():
            for fine in fine_set:
                M[fine, coarse] = 1
        return M
