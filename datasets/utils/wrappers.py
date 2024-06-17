from torch.utils.data import Dataset, Subset
import numpy as np
import torch


class NeighborsWrapper(Dataset):
    def __init__(self, dataset, neighbors, num_neighbors):
        self.dataset = dataset
        self.neighbors = neighbors
        self.num_neighbors = num_neighbors
        assert num_neighbors <= neighbors.shape[-1]

    def _extract_img(self, idx):
        x = self.dataset[idx]['inputs']
        if isinstance(x, list) or isinstance(x, tuple):
            return x[1]
        else:
            return x

    def __getitem__(self, idx):
        neighbors = torch.cat([
            self._extract_img(id).unsqueeze(0) for id in np.random.choice(self.neighbors[idx], self.num_neighbors, replace=False)
        ], dim=0)
        out = self.dataset[idx]
        out['neighbors'] = neighbors
        return out

    def __len__(self):
        return len(self.dataset)
