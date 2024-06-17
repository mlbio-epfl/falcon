import torch
from torch.utils.data import Dataset

class SingleCellData(Dataset):
    def __init__(self, file_path, two_views=False):
        self.file_path = file_path
        self.two_views = two_views
        print(f'Loading single cell data from {self.file_path}')

        self.inputs, self.labels, self.coarse_labels = SingleCellData.load_data(self.file_path)

        self.num_fine = int(self.labels.unique().shape[0])
        self.num_coarse = int(self.coarse_labels.unique().shape[0])

    @staticmethod
    def load_data(file_path):
        data_obj = torch.load(file_path)
        X = data_obj['inputs']
        fine_labels = data_obj['fine_labels']
        coarse_labels = data_obj['coarse_labels']
        return X, fine_labels, coarse_labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        input = self.inputs[item]

        return {
            'index': item,
            'inputs': input if not self.two_views else [input, input],
            'fine_label': self.labels[item],
            'coarse_label': self.coarse_labels[item],
        }

    def get_graph(self):
        M = torch.zeros((self.num_fine, self.num_coarse))
        for coarse, fine in zip(self.coarse_labels, self.labels):
            M[fine, coarse] = 1
        assert torch.sum(M) == self.num_fine
        return M.numpy()


