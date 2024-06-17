import torch

class Taxonomy:
    def __init__(self, class_groups):
        self.num_classes = len(class_groups)
        self.name2id = dict()
        for idx, items in enumerate(class_groups):
            for item in items:
                if type(item) is list:
                    for subitem in item:
                        self.name2id[subitem] = idx
                else:
                    self.name2id[item] = idx

    def __getitem__(self, name):
        return self.get_id(name)

    def get_id(self, name):
        return self.name2id[name]

    def get_class_count(self):
        return self.num_classes

    def get_graph(self):
        M = torch.zeros(len(self.name2id.keys()), self.num_classes)
        for fine_id, (name, coarse_id) in enumerate(self.name2id.items()):
            M[fine_id, coarse_id] = 1
        return M

    def __str__(self):
        return str(self.name2id)

    def __len__(self):
        return self.num_classes