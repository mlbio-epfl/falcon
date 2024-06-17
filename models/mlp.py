import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, hidden_dim, input_dim, output_dim, num_classes, num_layers=2):
        super().__init__()
        self.feature_extractor = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self.feature_extractor.add_module(f"linear_{i}", nn.Linear(input_dim, hidden_dim))
            elif i == num_layers - 1:
                self.feature_extractor.add_module(f"linear_{i}", nn.Linear(hidden_dim, output_dim))
            else:
                self.feature_extractor.add_module(f"linear_{i}", nn.Linear(hidden_dim, hidden_dim))
            self.feature_extractor.add_module(f"relu_{i}", nn.ReLU())

        self.fc = nn.Linear(output_dim, num_classes) if num_classes is not None else nn.Identity()

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x

def construct_mlp(cfg):
    def model_fn(pretrained=False):
        return MLP(cfg.MODEL.HIDDEN_DIM, cfg.MODEL.INPUT_DIM,
            cfg.MODEL.OUTPUT_DIM, cfg.MODEL.NUM_CLASSES,
            cfg.MODEL.NUM_LAYERS), cfg.MODEL.OUTPUT_DIM

    return model_fn