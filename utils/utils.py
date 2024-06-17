import torch
from sklearn.cluster import KMeans
import math
import os

def entropy(p):
    return torch.special.entr(p).sum()
    # return - (p * p.log()).sum(-1)

def forward_feats(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    x = model.avgpool(x)
    x = torch.flatten(x, 1)

    return x


def freeze_backbone_layers(model):
    for name, param in model.named_parameters():
        if name not in ["fc.weight", "fc.bias", "fc.prototypes"]:
            param.requires_grad = False
        else:
            print(f"Fine-tuning {name}")
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):  # Adjust for your specific layer type
            layer.track_running_stats = False

def cosine_annealing(initial_value, min_value, cur_step, total_steps):
    assert min_value <= initial_value
    assert cur_step <= total_steps
    return min_value + (initial_value - min_value) * 0.5 * (1 + math.cos(math.pi * cur_step / total_steps))

def load_gt_adjecency(wadjm, gt_matrix):
    wadjm.data = torch.from_numpy(gt_matrix).float().cuda()
    wadjm.requires_grad = False
    return wadjm

def plot_confusion(cfg, pred, actual, epoch):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(actual, pred)
    sns.heatmap(cm, annot=False)
    plt.title(f'Epoch {epoch}')
    plt.savefig(os.path.join(cfg.OUTPUT_DIR, 'plots', f'confusion_{epoch}.png'))
    plt.clf()
    plt.close()
