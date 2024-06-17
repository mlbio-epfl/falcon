from torchvision.models import resnet50, resnet18
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch

def _load_selfsup_from_url(model, url):
    print("Loading weights", url)
    d = load_state_dict_from_url(url)
    d = {k.replace('module.encoder_q.', ''): v for k, v in d['state_dict'].items()}
    d = {k.replace('module.base_encoder.', ''): v for k, v in d.items()}

    filtered_items = filter(lambda item: 'module' not in item[0], d.items())
    d = dict(filtered_items)

    out = model.load_state_dict(d, strict=False)
    print(f"Loading pretrained weights result: {out[0]}")
    return model


def _load_selfsup(model, path):
    print("Loading weights", path)
    dict = torch.load(path, map_location='cpu')
    dict = {k.replace('module.encoder_q.', ''): v for k, v in dict['state_dict'].items()}
    dict = {k.replace('backbone.', ''): v for k, v in dict.items()}
    if 'fc.weight' in dict:
        del dict['fc.weight']
    if 'fc.bias' in dict:
        del dict['fc.bias']
    state = {}
    for k, v in dict.items():
        if 'encoder_k' not in k:
            state[k] = v
    out = model.load_state_dict(state, strict=False)
    print(f"Loading pretrained weights result: {out[0]}")
    return model


def construct_resnet50(pretrained=False):
    model = resnet50()
    if pretrained:
        model = _load_selfsup(model, pretrained) if 'https' not in pretrained else _load_selfsup_from_url(model,
                                                                                                          pretrained)
    embed_dim = model.fc.weight.shape[1]
    # model.fc = nn.Linear(embed_dim, num_classes)
    model.fc = nn.Identity()
    return model, embed_dim
    # return ResNetBase(model)


def construct_resnet18(pretrained=False):
    model = resnet18()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
    model.maxpool = nn.Identity()
    if pretrained:
        model = _load_selfsup(model, pretrained) if 'https' not in pretrained else _load_selfsup_from_url(model,
                                                                                                          pretrained)
        # model = _load_from_url(model, 'https://download.pytorch.org/models/resnet50-11ad3fa6.pth')
    embed_dim = 512
    # model.fc = nn.Linear(embed_dim, num_classes)
    model.fc = nn.Identity()
    return model, embed_dim

