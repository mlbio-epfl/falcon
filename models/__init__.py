from .resnet import construct_resnet50, construct_resnet18
from .heads import HypersphereHead, GaussianHead, StudentHead
from .mlp import construct_mlp
import torch.nn as nn

def create_backbone_factory(cfg):
    if cfg.MODEL.BACKBONE_NAME == 'ResNet18':
        return construct_resnet18
    elif cfg.MODEL.BACKBONE_NAME == 'ResNet50':
        return construct_resnet50
    elif cfg.MODEL.BACKBONE_NAME == 'MDISCRETE_OPTIM':
        return construct_mlp(cfg)
    else:
        raise NotImplementedError('Model not implemented: {}'.format(cfg.MODEL.BACKBONE_NAME))

def construct_classifier(head_type, model, embed_dim, num_classes):
    if head_type == 'Hypersphere':
        model.fc = HypersphereHead(num_class=num_classes, embed_dim=embed_dim)
        return model
    elif head_type == 'Gaussian':
        model.fc = GaussianHead(num_class=num_classes, embed_dim=embed_dim)
        return model
    elif head_type == 'Student':
        model.fc = StudentHead(num_class=num_classes, embed_dim=embed_dim)
        return model
    elif head_type == 'Linear':
        model.fc = nn.Linear(embed_dim, num_classes, bias=True)
        return model
    else:
        raise NotImplementedError(f'Classifier not implemented: {head_type}')
