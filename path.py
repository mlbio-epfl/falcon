import os
def has_env_var(var_name):
    return var_name in os.environ.keys() and os.environ[var_name] is not None

dataset_paths = {
    'CIFAR100': '/home/shared/datasets/CIFAR100/cifar-100-python' if not has_env_var('CIFAR100_ROOT') else os.environ['CIFAR100_ROOT'],
    'ImageNet': '/mnt/sda1/ImageNet_dataset/ILSVRC/' if not has_env_var('IMAGENET_ROOT') else os.environ['IMAGENET_ROOT'],
    'PBMC': '/home/shared/datasets/PBMC' if not has_env_var('PBMC_ROOT') else os.environ['PBMC_ROOT'],
}