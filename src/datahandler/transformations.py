import albumentations as A
from absl import logging
from torch import nn
from ast import literal_eval
from torchvision import transforms as tfs
from albumentations.pytorch import ToTensorV2
from src.utils import NORMALIZATION_MEAN, NORMALIZATION_STD


def select_transforms(args):
    """Create torchvision.transforms

    Args:
        args (configdct.ConfigDict): Input image size and type of augmentation

    Raises:
        Exception: Throws exception if args.augment_type is not any one of predefined values. 

    Returns:
        torchvision.transforms.Compose: List of selected PyTorch transformations.
    """
    try:
        inp_size = literal_eval(args.input_size)
    except ValueError:
        pass
    
    augment_type = str(args.augment_type)

    if augment_type == "0":
        transforms = basic_transforms(inp_size)
    elif augment_type == "1":
        transforms = augment_one(inp_size)
    elif augment_type == "2gray":
        transforms = augment_two_gray(inp_size)
    elif augment_type == "2":
        transforms = augment_two(inp_size)
    else:
        raise Exception("Unknown augment type")
    
    return transforms

def augment_one(input_size):
    """Create augmentation Set 1

    Args:
        input_size (tuple or list or int): height and width of an image; if int then image will be resized to a square.

    Returns:
        dictionary : Dictionary with train and val augmentations
    """
    input_size = input_size if type(input_size) in [list, tuple] else (input_size, input_size)
    logging.log(logging.INFO, "IMAGENET NORMALIZATION")
    return {
        'train': A.Compose([
            A.SmallestMaxSize(max_size=160),
            # A.Resize(height = input_size[0], width=input_size[1]),
            A.Resize(*input_size),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            # A.RandomCrop(height=128, width=128),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.HorizontalFlip(p=0.25),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD),
            ToTensorV2(),
        ]),
        'val': A.Compose([
            A.Resize(*input_size),
            A.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
            ToTensorV2(),
        ]),
        'val_augmentations': ['imagenet_normalization']
    }

def augment_two_gray(input_size):
    """Create augmentation Set 2 in Grayscale

    Args:
        input_size (tuple or list or int): height and width of an image; if int then image will be resized to a square.

    Returns:
        dictionary : Dictionary with train and val augmentations
    """

    input_size = input_size if type(input_size) in [list, tuple] else (input_size, input_size)
    return {
        "val": tfs.Compose([
                       tfs.Resize(size=input_size),
                       tfs.ToTensor(),
                       tfs.Grayscale(num_output_channels=3)
                ]),
        'val_augmentations': ['grayscale'],
        "train": tfs.Compose([
                       tfs.Resize(size=input_size),
                       tfs.RandomApply(nn.ModuleList([
                           tfs.ColorJitter(brightness=.5, hue=.3),
                       ]), p=0.1),
                       tfs.RandomHorizontalFlip(p=0.3),
                       tfs.RandomApply(nn.ModuleList([
                           tfs.RandomRotation(180),
                       ]), p=0.3),
                       tfs.ToTensor(),
                       tfs.Grayscale(num_output_channels=3)
                       ])
        }

def augment_two(input_size):
    """Create augmentation Set 2

    Args:
        input_size (tuple or list or int): height and width of an image; if int then image will be resized to a square.

    Returns:
        dictionary : Dictionary with train and val augmentations
    """
    input_size = input_size if type(input_size) in [list, tuple] else (input_size, input_size)
    return {
        "val": tfs.Compose([
                       tfs.Resize(size=input_size),
                       tfs.ToTensor(),
                ]),
        'val_augmentations': [],
        "train": tfs.Compose([
                       tfs.Resize(size=input_size),
                       tfs.RandomApply(nn.ModuleList([
                           tfs.ColorJitter(brightness=.5, hue=.3),
                       ]), p=0.1),
                       tfs.RandomHorizontalFlip(p=0.3),
                       tfs.RandomPerspective(distortion_scale=0.3, p=0.2),
                       tfs.RandomApply(nn.ModuleList([
                           tfs.RandomRotation(180),
                       ]), p=0.3),
                       tfs.ToTensor(),
                       tfs.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
                       # tfs.RandomErasing(p=0.25, scale=(0.02, 0.02), value=1) Random Erasing?
                       ])
        }

def basic_transforms(input_size):
    """Create augmentation Set 0

    Args:
        input_size (tuple or list or int): height and width of an image; if int then image will be resized to a square.

    Returns:
        dictionary : Dictionary with train and val augmentations
    """
    input_size = input_size if type(input_size) in [list, tuple] else (input_size, input_size)
    return {
        'train': tfs.Compose([
            # tfs.RandomResizedCrop(input_size),
            tfs.Resize(input_size),
            tfs.RandomHorizontalFlip(),
            tfs.ToTensor(),
            # tfs.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
        ]),
        'val': tfs.Compose([
            tfs.Resize(input_size),
            tfs.ToTensor(),
        ]),
        'val_augmentations': [],
    }