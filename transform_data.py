from typing import List, Tuple, Dict, Optional

import torch
import torchvision
from torch import nn, Tensor
from torchvision.transforms.v2 import functional as F
from torchvision.transforms import transforms as T
import copy


def MyRandomHorizontalFlip(image, target, p = 0.5):
    if torch.rand(1) < p:
        imagef = F.hflip(image)
        width = image.shape[2]
        targetf = copy.deepcopy(target)
        targetf["boxes"][:, [0, 2]] = width - targetf["boxes"][:, [2, 0]]
        return imagef, targetf
    else:
        return image, target



def MyRandomVericalFlip(image, target, p = 0.5):
    if torch.rand(1) < p:
        #print("flipped!")
        imagef = F.vflip(image)
        height = image.shape[1]
        targetf = copy.deepcopy(target)
        targetf["boxes"][:, [1, 3]] = height - targetf["boxes"][:, [3, 1]]
        return imagef, targetf
    else:
        return image, target

def MyToTensor(image, target):
    image = F.pil_to_tensor(image)
    image = F.convert_image_dtype(image)
    return image, target


