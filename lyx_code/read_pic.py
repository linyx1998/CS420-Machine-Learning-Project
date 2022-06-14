import numpy as np
import os
from torch.utils.data import TensorDataset
import torch
import torch.utils.data
from torchvision import transforms
from torchvision.datasets import ImageFolder

def read_data_pic():
    data_folder = "../pic_dataset/"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,), (0.5,))
    ])
    train_dataset = ImageFolder(data_folder + 'train', transform)
    valid_dataset = ImageFolder(data_folder + 'valid', transform)
    test_dataset = ImageFolder(data_folder + 'test', transform)

    return train_dataset, valid_dataset, test_dataset