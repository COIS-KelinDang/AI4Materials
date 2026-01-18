import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import copy
import random


class TrainGANDataset(Dataset):
    def __init__(self, image_dir, img_size, train=False):
        self.image_dir = image_dir
        self.img_size = img_size
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.tif', '.jpg', '.png', '.tiff', 'jpeg'))])
        self.train = train
        self.transform_train = transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                               transforms.Resize(size=(self.img_size, self.img_size)),
                               transforms.RandomHorizontalFlip()])
        self.transform_test = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                              transforms.Resize(size=(self.img_size, self.img_size))])

    def __len__(self):
        return len(os.listdir(self.image_dir))

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        return self.transform_train(image) if self.train else self.transform_test(image)
