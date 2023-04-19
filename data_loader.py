from torchvision import transforms
import torch
import torchvision
import numpy as np
from PIL import Image

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
from skimage import io
os.environ['KMP_DUPLICATE_LIB_OK']='True'

IMG_SIZE = 64
BATCH_SIZE = 128

data_path = "C:\\Users\\Seunghwi\\Documents\\Diffusion\\archive\\Humans"

class FaceDataset(Dataset):
    def __init__(self,root_dir,transform=None):
        self.annotaions = os.listdir(data_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotaions)

    def __getitem__(self,index):
        img_path = os.path.join(self.root_dir, self.annotaions[0])
        pil_image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            pil_image = self.transform(pil_image)
        return pil_image


def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1] 
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1] 
    ]
    data_transform = transforms.Compose(data_transforms)

    dataset = FaceDataset(root_dir=data_path, transform=data_transform)

    train, test = torch.utils.data.random_split(dataset,[7000,220])

    return torch.utils.data.ConcatDataset([train, test])

