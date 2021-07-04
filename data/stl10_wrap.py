import random

import torch
from torchvision import datasets
from torch.utils import data
from torchvision.transforms import transforms

import time

class STL10DatasetWrap(data.Dataset):
    def __init__(self, data_transform, root_dir='/home/work/Datasets/', split='train+unlabeled'):
        self.dataset = datasets.STL10(root_dir, split=split, download=True)

        self.transform = data_transform
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, index):
        other_index = random.randint(0, len(self.dataset) - 1)

        img = self.dataset[index][0]
        img2 = self.dataset[other_index][0]

        img1_view1, img2_view1 = self.transform(torch.stack((self.to_tensor(img), self.to_tensor(img2))))
        img1_view2, img2_view2 = self.transform(torch.stack((self.to_tensor(img), self.to_tensor(img2))))

        return img1_view1, img1_view2, img2_view1, img2_view2

    def __len__(self):
        return len(self.dataset)