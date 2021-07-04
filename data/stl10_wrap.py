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
        other_img = self.dataset[other_index][0]

        view_1 = self.transform(self.to_tensor(img))
        view_2, other_img = self.transform(torch.stack((self.to_tensor(img), self.to_tensor(other_img))))

        return view_1, view_2, other_img

    def __len__(self):
        return len(self.dataset)