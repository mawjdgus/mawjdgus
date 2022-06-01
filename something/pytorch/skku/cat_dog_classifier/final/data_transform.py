


import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class TrainDataset(Dataset):
    def __init__(self, files, root, transform):
        self.files = files
        self.root = root
        self.transform = transform

        if 'cat' in files[0]:
            self.label = 0
        else:
            self.label = 1

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.files[index]))

        if self.transform:
            image = self.transform(image)
        return image, self.label

    def __len__(self):
        return len(self.files)

class TestDataset(Dataset):
    def __init__(self, files, root, transform):
        self.files = files
        self.root = root
        self.transform = transform


    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.files[index]))

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.files)

