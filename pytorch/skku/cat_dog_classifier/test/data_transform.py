


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


class train_transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((299,299)),
            transforms.RandomAffine(degrees=20, translate=None, scale=None, shear=None, resample=False, fillcolor=0),
            transforms.RandomCrop(255),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    def __call__(self):
        return self.transform



# test_transform = transforms.Compose([
#         transforms.Resize((299,299)),
#         transforms.RandomHorizontalFlip(0.5),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
#     ])
