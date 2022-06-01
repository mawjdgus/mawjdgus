

import os
import zipfile
from PIL import Image
import time
from torch.optim import optimizer
from torchsummary import summary

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.cuda.amp import grad_scaler, autocast_mode


from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from sklearn.model_selection import StratifiedKFold

torch.manual_seed(1211)

# data_zip_dir = '/opt/ml/skku/dog_classifier'
# train_zip_dir = os.path.join(data_zip_dir, 'train.zip')
# test_zip_dir = os.path.join(data_zip_dir, 'test.zip')

# with zipfile.ZipFile(train_zip_dir, 'r') as z:
#     z.extractall()
# with zipfile.ZipFile(test_zip_dir, 'r') as z:
#     z.extractall()

# train_dir = os.path.join('/opt/ml/skku/dog_classifier', 'train')
# test_dir = os.path.join('/opt/ml/skku/dog_classifier', 'test')



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


# # Depthwise Separable Convolution
# class SeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()

#         self.seperable = nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False),
#             nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
#         )

#     def forward(self, x):
#         x = self.seperable(x)
#         return x

# # EnrtyFlow
# class EntryFlow(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, 3, stride=1, padding=0, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )

#         self.conv2_residual = nn.Sequential(
#             SeparableConv(64, 128),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             SeparableConv(128, 128),
#             nn.BatchNorm2d(128),
#             nn.MaxPool2d(3, stride=2, padding=1)
#         )

#         self.conv2_shortcut = nn.Sequential(
#             nn.Conv2d(64, 128, 1, stride=2, padding=0),
#             nn.BatchNorm2d(128)
#         )

#         self.conv3_residual = nn.Sequential(
#             nn.ReLU(),
#             SeparableConv(128, 256),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             SeparableConv(256, 256),
#             nn.BatchNorm2d(256),
#             nn.MaxPool2d(3, stride=2, padding=1)
#         )

#         self.conv3_shortcut = nn.Sequential(
#             nn.Conv2d(128, 256, 1, stride=2, padding=0),
#             nn.BatchNorm2d(256)
#         )

#         self.conv4_residual = nn.Sequential(
#             nn.ReLU(),
#             SeparableConv(256, 728),
#             nn.BatchNorm2d(728),
#             nn.ReLU(),
#             SeparableConv(728, 728),
#             nn.BatchNorm2d(728),
#             nn.MaxPool2d(3, stride=2, padding=1)
#         )

#         self.conv4_shortcut = nn.Sequential(
#             nn.Conv2d(256, 728, 1, stride=2, padding=0),
#             nn.BatchNorm2d(728)
#         )

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2_residual(x) + self.conv2_shortcut(x)
#         x = self.conv3_residual(x) + self.conv3_shortcut(x)
#         x = self.conv4_residual(x) + self.conv4_shortcut(x)
#         return x


# # MiddleFlow
# class MiddleFlow(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.conv_residual = nn.Sequential(
#             nn.ReLU(),
#             SeparableConv(728, 728),
#             nn.BatchNorm2d(728),
#             nn.ReLU(),
#             SeparableConv(728, 728),
#             nn.BatchNorm2d(728),
#             nn.ReLU(),
#             SeparableConv(728, 728),
#             nn.BatchNorm2d(728)
#         )

#         self.conv_shortcut = nn.Sequential()

#     def forward(self, x):
#         return self.conv_shortcut(x) + self.conv_residual(x)


# # ExitFlow
# class ExitFlow(nn.Module):
#     def __init__(self, num_classes=10):
#         super().__init__()

#         self.conv1_residual = nn.Sequential(
#             nn.ReLU(),
#             SeparableConv(728, 1024),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(),
#             SeparableConv(1024, 1024),
#             nn.BatchNorm2d(1024),
#             nn.MaxPool2d(3, stride=2, padding=1)
#         )

#         self.conv1_shortcut = nn.Sequential(
#             nn.Conv2d(728, 1024, 1, stride=2, padding=0),
#             nn.BatchNorm2d(1024)
#         )

#         self.conv2 = nn.Sequential(
#             SeparableConv(1024, 1536),
#             nn.BatchNorm2d(1536),
#             nn.ReLU(),
#             SeparableConv(1536, 2048),
#             nn.BatchNorm2d(2048),
#             nn.ReLU()
#         )

#         self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
    
#     def forward(self, x):
#         x = self.conv1_residual(x) + self.conv1_shortcut(x)
#         x = self.conv2(x)
#         x = self.avg_pool(x)
#         return x




# # Xception
# class Xception(nn.Module):
#     def __init__(self, num_classes=2, init_weights=True):
#         super().__init__()
#         self.init_weights = init_weights

#         self.entry = EntryFlow()
#         self.middle = self._make_middle_flow()
#         self.exit = ExitFlow()

#         self.linear = nn.Linear(2048, num_classes)

#         def init_weight(m):
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.xavier_uniform(m.weight)
                

#         # weights initialization
#         if self.init_weights:
#             self.entry.conv1.apply(init_weight)
        
        
    

    



#     def forward(self, x):
#         x = self.entry(x)
#         x = self.middle(x)
#         x = self.exit(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear(x)
#         return x

#     def _make_middle_flow(self):
#         middle = nn.Sequential()
#         for i in range(8):
#             middle.add_module('middle_block_{}'.format(i), MiddleFlow())
#         return middle

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init_kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init_constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init_constant_(m.weight, 1)
#                 nn.init_bias_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init_normal_(m.weight, 0, 0.01)
#                 nn.init_constant_(m.bias, 0)

# # check model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# x = torch.randn(3, 3, 299, 299).to(device)
# model = Xception().to(device)
# output = model(x)
# print('output size:', output.size())

# # print summary
# summary(model, (3, 299, 299), device=device.type)

import timm
from tqdm import tqdm
###########


def rand_bbox(size, lam):
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)


    cx = np.random.randn() + W//2
    cy = np.random.randn() + H//2

    # 패치의 4점
    bbx1 = np.clip(cx - cut_w // 2, 0, W//2)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W//2)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return int(bbx1), int(bby1), int(bbx2), int(bby2)


if __name__=='__main__':

    k_folds = 5
    num_epochs = 50
    

    # For fold results
    results = {}

    # Set fixed random number seed


    dog_files = [f'dog.{i}.jpg' for i in range(12500)]
    cat_files = [f'cat.{i}.jpg' for i in range(12500)]
    test_files = [f'{i}.jpg' for i in range(1,12500)]
    
    train_transform = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.RandomAffine(degrees=20, translate=None, scale=None, shear=None, resample=False, fillcolor=0),
        transforms.RandomCrop(255),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    test_transform = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])


    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)

    # Start print
    print('--------------------------------')

    train_dog_dataset = TrainDataset(dog_files, '/opt/ml/train', train_transform)
    train_cat_dataset = TrainDataset(cat_files, '/opt/ml/train', train_transform)
    # test_dataset = TestDataset(test_files, '/opt/ml/test1', test_transform)

    train_dataset = ConcatDataset([train_dog_dataset, train_cat_dataset])
    target_dog = torch.ones(len(dog_files))
    target_cat = torch.zeros(len(cat_files))
    target = torch.cat([target_dog,target_cat])
    
    

    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset,target)):
        torch.cuda.empty_cache()
        model = timm.create_model('xception', pretrained=True, num_classes=2).to(device)
        # model = timm.create_model('xception', pretrained=True, num_classes=2)
        # model.to(device)

        print(train_ids, val_ids)
        print(f'FOLD {fold}')
        print('-----------------------------------')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)


        train_loader = DataLoader(train_dataset, batch_size=192, sampler=train_subsampler, drop_last=True)
        val_loader = DataLoader(train_dataset, batch_size=192, sampler=val_subsampler, drop_last=True)
        # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        criterion = nn.CrossEntropyLoss(reduction='sum')
        opt = optim.Adam(model.parameters(), lr=0.0002)

        scaler = grad_scaler.GradScaler()


        for epoch in range(20):
            print(f'=====EPOCH : {epoch}=====')
            model.train()
            train_loss = 0
            train_acc = 0
            val_loss =0
            val_acc =0
            epoch_loss = 0
            epoch_acc = 0
            epoch_val_loss = 0
            epoch_val_acc = 0
            

            for i, data in enumerate(tqdm(train_loader)):
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                opt.zero_grad()
                with autocast_mode.autocast():

                    if np.random.random() > 0.5: # Cutmix
                        random_index = torch.randperm(inputs.size()[0])
                        target_a = targets
                        targeb_b = targets[random_index]

                        lam = np.random.beta(1.0, 1.0)
                        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)

                        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[random_index, :, bbx1:bbx2, bby1:bby2]
                        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))

                        pred = model(inputs.float())
                        loss = criterion(pred, target_a) * lam + criterion(pred, targeb_b) * (1. - lam)

                        _, preds = torch.max(pred, 1)
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()

                    else:
                        pred = model(inputs)
                        loss = criterion(pred, targets)

                        _,preds = torch.max(pred, 1)

                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                

                    train_loss += loss.item()
                    train_acc += torch.sum(preds == targets.data)
            epoch_loss = train_loss / len(train_loader.dataset)
            epoch_acc = train_acc / len(train_loader.dataset)
            print(epoch_loss)
            print(epoch_acc)

        # Iterate over the test data and generate predictions
            with torch.no_grad():
                # Iterate over the test data and generate predictions
                for i, data in enumerate(val_loader, 0):
                    model.eval()
                    inputs, targets = data
                    inputs, targets = inputs.to(device), targets.to(device)


                    pred = model(inputs)
                    loss = criterion(pred, targets)
                    # Set total and correct
                    _, preds= torch.max(pred, 1)
                    val_loss += loss.item()
                    val_acc += torch.sum(preds == targets.data)
                epoch_val_loss = val_loss / len(val_loader.dataset)
                epoch_val_acc = val_acc / len(val_loader.dataset)
                print(epoch_val_loss)
                print(epoch_val_acc)
                # Print accuracy
            print('Accuracy for fold %d: %d %%' % (fold, 100.0 * epoch_val_acc))
            print('--------------------------------')
            results[fold] = 100.0 * (epoch_val_acc)



        # Print fold results
        print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
        print('--------------------------------')
        sums = 0.0
        for key, value in results.items():
            print(f'Fold {key}: {value} %')
            sums += value
        print(f'Average: {sums/len(results.items())} %')
        torch.save(model.state_dict(), f'/opt/ml/skku/dog_classifier/Xception_fold_{fold}.pt')



    # samples, labels = iter(train_loader).next()

    # classes = {0:'cat', 1:'dog'}
    # fig = plt.figure(figsize=(10,10))
    # for i in range(25):
    #     a = fig.add_subplot(5, 5, i+1)
    #     a.set_title(classes[labels[i].item()])
    #     a.axis('off')
    #     a.imshow(np.transpose(samples[i].numpy(), (1,2,0)))
    # plt.subplots_adjust(bottom=0.2, top=0.6, hspace=0)



