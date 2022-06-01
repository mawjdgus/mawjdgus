
import torch
import torch.nn as nn
import timm

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torch import optim

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import argparse
import random
import numpy as np
import os
from glob import glob

from tqdm import tqdm

import PIL.Image as Image

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
set_seed(2022)

parser = argparse.ArgumentParser(description='WDC2022')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--resize', default=128, type=int, help='transform size')
parser.add_argument('--dataname', default='DeepFake', type=str, help='data name')
args = parser.parse_args()

ckpt_path = glob(f'./ckpt/{args.dataname}/*.pth')[0]
print(f"ckpt_path : {ckpt_path}...")

transform_train = transforms.Compose([
                transforms.Resize((args.resize, args.resize)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

class train_dataset(Dataset):
    def __init__(self, root, method, resolution, transform, train=False, val=False, test=False):
        self.root = root
        self.transform = transform
        self.resolution = resolution # low_resolution
        if train:
            self.file_path = os.path.join(root, method, 'train', resolution)
        elif val:
            self.file_path = os.path.join(root, method, 'val', resolution)
        elif test:
            self.file_path = os.path.join(root, method, 'test', resolution)

        if 'fake' in self.resolution:
            self.label = 0
        else:
            self.label = 1

        self.files = os.listdir(os.path.join(self.file_path))
    
    def __getitem__(self, index):
        image = Image.open(os.path.join(self.file_path, self.files[index])).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, self.label

    def __len__(self):
        return len(self.files)

test_fake_dataset = train_dataset('/home/data/WDC2022', args.dataname, 'low_fake', transform=transform_train, test=True)
test_real_dataset = train_dataset('/home/data/WDC2022', args.dataname, 'low_real', transform=transform_train, test=True)

test_data = torch.utils.data.ConcatDataset([test_fake_dataset, test_real_dataset])

test_loader = DataLoader(test_data, args.batch_size, shuffle=True, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('xception', pretrained=True, num_classes=1).to(device)
model.load_state_dict(torch.load(ckpt_path))
print("model load complete...")

m = nn.Sigmoid()

model.eval()
test_loss, test_acc = 0, 0
precision, recall, f1, auc = [], [], [], []

with torch.no_grad():
    for batch_idx, data in enumerate(tqdm(test_loader)):
        inputs, targets = data

        inputs, targets = inputs.to(device), targets.to(device)

        output = m(model(inputs).squeeze())
        preds = (output >= 0.5).float()

        test_acc += (preds == targets).float().mean()

        # f1-score
        precision.append(precision_score(targets.cpu(), preds.cpu()))
        recall.append(recall_score(targets.cpu(), preds.cpu()))
        f1.append(f1_score(targets.cpu(), preds.cpu()))
        auc.append(roc_auc_score(targets.cpu(), preds.cpu()))

    test_acc /= len(test_loader)
        
    print('====TEST_ACC : {:.4f}===='.format(test_acc))
    print('====TEST_PRECISION : {:.4f}===='.format(np.mean(precision)))
    print('====TEST_RECALL : {:.4f}===='.format(np.mean(recall)))
    print('====TEST_F1 : {:.4f}===='.format(np.mean(f1)))
    print('====TEST_AUC : {:.4f}===='.format(np.mean(auc)))
    
    

    