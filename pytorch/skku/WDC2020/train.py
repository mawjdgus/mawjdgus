
import torch
import torch.nn as nn
import timm

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from torch import optim

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

import argparse
import random
import numpy as np
import os

from tqdm import tqdm

import PIL.Image as Image

import wandb

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

wandb.init(project="workshop_highhigh_taejune",  entity="taemo")

wandb.config = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 128
}
wandb.run.name = args.dataname

transform_train = transforms.Compose([
                transforms.Resize((args.resize, args.resize)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

class train_dataset(Dataset):
    def __init__(self, root, method, resolution, transform, train=True):
        self.root = root
        self.transform = transform
        self.resolution = resolution # low_resolution
        if train:
            self.file_path = os.path.join(root, method, 'train', resolution)
        else:
            self.file_path = os.path.join(root, method, 'val', resolution)

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

fake_dataset = train_dataset('/home/data/WDC2022/', args.dataname, 'high_fake', transform=transform_train)
real_dataset = train_dataset('/home/data/WDC2022/', args.dataname, 'high_real', transform=transform_train)

train_data = torch.utils.data.ConcatDataset([fake_dataset, real_dataset])

val_fake_dataset = train_dataset('/home/data/WDC2022', args.dataname, 'high_fake', transform=transform_train, train=False)
val_real_dataset = train_dataset('/home/data/WDC2022', args.dataname, 'high_real', transform=transform_train, train=False)

val_data = torch.utils.data.ConcatDataset([val_fake_dataset, val_real_dataset])

train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, args.batch_size, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = timm.create_model('xception', pretrained=True, num_classes=1).to(device)

m = nn.Sigmoid()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

best_val_f1 = 0
best_epoch = 0
best_val_acc = 0

for epoch in range(num_epochs):
    print(f'======EPOCH : {epoch}======')
    model.train()

    train_precision = []
    train_recall = []
    train_f1 = []

    val_loss = 0
    val_acc = 0
    val_precision = []
    val_recall = []
    val_f1 = []

    train_epoch_loss, val_epoch_loss = 0, 0
    train_epoch_acc, val_epoch_acc = 0, 0

    for batch_idx, data in enumerate(tqdm(train_loader)):
        inputs, targets = data
        
        # targets = targets.unsqueeze(1).type(torch.FloatTensor)
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()

        output = m(model(inputs).squeeze())
        loss = criterion(output, targets.to(torch.float32))
        preds = (output >= 0.5).float()
        
        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.item()
        train_epoch_acc += (preds == targets).float().mean()
        
        # f1-score
        train_precision.append(precision_score(targets.to('cpu'), preds.to('cpu')))
        train_recall.append(recall_score(targets.to('cpu'), preds.to('cpu')))
        train_f1.append(f1_score(targets.to('cpu'), preds.to('cpu')))
        
    train_epoch_loss /= len(train_loader)
    train_epoch_acc /= len(train_loader)
        
    print('====EPOCH_LOSS : {:.4f}===='.format(train_epoch_loss))
    print('====EPOCH_ACC : {:.4f}===='.format(train_epoch_acc))
    print('====EPOCH_PRECISION : {:.4f}===='.format(np.mean(train_precision)))
    print('====EPOCH_RECALL : {:.4f}===='.format(np.mean(train_recall)))
    print('====EPOCH_F1 : {:.4f}===='.format(np.mean(train_f1)))

    wandb.log({"EPOCH_LOSS": train_epoch_loss})
    wandb.log({"EPOCH_ACCURACY": train_epoch_acc})
    wandb.log({"EPOCH_PRECISION": np.mean(train_precision)})
    wandb.log({"EPOCH_RECALL": np.mean(train_recall)})
    wandb.log({"EPOCH_F1": np.mean(train_f1)})

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(val_loader)):
            inputs, targets = data

            # targets = targets.unsqueeze(1).type(torch.FloatTensor)
            inputs, targets = inputs.to(device), targets.to(device)

            output = m(model(inputs).squeeze())
            loss = criterion(output, targets.to(torch.float32))
            preds = (output >= 0.5).float()

            val_epoch_loss += loss.item()
            val_epoch_acc += (preds == targets).float().mean()

            # f1-score
            val_precision.append(precision_score(targets.to('cpu'), preds.to('cpu')))
            val_recall.append(recall_score(targets.to('cpu'), preds.to('cpu')))
            val_f1.append(f1_score(targets.to('cpu'), preds.to('cpu')))
        
        current_f1 = np.mean(val_f1)
        val_epoch_loss /= len(val_loader)
        val_epoch_acc /= len(val_loader)
        
        print('====VAL_EPOCH_LOSS : {:.4f}===='.format(val_epoch_loss))
        print('====VAL_EPOCH_ACC : {:.4f}===='.format(val_epoch_acc))
        print('====VAL_PRECISION : {:.4f}===='.format(np.mean(val_precision)))
        print('====VAL_RECALL : {:.4f}===='.format(np.mean(val_recall)))
        print('====VAL_F1 : {:.4f}===='.format(np.mean(val_f1)))

        wandb.log({'VAL_EPOCH_LOSS' : val_epoch_loss})
        wandb.log({'VAL_EPOCH_ACC' : val_epoch_acc})    
        wandb.log({'VAL_PRECISION' : np.mean(val_precision)})
        wandb.log({'VAL_RECALL' : np.mean(val_recall)})
        wandb.log({'VAL_F1' : np.mean(val_f1)})
    
    if best_val_acc < val_epoch_acc:
        if epoch == 0:
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), os.path.join('./ckpt', args.dataname, f'best_model_epoch_{epoch}.pth'))
            print("model saved!!")
        else:
            os.remove(os.path.join('./ckpt', args.dataname, f'best_model_epoch_{best_epoch}.pth'))
            best_epoch=epoch
            best_val_acc = val_epoch_acc
            torch.save(model.state_dict(), os.path.join('./ckpt', args.dataname, f'best_model_epoch_{best_epoch}.pth'))
            print("model_updated!!")

