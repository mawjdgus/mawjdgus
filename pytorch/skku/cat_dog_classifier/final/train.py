


import numpy as np
import random
import timm



import torch
import torch.nn as nn

from torch import optim
from torch.cuda.amp import grad_scaler, autocast_mode
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from sklearn.model_selection import StratifiedKFold

from data_transform import TrainDataset


import timm
from tqdm import tqdm


####################################################

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)




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



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__=='__main__':

    k_folds = 5
    num_epochs = 1
    

    # For fold results
    results = {}

    # Set fixed random number seed


    dog_files = [f'dog.{i}.jpg' for i in range(12500)]
    cat_files = [f'cat.{i}.jpg' for i in range(12500)]
    test_files = [f'{i}.jpg' for i in range(1,12500)]
    all_files = dog_files + cat_files
    

    train_transform = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.RandomAffine(degrees=20, translate=None, scale=None, shear=None, resample=False, fillcolor=0),
        transforms.RandomCrop(255),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    stratified_kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)

    # Start print
    print('--------------------------------')
    
    target_dog = torch.ones(len(dog_files))
    target_cat = torch.zeros(len(cat_files))
    target = torch.cat([target_dog,target_cat])



    for fold, (train_ids, val_ids) in enumerate(stratified_kfold.split(all_files,target)):
        torch.cuda.empty_cache()
        model = timm.create_model('xception', pretrained=True, num_classes=2).to(device)
        # model = timm.create_model('xception', pretrained=True, num_classes=2)
        # model.to(device)
        # test_dataset = TestDataset(test_files, '/opt/ml/test1', test_transform)
        all_train_files = [all_files[ids] for ids in train_ids]
        all_val_files = [all_files[ids] for ids in val_ids] 
        train_dataset = TrainDataset(all_train_files, '/opt/ml/train', train_transform)
        val_dataset = TrainDataset(all_val_files, '/opt/ml/train', train_transform)


        print(f'FOLD {fold}')
        print('-----------------------------------')
        

        train_loader = DataLoader(train_dataset, batch_size=192,  drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=192,  drop_last=True)
        # test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        print(len(train_loader.dataset))
        criterion = nn.CrossEntropyLoss(reduction='sum')
        opt = optim.Adam(model.parameters(), lr=0.0002)

        scaler = grad_scaler.GradScaler()


        for epoch in range(num_epochs):
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
            print(len(train_loader.dataset))
            epoch_loss = train_loss / len(train_loader.dataset)
            epoch_acc = train_acc / len(train_loader.dataset)
            print(epoch_loss)
            print(epoch_acc)
            torch.save(model.state_dict(), f'/opt/ml/skku/dog_classifier/final/aa_Xception_fold_{fold}.pt')

        # Iterate over the test data and generate predictions
            with torch.no_grad():
                # Iterate over the test data and generate predictions
                for i, data in enumerate(tqdm(val_loader)):
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
                print(len(val_loader.dataset))
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
        

