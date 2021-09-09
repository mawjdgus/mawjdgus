



from timm.models import xception
from data_transform import TestDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm

import timm
import pandas as pd

import matplotlib.pyplot as plt
from importlib import import_module

# MODEL_PATH = '/opt/ml/skku/dog_classifier/5_Xception.pt'
# SUBMISSION_PATH = '/opt/ml/skku/dog_classifier/sampleSubmission.csv'
# SAVED_INFERENCE_PATH = '/opt/ml/skku/dog_classifier/final/subsub.csv'

MODEL_PATH = ''
SUBMISSION_PATH = ''
SAVED_INFERENCE_PATH = ''
test_files = [f'{i}.jpg' for i in range(1,12501)]

torch.cuda.empty_cache()

test_transform = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.RandomCrop(255),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
test_dataset = TestDataset(test_files, '/opt/ml/test1', test_transform)

test_loader = DataLoader(test_dataset, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = timm.create_model('xception', pretrained=False, num_classes=2).to(device)

df = pd.read_csv(SUBMISSION_PATH,index_col=0)

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

preds_list = []

with torch.no_grad():
    for i, test_batch in enumerate(tqdm(test_loader)):
        

        inputs = test_batch
        inputs = inputs.to(device)

        outs = model(inputs)


        preds = torch.functional.F.softmax(outs, dim=1)[:,1].tolist()
        preds_list.append(float(preds[0]))
        
## save 

df['label'] = preds_list
df.to_csv(SAVED_INFERENCE_PATH,mode='w')

