

import torch
import torch.nn as nn
import torch.optim.optimizer as optimizer
from torch import Tensor


class MyLiner(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weights = nn.Parameter(
            torch.randn(in_features, out_features)
        )

        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x : Tensor):
        return x @ self.weights + self.bias

x = torch.rand(5, 7)

layer = MyLiner(7, 12)
layer(x)

layer(x).shape

for value in layer.parameters(): ## parameter는 미분의 대상이 되는 것만 보여줄 수 있다. 즉, 13번째 줄에서 Parameter나 17줄의 Parameter를 Tensor로 바꿔주게되면, 여기서 출력되지 않는다.
    print(value)


##########################################################


import numpy as np

# create nummy data for training
x_values = [i for i in range(11)]
x_train =  np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1,1)

y_value = [2*i + 1 for i in x_values]
y_train = np.array(y_value, dtype=np.float32)
y_train = y_train.reshape(-1,1)


x_train
y_train

import torch
from torch.autograd import Variable

class LinearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x) # Parameter를 직접 가져올 필욘 없다
        return out

inputDim = 1
outputDim = 1
learningRate = 0.01
epochs = 100

model = LinearRegression(inputDim, outputDim)

if torch.cuda.is_available():
    model.cuda()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

for epoch in range(epochs):
    # Converting inputs and labels to Variable
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_train).cuda())
        labels = Variable(torch.from_numpy(y_train).cuda())
    else:
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))


    # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, don't wnat to cummulate gradients
    optimizer.zero_grad()

    # get output from the model, given the inputs
    outputs = model(inputs)

    # get loss for the predicted output
    loss = criterion(outputs, labels)
    print(loss)

    # get gradients w.r.t to parameters
    loss.backward()

    # update parameters
    optimizer.step()

    print('epoch {}, loss {}'.format(epoch, loss.item()))


with torch.no_grad(): # we don't need gradients in the testing phase
    if torch.cuda.is_available():
        predicted = model(Variable(torch.from_numpy(x_train).cuda())).cpu().data.numpy()
    else:
        predicted = model(Variable(torch.from_numpy(x_train))).data.numpy()
    print(predicted)

for p in model.parameters():
    if p.requires_grad:
        print(p.name, p.data)


# 실제 Backward는 Module 단계에서 직접 지정가능
# Module에서 backward와 optimizer 오버라이딩
# 사용자가 직접 미분 수식을 써야하는 부담
# -> 쓸 일은 없으나 순서는 이해할 필요는 있다.


#################################

class LR(nn.Module):
    def __init__(self, dim, lr=torch.scalar_tensor(0.01)):
        super(LR, self).__init__()
        #initialize parameters
        self.w = torch.zeros(dim, 1, dtype=torch.float).to(device)
        self.b = torch.scalar_tensor(0).to(device)
        self.grads = {"dw":torch.zeros(dim, 1, dtype=torch.float).to(device),
                      "db":torch.scalar_tensor(0).to(device)}
        self.lr = lr.to(device)

    def forward(self, x):
        # compute forward
        z = torch.mm(self.w.T, x)
        a = self.sigmoid(z)
        return a

    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def backward(self, x, yhat, y):
        ## compute backward
        self.grads["dw"] = (1 / x.shape[1]) * torch.mm(x,(yhat - y).T)
        self.grads["db"] = (1 / x.shape[1]) * torch.sum(yhat - y)

    def optimize(self):
        ## optimization step
        self.w = self.w - self.lr*self.grads["dw"]
        self.b = self.b - self.lr*self.grads["db"]


################################## Logistic regression

import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import numpy as np
import matplotlib.pyplot as plt

print(torch.__version__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import urllib
import os
import shutil
from zipfile import ZipFile

DATA_PATH = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
urllib.request.urlretrieve(DATA_PATH, "hymenoptera_data.zip")

with ZipFile("hymenoptera_data.zip", 'r') as zipObj:
    # Extract all the contents of zip file in current directory
    zipObj.extractall()

os.rename("hymenoptera_data", "data2")

## configure root folder on your gdrive
data_dir = './data2'

## custom transformer to flatten the image tensors
class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        result = torch.reshape(img, self.new_size)
        return result

## transformations used to standardize and normalize the dataets
data_transforms = {
    'train' : transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ReshapeTransform((-1,)) ## flattens the data
    ]),
    'val' : transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        ReshapeTransform((-1,)) ## flattens the data
    ]),
}

## load the corresponding folders
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x]) for x in ['train', 'val']}

## load the entire dataset: we are not using minibatches here
train_dataset = torch.utils.data.DataLoader(image_datasets['train'],
                                            batch_size=len(image_datasets['train']),
                                            shuffle=True)

test_dataset = torch.utils.data.DataLoader(image_datasets['val'],
                                            batch_size=len(image_datasets['val']),
                                            shuffle=True)

## load the entire datset
x, y = next(iter(train_dataset))

## print one example
dim = x.shape[1]
print("Dimension of image:", x.shape, '\n',
      "Dimension of labels:", y.shape)

plt.imshow(x[160].reshape(1,3,244,244).squeeze().T.numpy())

