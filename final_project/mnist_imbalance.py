# %%
import argparse
import random
import numpy as np
from torch import nn
import torch.optim as optim
import copy
import torch
import torch.utils.data
import torchvision
import pandas as pd
from torch.autograd import Variable
from torchvision import transforms, datasets
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
# import torchvision.transforms as transforms
# import torchvision.datasets as dsets
# import sklearn.metrics aspy sm
import torch.nn.functional as F

# parse arguments
parser = argparse.ArgumentParser(description='Imbalanced MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--nrow', type=int, default=5,
                    help='rows of example')
parser.add_argument('--ncol', type=int, default=10,
                    help='columns of example')


args = parser.parse_args()

torch.manual_seed(args.seed)


# %%
imbalanced_linear_train_dataset = torch.load('imbalanced_linear_train_dataset.pt')
# 





imbalanced_linear_train_loader = torch.utils.data.DataLoader(imbalanced_linear_train_dataset, batch_size=args.batch_size, shuffle=True)

imbalanced_step_train_dataset = torch.load('imbalanced_step_train_dataset.pt')
imbalanced_step_train_loader = torch.utils.data.DataLoader(imbalanced_step_train_dataset, batch_size=args.batch_size, shuffle=True)




# %%
test_dataset = torch.load('test_dataset.pt')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)
# print(imbalanced_linear_train_loader.data)
# %%
import matplotlib.pyplot as plt

import seaborn as sns


def show_mnist(arr, nrow=args.nrow, ncol=args.ncol, figsize=None):
    
    if figsize is None:
        figsize = (ncol, nrow)
        
    f, a = plt.subplots(nrow, ncol, figsize=figsize)
    
    def _do_show(the_figure, the_array):
        the_figure.imshow(the_array)
        the_figure.axis('off')
    
    for i in range(nrow):
        for j in range(ncol):
            _do_show(a[i][j], np.reshape(arr[i * ncol + j], (28, 28)))
            
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.draw()
    plt.savefig('examples.png')


#print('Distribution of classes in linear imbalanced dataset:')
fig, ax = plt.subplots()
_, counts = np.unique(imbalanced_linear_train_loader.dataset.train_labels, return_counts=True)
num_classes = 10
classe_labels = range(num_classes)
ax.bar(classe_labels, counts)
ax.set_xticks(classe_labels)
plt.savefig('dist linear.png')
# plt.show()


#print('Distribution of classes in step imbalanced dataset:')
fig, ax = plt.subplots()
_, counts = np.unique(imbalanced_step_train_loader.dataset.train_labels, return_counts=True)
num_classes = 10
classe_labels = range(num_classes)
ax.bar(classe_labels, counts)
ax.set_xticks(classe_labels)
plt.savefig('dist step.png')
# plt.show()


X_linear_train, y_linear_train = next(iter(imbalanced_linear_train_loader))
# train_set = imbalanced_linear_train_dataset.numpy()
# for feature,label in imbalanced_linear_train_loader:
#     X_linear_train.append(feature)
#     y_linear_train.append(label)
#     # show_mnist(data)
#     break

X_step_train, y_step_train = next(iter(imbalanced_step_train_loader))
# for feature, label in imbalanced_step_train_loader:
#     X_step_train.append(feature)
#     y_step_train.append(label)
#     break

X_test, y_test = next(iter(test_loader))
nsamples, col, nx, ny = X_linear_train.shape
X_linear_train = X_linear_train.reshape((nsamples,nx*ny))
X_linear_train = preprocessing.scale(X_linear_train)
X_linear_train = X_linear_train.reshape((nsamples,nx,ny))

# preprocessing.Normalizer(X_linear_train_minmax)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #input shape (3,256,256)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, #input channels
                      out_channels=16, #n_filter
                     kernel_size=5, #filter size
                     stride=1, #filter step
                     padding=2 #same output size
                     ), #shape (16,256,256)
            nn.ReLU(), #fully-connected layer
            nn.MaxPool2d(kernel_size=2)) #max pooling layer. 2x2 sampling
        #output shape (16,128,128)
        
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, 1, 2), 
                                  nn.ReLU(),
                                  nn.MaxPool2d(2))
        #output shape (32,64,64)
        
        self.out = nn.Linear(32*64*64,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        #Flatten
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output

# model1 = CNN()
# # test_numpy = test_dataset.numpy()
# # print(test_dataset)
# # x = torchvision.transforms.Normalize(mean= 0.1307, std= 0.3081)
# data = next(iter(imbalanced_linear_train_loader))
# mean = data[0].mean()
# std = data[0].std()
# print(mean)
# print

# torchvision.transforms.Normalize(imbalanced_linear_train_dataset)
# print(imbalanced_linear_train_dataset)

# print(imbalanced_step_train_dataset)
# print(test_numpy)
# # x = pd.DataFrame(test_dataset)
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0.1307,), (0.3081,))])

# imbalanced_linear_train_array = next(iter(imbalanced_linear_train_loader))[0].numpy()
# imbalanced_step_train_array = next(iter(imbalanced_linear_train_loader))[0].numpy()
# test_dataset_array = next(iter(test_dataset))[0].numpy()
# print(train_dataset_array.shape)

# for images, labels in imbalanced_linear_train_dataset.take(-1):  # only take first element of dataset
#     numpy_images = images.numpy()
#     numpy_labels = labels.numpy()
# def standard_normalizer(x):
#     scaler = StandardScaler()
#     scaler.fit(x)
#     x = scaler.transform(x)
#     return x

# norm_numpy_images = standard_normalizer(numpy_images)
# nsamples,col, nx, ny = imbalanced_linear_train_array.shape
# d2_train_dataset = imbalanced_linear_train_array.reshape((nsamples,nx*ny))
# print(d2_train_dataset.shape)
# normalized_linear = standard_normalizer(d2_train_dataset)
# print(d2_train_dataset)
# nsamples,col, nx, ny = test_dataset.shape
# d2_test_dataset = test_dataset.reshape((nsamples,nx*ny))
# normalized_test = standard_normalizer(d2_test_dataset)

log_model = LogisticRegressionCV().fit(X_linear_train, y_linear_train)