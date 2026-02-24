import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle




class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train",add_noise=True):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.add_noise = add_noise

        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, :] 
        # data = data.iloc[:,:]
        data = np.nan_to_num(data)


        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        val_data = test_data[:5000, :]  
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        # self.val = self.test
        self.val = self.scaler.transform(val_data)  

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class ASDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, :] 
        # data = data.iloc[:,:]
        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, :]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, :]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



class BaseSMDSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, exclude_index=None, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Load and preprocess training data
        data = pd.read_csv(data_path + '/train.csv')
        data = np.nan_to_num(data)
        print(f'Data without the time series {exclude_index}')
        if exclude_index is not None:
            print(f"Excluding time series index {exclude_index}")
            data = np.delete(data, exclude_index, axis=1)
        # data = np.delete(data, exclude_index, axis=1)

        self.scaler.fit(data)
        data = self.scaler.transform(data)

        # Load and preprocess test data
        test_data = pd.read_csv(data_path + '/test.csv').values[:,1:]
        test_data = np.nan_to_num(test_data)
        
        if exclude_index is not None:
            print(f"Excluding time series index {exclude_index}")
            test_data = np.delete(test_data, exclude_index, axis=1)        
        # test_data = np.delete(test_data, exclude_index, axis=1)
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

# Dynamically create classes for SMD0SegLoader to SMD37SegLoader
SMDLoaders = {}
for i in range(38):
    class_name = f"SMD{i}SegLoader"
    exclude_index = i
    def init(self, data_path, win_size, step, mode='train', exclude_index=exclude_index):
        BaseSMDSegLoader.__init__(self, data_path, win_size, step, exclude_index, mode)
    SMDLoaders[class_name] = type(class_name, (BaseSMDSegLoader,), {'__init__': init})


class BaseASDSegLoader(Dataset):
    def __init__(self, data_path, win_size, step, exclude_index, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        # Load and preprocess training data
        data = pd.read_csv(data_path + '/train.csv')
        data = np.nan_to_num(data)
        print(f'Data without the time series {exclude_index}')
        data = np.delete(data, exclude_index, axis=1)

        self.scaler.fit(data)
        data = self.scaler.transform(data)

        # Load and preprocess test data
        test_data = pd.read_csv(data_path + '/test.csv')
        test_data = np.nan_to_num(test_data)
        test_data = np.delete(test_data, exclude_index, axis=1)
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, :]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'val':
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'val':
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif self.mode == 'test':
            return np.float32(self.test[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


ASDLoaders = {}
for i in range(19):
    class_name = f"ASD{i}SegLoader"
    exclude_index = i
    def init(self, data_path, win_size, step, mode='train', exclude_index=exclude_index):
        BaseASDSegLoader.__init__(self, data_path, win_size, step, exclude_index, mode)
    ASDLoaders[class_name] = type(class_name, (BaseASDSegLoader,), {'__init__': init})


class WAVESSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train",add_noise=True):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        # self.add_noise = add_noise

        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:] 
        # data = data.iloc[:,:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        # self.test = self.scaler.transform(test_data)
        self.test = self.scaler.transform(test_data)

        self.train = data
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_labels.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.mode == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])



def get_loader_segment(data_path, batch_size, win_size=100, step=1, mode='train', dataset='KDD'):
    if dataset == "SMD":
        dataset = SMDSegLoader(data_path, win_size, step, mode=mode)  
    elif dataset.startswith('SMD') and dataset[3:].isdigit():
        i = int(dataset[3:])
        loader_class = SMDLoaders.get(f"SMD{i}SegLoader")
        if loader_class is not None:
            dataset = loader_class(data_path, win_size, step, mode)
    elif dataset == 'ASD':
        dataset = ASDSegLoader(data_path, win_size, 1, mode)
    elif dataset == 'MSDS':
        dataset = MSDSSegLoader(data_path, win_size, 1, mode)
    elif dataset == 'WAVES':
        dataset = WAVESSegLoader(data_path, win_size, 1, mode)

    elif dataset == 'HAI':
        dataset = HAISegLoaderFull(data_path, win_size, 1, mode)
    elif dataset == 'FNUSAfeatures':
        dataset = FNUSAfeaturesSegLoader(data_path, win_size, 1, mode)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader


