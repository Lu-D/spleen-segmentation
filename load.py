# Author: Daiwei (David) Lu
# A fully custom dataloader for the cellphone dataset

import glob
import os
import torch
import pandas as pd
from PIL import Image
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch import nn as nn
import random

from nibabel.testing import data_path
import nibabel as nib


import warnings

warnings.filterwarnings("ignore")
plt.ion()

spleen = 1

class OrganDataset(Dataset):
    def __init__(self, mode, labels, transform=None, preload=False):
        if mode == 'train':
            self.root = 'Training'
        elif mode == 'val':
            self.root = 'Validation'
        else:
            self.root = 'Testing'
        fname = os.path.join(self.root, 'img/')
        self.input_files = sorted(glob.glob(fname + '*.nii.gz'))
        fname = os.path.join(self.root, 'label/')
        self.seg_files = sorted(glob.glob(fname + '*.nii.gz'))
        self.labels = labels
        self.output_x, self.output_y, self.output_z = 512,512, 3
        self.file_lengths = []
        self.filtered_idx = []
        print('processing')
        sum = 0
        count = 0
        threshold = 1e-2
        for f in self.seg_files:
            print(f)
            label = self.get_whole_y(f)
            spleencheck = label[1] == spleen
            size = np.size(label[1][0])
            for i in range(label.shape[1]):
                percent = np.sum(spleencheck[i]) / size
                if percent > threshold:
                    self.filtered_idx.append(count)
                count+=1
            self.file_lengths.append(label.shape[1])
        self.sum = sum
        self.transform = transform
        # remove those below threshold

    def __len__(self):
        return len(self.filtered_idx)

    # def __getitem__(self, idx):
    #     file_idx = self.getfileidx(idx)
    #     labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    #     # labels = [0, 1]
    #     num_labels = len(labels)
    #     # num_labels = 2
    #     img_file = self.input_files[idx]
    #     img_3d = nib.load(img_file)
    #     z_depth = 8
    #     output_x, output_y, output_z = 512,512,(img_3d.shape[2]//z_depth + 1)*z_depth
    #     # output_x, output_y, output_z = 512, 512, img_3d.shape[2] + 2
    #     img = img_3d.get_data()
    #     img_min = -150
    #     img_max = 600
    #     img[img<-img_min] = -img_min
    #     img[img>img_max] = img_max
    #     img = (img - img.min())/(img.max() - img.min())
    #     # img = img*255.0
    #     img = np.transpose(img,(2,0,1))
    #     img = np.pad(img, ((output_z - img_3d.shape[2], 0), (0, 0), (0, 0)), 'constant', constant_values=0)
    #     # img = np.pad(img, ((1,1),(0,0),(0,0)), 'constant', constant_values=0)
    #     x = np.zeros((1, output_z, output_x, output_y))
    #     x[0,:,:,:] = img[0:output_z,0:output_x,0:output_y]
    #     # x=x.astype('float32')
    #     x = x.astype('float16')
    #
    #     y = np.zeros((num_labels, output_z, output_x, output_y))
    #     seg_file = self.seg_files[idx]
    #     seg_3d = nib.load(seg_file)
    #     seg = seg_3d.get_data()
    #     seg = np.transpose(seg,(2,0,1))
    #     seg = np.pad(seg, ((output_z - seg_3d.shape[2], 0), (0, 0), (0, 0)), 'constant', constant_values=0)
    #     # seg = np.pad(seg, ((1,1),(0,0),(0,0)), 'constant', constant_values=0)
    #     y[0,:,:,:] = np.ones([output_z, output_x, output_y])
    #     for i in range(1, num_labels): # for i in range(1, num_labels):
    #         seg_one = seg == labels[i]
    #         y[i,:,:,:] = seg_one[0:output_z,0:output_x,0:output_y]
    #         y[0,:,:,:] = y[0,:,:,:] - y[i,:,:,:]
    #     # y = y.astype('float32')
    #     y = y.astype('float16')
    #     return x, y

    def __getitem__(self, idx):
        idx, file_idx = self.getfileidx(self.filtered_idx[idx])
        x = self.get_x(self.input_files[idx], file_idx)
        y = self.get_y(self.seg_files[idx], file_idx)
        return x,y

    def get_x(self, path, index):
        img_3d = nib.load(path)
        img = img_3d.get_data()
        img = (img - img.min()) / (img.max() - img.min())
        img = np.transpose(img, (2, 0, 1))
        img = np.pad(img, ((1, 1), (0, 0), (0, 0)), 'constant', constant_values=0)
        img = img[index:index+3]
        x = np.zeros((self.output_z, self.output_x, self.output_y))
        x[0:, :, :] = img[0:self.output_z,0:self.output_x, 0:self.output_y]
        x = x.astype('float32')
        return x

    def get_y(self, path, index):
        y = np.zeros((len(self.labels), self.output_z, self.output_x, self.output_y))
        seg_3d = nib.load(path)
        seg = seg_3d.get_data()
        seg = np.transpose(seg, (2, 0, 1))
        seg = seg[index:index+3]
        y[0, :, :] = np.ones([self.output_z, self.output_x, self.output_y])
        for i in range(1, len(self.labels)):
            seg_one = seg == self.labels[i]
            y[i,:, :, :] = seg_one[0:self.output_z,0:self.output_x, 0:self.output_y]
            y[0,:, :, :] = y[0, :, :, :] - y[i,:, :, :]
        y = y.astype('float32')
        return y

    def get_whole_y(self, path):
        seg_3d = nib.load(path)
        seg = seg_3d.get_data()
        seg = np.transpose(seg,(2,0,1))
        y = np.zeros((len(self.labels), seg.shape[0], self.output_x, self.output_y))
        y[0,:,:,:] = np.ones([seg.shape[0], self.output_x, self.output_y])
        for i in range(1, len(self.labels)): # for i in range(1, num_labels):
            seg_one = seg == self.labels[i]
            y[i,:,:,:] = seg_one[0:seg.shape[0],0:self.output_x,0:self.output_y]
            y[0,:,:,:] = y[0,:,:,:] - y[i,:,:,:]
        y = y.astype('float32')
        return y


    def getfileidx(self, idx):
        copyidx = idx
        ptr = 0
        while (self.file_lengths[ptr] <= copyidx):
            copyidx -= self.file_lengths[ptr]
            ptr += 1
        return ptr, copyidx

class OrganTestSet(Dataset):
    def __init__(self, mode, labels, transform=None, preload=False):
        if mode == 'train':
            self.root = 'Training'
        elif mode == 'val':
            self.root = 'Validation'
        else:
            self.root = 'Testing'
        fname = os.path.join(self.root, 'img/')
        self.input_files = sorted(glob.glob(fname + '*.nii.gz'))
        fname = os.path.join(self.root, 'label/')
        self.seg_files = sorted(glob.glob(fname + '*.nii.gz'))
        self.labels = labels
        self.output_x, self.output_y = 512,512
        print('processing')
        self.transform = transform
        # remove those below threshold

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        num_labels = len(labels)
        img_file = self.input_files[idx]
        img_3d = nib.load(img_file)
        z_depth = 8
        output_x, output_y, output_z = 512,512, img_3d.shape[2]
        img = img_3d.get_data()
        # img_min = -150
        # img_max = 600
        # img[img<-img_min] = -img_min
        # img[img>img_max] = img_max
        img = (img - img.min())/(img.max() - img.min())
        # img = img*255.0
        img = np.transpose(img,(2,0,1))
        x = np.zeros((1, output_z, output_x, output_y))
        x[0,:,:,:] = img[0:output_z,0:output_x,0:output_y]
        x=x.astype('float32')

        y = np.zeros((num_labels, output_z, output_x, output_y))
        seg_file = self.seg_files[idx]
        seg_3d = nib.load(seg_file)
        seg = seg_3d.get_data()
        seg = np.transpose(seg,(2,0,1))
        y[0,:,:,:] = np.ones([output_z, output_x, output_y])
        for i in range(1, num_labels):
            seg_one = seg == labels[i]
            y[i,:,:,:] = seg_one[0:output_z,0:output_x,0:output_y]
            y[0,:,:,:] = y[0,:,:,:] - y[i,:,:,:]
        y = y.astype('float32')
        return x, y

