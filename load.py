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

class OrganDataset(Dataset):
    def __init__(self, mode='test', transform=None, preload=False):
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

        self.transform = transform
        # self.preload = preload
        # if preload:
        #     self.preloaded = []
        #     for i in range(len(self.images)):
        #         img_name = os.path.join(self.root, self.images[i] + '.jpg')
        #         self.preloaded.append(io.imread(img_name))

    # def __len__(self):
    #     return len(self.input_files) * 4
    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        # labels = [1]
        num_labels = len(labels)
        # num_labels = 2
        img_file = self.input_files[idx]
        img_3d = nib.load(img_file)
        z_depth = 8
        output_x, output_y, output_z = 512,512,(img_3d.shape[2]//z_depth + 1)*z_depth
        # output_x, output_y, output_z = 512, 512, img_3d.shape[2] + 2
        img = img_3d.get_data()
        img_min = 50
        img_max = 400
        img[img<-img_min] = -img_min
        img[img>img_max] = img_max
        img = (img - img.min())/(img.max() - img.min())
        # img = img*255.0
        img = np.transpose(img,(2,0,1))
        img = np.pad(img, ((output_z - img_3d.shape[2], 0), (0, 0), (0, 0)), 'constant', constant_values=0)
        # img = np.pad(img, ((1,1),(0,0),(0,0)), 'constant', constant_values=0)
        x = np.zeros((1, output_z, output_x, output_y))
        x[0,:,:,:] = img[0:output_z,0:output_x,0:output_y]
        # x=x.astype('float32')
        x = x.astype('float16')

        y = np.zeros((num_labels, output_z, output_x, output_y))
        seg_file = self.seg_files[idx]
        seg_3d = nib.load(seg_file)
        seg = seg_3d.get_data()
        seg = np.transpose(seg,(2,0,1))
        seg = np.pad(seg, ((output_z - seg_3d.shape[2], 0), (0, 0), (0, 0)), 'constant', constant_values=0)
        # seg = np.pad(seg, ((1,1),(0,0),(0,0)), 'constant', constant_values=0)
        y[0,:,:,:] = np.ones([output_z, output_x, output_y])
        for i in range(1, num_labels): # for i in range(1, num_labels):
            seg_one = seg == labels[i]
            y[i,:,:,:] = seg_one[0:output_z,0:output_x,0:output_y]
            y[0,:,:,:] = y[0,:,:,:] - y[i,:,:,:]
        # y = y.astype('float32')
        y = y.astype('float16')
        return x, y

    # def __getitem__(self, index):
    #     # divided into
    #     # 0 | 1
    #     # -----
    #     # 2 | 3
    #     qx = index % 2
    #     qy = (index % 4) // 2
    #     idx = index//4
    #     labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    #     num_labels = len(labels)
    #     # num_labels = 2
    #     img_file = self.input_files[idx]
    #     img_3d = nib.load(img_file)
    #     output_x, output_y, output_z = 512,512,(img_3d.shape[2]//24 + 1)*24
    #     img = img_3d.get_data()
    #     img = (img - img.min())/(img.max() - img.min())
    #     img = img*255.0
    #     img = np.transpose(img,(2,0,1))
    #     img = np.pad(img, ((output_z - img_3d.shape[2],0),(0,0),(0,0)), 'constant', constant_values=0)
    #     x = np.zeros((1, output_z, output_x, output_y))
    #     x[0,:,:,:] = img[0:output_z,qx*output_x:(1+qx)*output_x,qy*output_y:(1+qy)*output_y]
    #     x=x.astype('float32')
    #     # x = x.astype('float16')
    #
    #     y = np.zeros((num_labels, output_z, output_x, output_y))
    #     seg_file = self.seg_files[idx]
    #     seg_3d = nib.load(seg_file)
    #     seg = seg_3d.get_data()
    #     seg = np.transpose(seg,(2,0,1))
    #     seg = np.pad(seg, ((output_z - seg_3d.shape[2],0),(0,0),(0,0)), 'constant', constant_values=0)
    #     y[0,:,:,:] = np.ones([output_z, output_x, output_y])
    #     for i in range(1, num_labels):
    #         seg_one = seg == labels[i]
    #         y[i,:,:,:] = seg_one[0:output_z,qx*output_x:(1+qx)*output_x,qy*output_y:(1+qy)*output_y]
    #         y[0,:,:,:] = y[0,:,:,:] - y[i,:,:,:]
    #     y = y.astype('float32')
    #     # y = y.astype('float16')
    #
    #     return x, y

class TumorImage(Dataset):
    def __init__(self, path, transform=None):
        img_name = os.path.join(path)
        self.image = [io.imread(img_name)]
        self.transform = transform

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.image[idx]
        if self.transform:
            image = self.transform(image)
        return image

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample
        new_h, new_w = self.output_size, self.output_size
        img = transform.resize(image, (new_h, new_w))
        return img


class Normalize(object):
    def __init__(self, inplace=False):
        #dataset mean/std
        # self.mean = (0.76964605, 0.54124683, 0.56347674)
        # self.std = (0.1364224, 0.15036866, 0.1672849)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.inplace = inplace

    def __call__(self, sample):
        return TF.normalize(sample, self.mean, self.std, self.inplace)


class ToTensor(object):

    def __call__(self, sample):
        dtype = torch.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        # image, labels = sample['image'], sample['labels']
        # return {'image': TF.to_tensor(image), 'labels': labels}
        return TF.to_tensor(sample)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image = sample
        if random.random() < self.p:
            image *= 255
            image = Image.fromarray(np.uint8(image))
            image = TF.hflip(image)
            image = np.array(image)
            image = np.double(image) / 255.
        return image


class RandomVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image = sample
        if random.random() < self.p:
            image *= 255.
            image = Image.fromarray(np.uint8(image))
            image = TF.vflip(image)
            image = np.array(image)
            image = np.double(image) / 255.
        return image


class RandomColorJitter(object):
    def __init__(self, p=0.2, brightness=(0.5, 1.755), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.1, 0.1)):
        self.p = p
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, sample):
        image = sample
        if random.random() < self.p:
            image *= 255.
            image = Image.fromarray(np.uint8(image))
            modifications = []

            brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            modifications.append(transforms.Lambda(lambda img: TF.adjust_brightness(image, brightness_factor)))

            contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            modifications.append(transforms.Lambda(lambda img: TF.adjust_contrast(image, contrast_factor)))

            saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            modifications.append(transforms.Lambda(lambda img: TF.adjust_saturation(image, saturation_factor)))

            hue_factor = random.uniform(self.hue[0], self.hue[1])
            modifications.append(transforms.Lambda(lambda img: TF.adjust_hue(image, hue_factor)))

            random.shuffle(modifications)
            modification = transforms.Compose(modifications)
            image = modification(image)

            image = np.array(image)
            image = np.double(image) / 255.
        return image
