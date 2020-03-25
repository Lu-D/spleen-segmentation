# # Author: Daiwei (David) Lu
# # Train custom model
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from load import *
from model import UNet2D, UNet3D
from torchvision import models
from torch.autograd import Variable
import tqdm
from utils import *
import warnings
from sklearn.utils import class_weight

warnings.filterwarnings("ignore")
plt.ion()
gpu = 0


def train(model, criterion, optim, train_loader, device, num_epochs=1):
    # total = []
    # for batch_idx, (img, label) in tqdm.tqdm(
    #     enumerate(train_loader), total=len(train_loader),
    #         desc='Train epoch=%d' % 1, ncols=80, leave=False):
    #     sum = np.sum(label.numpy(), axis=(0,2,3,4))
    #     total.append(sum)
    # total = np.array(total)
    # class_count = np.mean(total, axis=0)
    # print(class_count)
    class_count = np.array([3.1964254e+07, 1.6182105e+05, 7.5513086e+04, 7.5013875e+04, 1.3682458e+04,
     6.9336250e+03, 8.2505956e+05, 2.0520567e+05, 4.6131875e+04, 4.2656957e+04,
     1.7116166e+04, 3.9934168e+04, 2.0925000e+03, 2.5660417e+03])
    better = class_count / np.sum(class_count)
    print(better)
    beta = 0.99
    weights = beta ** better
    weights = (1 - beta) / (1 - weights)
    # weights = class_weight.compute_class_weight('balanced', np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]), class_count)
    print(weights)


    return model

preload = False
image_datasets = {'train': OrganDataset(mode='train',
                                        preload=preload),
                  'val': OrganDataset(mode='test',
                                      preload=preload)}


def main():
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=1,
                                                        shuffle=True),
                   'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=1,
                                                      shuffle=True)}

    device = torch.device("cuda:" + str(gpu))

    #################################
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # labels = [1]
    nclasses = len(labels)
    model = UNet2D(3, nclasses)
    model = model.to(device)
    # model.half()  # convert to half precision
    # for layer in model.modules():
    #     if isinstance(layer, nn.BatchNorm2d):
    #         layer.float()
    #################################

    criterion = DiceLoss()

    optimizer_conv = optim.Adam(model.parameters())
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_conv)
    print(model)
    epochs = 20
    # train(model, criterion, optimizer_conv, exp_lr_scheduler, dataloaders, device, num_epochs=epochs)
    model = train(model, criterion, optimizer_conv, dataloaders['train'], device, num_epochs=epochs)

    # torch.save(model.state_dict(), './trained_best.pth')

    # plot(train_loss, val_loss, epochs)

if __name__ == '__main__':
    main()
