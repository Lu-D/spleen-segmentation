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
from model import *
from model2d import UNet as Unet2D
from torchvision import models
from torch.autograd import Variable
import tqdm
from deeplabv3 import DeepLabV3
from utils import *
import warnings

warnings.filterwarnings("ignore")
plt.ion()
gpu = 0

# weights = torch.tensor([1.04711931e+00, 2.05851586e+02, 4.41124712e+02, 4.44060326e+02,
#  2.43453180e+03, 4.80417464e+03, 4.03782197e+01, 1.62331464e+02,
#  7.22072133e+02, 7.80893  115e+02, 1.94613655e+03, 8.34135581e+02,
#  1.59189110e+04, 1.29812090e+04]).cuda(gpu)

def train(model, criterion, optimizer, train_loader, device, num_epochs=25):
    for epoch in range(num_epochs):
        model.train()
        with torch.set_grad_enabled(True):
            epoch_loss = 0
            for batch_idx, (img, label) in tqdm.tqdm(
                enumerate(train_loader), total=len(train_loader),
                    desc='Train epoch=%d' % epoch, ncols=80, leave=False):
                # for i in range(int(img.shape[2]/8)):
                #     data = img[0:,0:,i*8:(i+1)*8]
                #     target = label[0:,0:,i*8:(i+1)*8]
                data, target = img.cuda(gpu), label.cuda(gpu)
                pred = model(data)
                loss = criterion(pred, target)
                epoch_loss += loss
                # print(loss)
                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('Avg Loss: ' + str(epoch_loss/10))
                    epoch_loss = 0
                    torch.save(model.state_dict(), './latest_3d_unet.pth')
    return model

def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=25):
    print('training')
    since = time.time()
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            total_loss = 0.0
            count = 0
            for inputs, labels in dataloaders[phase]:
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    data, target = inputs.cuda(gpu), labels.cuda(gpu)
                    pred = model(data)
                    loss = criterion(pred, target)
                    # del data, pred, target
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        count += 1
                        optimizer.zero_grad()
                        loss.backward()
                        # nn.utils.clip_grad_value_(model.parameters(), 0.05)
                        optimizer.step()
                        if count % 50 == 0:
                            print('average loss: ', running_loss)
                            print('Saving...')
                            torch.save(model.state_dict(), './latest_3d_unet.pth')
                    running_loss = 0.5 * loss + 0.5 * running_loss
                    if phase == 'train':
                        train_loss.append(running_loss)
                    else:
                        scheduler.step(running_loss)
                        val_loss.append(running_loss)
                    total_loss += loss

            epoch_loss = total_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            if phase == 'train':
                train_loss.append(epoch_loss)
            else:
                scheduler.step(epoch_loss)
                val_loss.append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, './trained_best.pth')
                plot(train_loss, val_loss, epoch)

        # if epoch % 10 == 0:
        #     torch.save(best_model_wts, './trained_best.pth')
        #     plot(train_loss, val_loss, epoch)
        epoch_time = time.time() - epoch_start
        print('Epoch {} in {:.0f}m {:.0f}s'.format(epoch,
            epoch_time // 60, epoch_time % 60))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss, val_loss

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
preload = False
image_datasets = {'train': OrganDataset('train', labels),
                  'val': OrganDataset('val', labels)}

def main():
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=1,
                                                        shuffle=True,
                                                        num_workers=8),
                   'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=1,
                                                      shuffle=True,
                                                      num_workers=8)}

    device = torch.device("cuda:" + str(gpu))

    #################################
    # labels = [0, 1]
    nclasses = len(labels)

    # model = Unet2D(1, nclasses)
    model = DeepLabV3(len(labels))
    model = model.to(device)
    # model.half()  # convert to half precision
    # for layer in model.modules():
    #     if isinstance(layer, nn.BatchNorm2d):
    #         layer.float()
    #################################
    # weights = None
    criterion = DiceLoss()
    # criterion = nn.CrossEntropyLoss()
    optimizer_conv = optim.Adam(model.parameters(), lr=0.01)
    # optimizer_conv = Adam16(model.parameters(), lr=1e-3)
    exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_conv)
    # print(model)
    epochs = 20
    train_model(model, criterion, optimizer_conv, exp_lr_scheduler, dataloaders, device, num_epochs=epochs)
    # model = train(model, criterion, optimizer_conv, dataloaders['train'], device, num_epochs=epochs)

    # torch.save(model.state_dict(), './trained_best.pth')

    # plot(train_loss, val_loss, epochs)

# def test_dice():
#     dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=1,
#                                                         shuffle=True),
#                    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=1,
#                                                       shuffle=True)}
#     loader = dataloaders['train']
#     criterion = DiceLoss()
#
#     for batch_idx, (img, label) in tqdm.tqdm(
#         enumerate(loader), total=len(loader),
#             desc='Train epoch=%d' % 1, ncols=80, leave=False):
#         for i in range(int(img.shape[2]/8)):
#             data = img[0:,0:,i*8:(i+1)*8]
#             target = label[0:,0:,i*8:(i+1)*8]
#             data, target = data.cuda(gpu), target.cuda(gpu)
#             data, target = Variable(data), Variable(target)
#
#             loss = criterion(target, target)
#             print(loss)
#         break


if __name__ == '__main__':
    main()
