import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from load import *
from model import UNet3D
from model2d import UNet as UNet2D
from torchvision import models
from torch.autograd import Variable
import tqdm
from utils import *
import warnings
gpu = 1

MODEL_PATH = './latest_3d_unet.pth'

def main():
    # Training settings
    device = torch.device("cuda:" + str(gpu))
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # labels = [0, 1]
    nclasses = len(labels)
    model = UNet2D(1, nclasses)
    model = model.to(device)
    # model.half()  # convert to half precision
    # for layer in model.modules():
    #     if isinstance(layer, nn.BatchNorm2d):
    #         layer.float()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    criterion = DiceLoss()

    with torch.no_grad():
        file = ''
        dataset = OrganTestSet('val', labels)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                      shuffle=False)
        epoch = 1
        predicted = torch.tensor(np.zeros((1, nclasses, 94, 512, 512)))
        control = torch.tensor(np.zeros((1, nclasses, 94, 512, 512)))
        for batch_idx, (img, label) in tqdm.tqdm(
                enumerate(dataloader), total=len(dataloader),
                desc='Train epoch=%d' % epoch, ncols=80, leave=False):
            for i in range(img.shape[2]):

                data, target = img[0:,0:,i].cuda(gpu), label[0:,0:,i].cuda(gpu)

                pred = model(data)
                loss = criterion(pred, target)
                print(loss)
                # pred[pred >= 0.5] = 1.
                # pred[pred < 0.5] = 0.
                predicted[0:,0:,i] = pred.cpu()
                control[0:,0:,i] = target.cpu()
            break
        print('Average MultiClass Dice: ', dice_3d(predicted, control))
        print('Average Spleen Dice: ', dice_3d(predicted[0:, 1:2], control[0:, 1:2]) )
        lbl_pred = predicted.data.max(1)[1].numpy()[:,:, :].astype('uint8')
        lbl_pred = np.transpose(lbl_pred, (0, 2, 3, 1))
        batch_num = lbl_pred.shape[0]
        for si in range(batch_num):
            curr_sub_name = labels[si]
            out_img_dir = os.path.join('result')
            # os.mkdir(out_img_dir)
            out_nii_file = os.path.join(out_img_dir,('%s_seg.nii.gz'%(curr_sub_name)))
            img = nib.load('./Validation/label/label0035.nii.gz')
            seg_img = nib.Nifti1Image(lbl_pred[si], img.affine, img.header)
            nib.save(seg_img, out_nii_file)
            print('saved')

        lbl_pred = control.data.max(1)[1].numpy()[:, :, :].astype('uint8')
        lbl_pred = np.transpose(lbl_pred, (0, 2, 3, 1))
        batch_num = lbl_pred.shape[0]
        for si in range(batch_num):
            curr_sub_name = labels[si]
            out_img_dir = os.path.join('result')
            # os.mkdir(out_img_dir)
            out_nii_file = os.path.join(out_img_dir, ('control'))
            img = nib.load('./Validation/label/label0035.nii.gz')
            seg_img = nib.Nifti1Image(lbl_pred[si], img.affine, img.header)
            nib.save(seg_img, out_nii_file)


if __name__ == '__main__':
    main()
