from nibabel.testing import data_path
import nibabel as nib
import glob
import numpy as np
import os
# file_name = ('./Training/label/label0001.nii.gz')
# img = nib.load(file_name)
# print(img.shape)

root = './Training'
path = os.path.join(root, 'label/')
fnames = glob.glob(path + '*.nii.gz')

for fname in sorted(fnames):
    labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    num_labels = len(labels)
    img = nib.load(fname)
    output_x, output_y, output_z = img.shape
    y = np.zeros((num_labels, output_z, output_x, output_y))
    seg = img.get_data()
    seg = np.transpose(seg, (2, 0, 1))
    y[0, :, :, :] = np.ones((output_z, output_x, output_y))
    for i in range(1, num_labels):
        seg_one = seg == labels[i]
        y[i, :, :, :] = seg_one[0:output_z, 0:output_x, 0:output_y]
        y[0, :, :, :] = y[0, :, :, :] - y[i, :, :, :]
    print(np.sum(y[1]))

    #### make prediction on img data

    lbl_pred = np.transpose(y, (0, 2, 3, 1))
    batch_num = lbl_pred.shape[0]
    for si in range(batch_num):
        curr_sub_name = labels[si]
        out_img_dir = os.path.join('result')
        # os.mkdir(out_img_dir)
        out_nii_file = os.path.join(out_img_dir,('%s_seg.nii.gz'%(curr_sub_name)))
        # img = nib.load(img_file)
        seg_img = nib.Nifti1Image(lbl_pred[si], img.affine, img.header)
        nib.save(seg_img, out_nii_file)

    break
    # print(img.shape)
