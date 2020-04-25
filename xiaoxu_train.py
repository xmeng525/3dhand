from __future__ import division
import os
import torch
from torch.autograd import Variable
from torch.utils import data
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from torchvision.transforms import ToTensor, Compose

from model import resnet34_Mano
from datasets import PanopticSet
from transform import Scale

from loss_utils import OxfordLoss, save_loss
from display_utils import display_hand_3d, plot_hand

init_model = False
# 1 use image and joint heat maps as input
# 0 use image only as input 
input_option = 1
batch_size = 50
image_scale = 256
learning_rate = 0.00005
epochs = 10
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

template = open('data/template.obj')
content = template.readlines()
template.close()

decoder_criterion = OxfordLoss()
decoder_criterion.a_2d = 10
decoder_criterion.a_3d = 10
decoder_criterion.a_mask = 100
decoder_criterion.a_reg = 0.0001

my_transform = Compose([Scale((image_scale, image_scale), Image.BILINEAR), ToTensor()])
root_dir = '/home/xiaoxu/Documents/rgb2mesh/BootStrapping/panoptic-toolbox/'
dataset_list = ['171026_pose3']

train_dataset = PanopticSet(root_dir, dataset_list=dataset_list, 
    image_size=image_scale, data_par='train', img_transform=my_transform)
valid_dataset = PanopticSet(root_dir, dataset_list=dataset_list, 
    image_size=image_scale, data_par='valid', img_transform=my_transform)
train_loader = data.DataLoader(dataset=train_dataset, batch_size=batch_size,
    shuffle=True)
valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
    shuffle=False)
loaders = {'train':train_loader, 'valid':valid_loader}

model = torch.nn.DataParallel(resnet34_Mano(input_option=input_option))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

if init_model:
    load_model_path = os.path.join('data', 'model-' + str(input_option) + '.pth')
    model.load_state_dict(torch.load(load_model_path))
    prev_epoch = 0
else:
    load_model_path = os.path.join(dataset_list[0], 'best_model_ep29.pth')
    loaded_state = torch.load(load_model_path)
    prev_epoch = loaded_state['epoch']
    print('prev_epoch = ', prev_epoch)
    model.load_state_dict(loaded_state['model_state_dict'])
    optimizer.load_state_dict(loaded_state['optimizer_state_dict'])
    scheduler.load_state_dict(loaded_state['scheduler_state_dict'])

total_epoch = prev_epoch + epochs
train_losses = []
valid_losses = []

min_err = 100000000

for epoch in range(epochs):
    print('Epoch {}/{}'.format(prev_epoch + epoch + 1, prev_epoch + epochs))
    print('-' * 10)    
    # Each epoch has a training and validation phase
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train(True)  # Set model to training mode
        else:
            model.train(False)  # Set model to evaluate mode

        running_loss = 0.0
        running_batch = 0

        # Iterate over data.
        for data in loaders[phase]:
            inputs, labels, masks = data["image"], data["param"], data["mask"]
            inputs, labels, masks = inputs.to(device), labels.to(device), masks.to(device)

            bs = inputs.shape[0]

            j3d_gt = labels[:,0:63].reshape((bs, 21,3))
            j2d_gt = labels[:,63:105].reshape((bs, 21,2))
            cam_K = labels[:,105:114].reshape((bs, 3,3))
            cam_R = labels[:,114:123].reshape((bs, 3,3))
            cam_t = labels[:,123:126].reshape((bs, 3,1))
            cam_distCoef = labels[:,126:131]

            x2d, x3d, embedding = model(inputs, j3d_gt[:,0:1,:], cam_K, cam_R, cam_t, cam_distCoef)

            v2d = x2d[:, 21:,:].reshape((bs, 778,2))
            j2d = x2d[:, :21,:].reshape((bs, 21,2))
            j3d = x3d[:, :21,:].reshape((bs, 21,3))
            # --  loss --
            loss_complex = decoder_criterion(
                j3d[5:,:,:], j3d_gt[5:,:,:], j2d[5:,:,:], j2d_gt[5:,:,:], v2d, embedding[:,6:], masks)

            # backward + optimize only if in training phase
            if phase == 'train':
                loss_complex.backward()
                optimizer.step()
                
            # statistics
            running_loss += loss_complex.item()
            running_batch +=1

        epoch_loss = running_loss / running_batch

        if phase == 'train':
            scheduler.step()
        else:
            curr_epoch = prev_epoch + epoch + 1
            print('epoch_loss = ', epoch_loss)
            print('min_err = ', min_err)
            if epoch_loss < min_err:
                min_err = epoch_loss
                print curr_epoch, ' Best model saved!' 
                torch.save({
                    'epoch': curr_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(dataset_list[0], 'best_model_ep' + str(total_epoch) + '.pth'))
        
        if phase == 'train':
            train_losses.append(epoch_loss)
        else:
            valid_losses.append(epoch_loss)
        print('{} Loss: {:.4f}'.format(phase, epoch_loss))
save_loss(train_losses, valid_losses, dataset_list[0], 'mine', 'loss', total_epoch)

torch.save({
    'epoch': total_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
}, os.path.join(dataset_list[0], 'final_model_ep' + str(total_epoch) + '.pth'))