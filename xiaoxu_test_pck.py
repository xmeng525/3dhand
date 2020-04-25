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

from pck_utils import PCK, PCK_3D
from display_utils import display_hand_3d, plot_hand, plot_hand_3d
# 1 use image and joint heat maps as input
# 0 use image only as input
if_save_results = False
if_export_images = False
init_model = False
input_option = 1
batch_size = 5
image_scale = 256

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

template = open('data/template.obj')
content = template.readlines()
template.close()

pck_util = PCK(image_scale)
pck_3d_util = PCK_3D(image_scale)

faces = np.loadtxt('/home/xiaoxu/Documents/rgb2mesh/BootStrapping/code/bs_hand/utils/faces.txt', dtype='i4', delimiter=',')
my_transform = Compose([Scale((image_scale, image_scale), Image.BILINEAR), ToTensor()])
# root_dir = '/mnt/ext_toshiba/rgb2mesh/DataSets/panoptic-toolbox/'
root_dir = '/home/xiaoxu/Documents/rgb2mesh/BootStrapping/panoptic-toolbox'
dataset_list = ['171026_pose3']

if if_save_results:
	os.makedirs(os.path.join(dataset_list[0], 'v3d'), exist_ok=True)   
	os.makedirs(os.path.join(dataset_list[0], 'j3d'), exist_ok=True)   
	os.makedirs(os.path.join(dataset_list[0], 'j2d'), exist_ok=True)

test_dataset = PanopticSet(root_dir, dataset_list=dataset_list, 
	image_size=image_scale, data_par='test', img_transform=my_transform)
testloader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model = torch.nn.DataParallel(resnet34_Mano(input_option=input_option))	
if init_model:
	load_model_path = os.path.join('data', 'model-' + str(input_option) + '.pth')
	model.load_state_dict(torch.load(load_model_path))
else:
	load_model_path = os.path.join(dataset_list[0], 'best_model_ep39.pth')
	loaded_state = torch.load(load_model_path)
	prev_epoch = loaded_state['epoch']
	print('prev_epoch = ', prev_epoch)
	model.load_state_dict(loaded_state['model_state_dict'])
model.eval()

writer_names = open(os.path.join(dataset_list[0], 'image_names.txt'), "w")
for data_idx in range(len(test_dataset.file_name_list[0])):
	scene_frame_name = test_dataset.file_name_list[0][data_idx]
	writer_names.writelines(scene_frame_name[:-4] + '\n')
writer_names.close()

for i, data in enumerate(testloader, 0):
	inputs, labels, masks = data["image"], data["param"], data["mask"]
	inputs, labels_cuda, masks = inputs.to(device), labels.to(device), masks.to(device)

	bs = inputs.shape[0]
	labels = labels.numpy()
	j3d_gt = np.reshape(labels[:,0:63], (bs, 21,3))
	j2d_gt = np.reshape(labels[:,63:105], (bs, 21,2))
	cam_K = np.reshape(labels[:,105:114], (bs, 3,3))
	cam_R = np.reshape(labels[:,114:123], (bs, 3,3))
	cam_t = np.reshape(labels[:,123:126], (bs, 3,1))
	cam_distCoef = labels[:,126:131]
	frame_idx = labels[:,131]
	camera_idx = labels[:, 132]
	people_idx = labels[:, 133]
	hand_idx = labels[:, 134]

	j3d_gt_cuda = labels_cuda[:,0:63].reshape((bs, 21,3))
	j2d_gt_cuda = labels_cuda[:,63:105].reshape((bs, 21,2))
	cam_K_cuda = labels_cuda[:,105:114].reshape((bs, 3,3))
	cam_R_cuda = labels_cuda[:,114:123].reshape((bs, 3,3))
	cam_t_cuda = labels_cuda[:,123:126].reshape((bs, 3,1))
	cam_distCoef_cuda = labels_cuda[:,126:131]

	x2d, x3d, embedding = model(inputs, j3d_gt_cuda[:,0:1,:], cam_K_cuda, 
		cam_R_cuda, cam_t_cuda, cam_distCoef_cuda)  

	v2d = x2d[:, 21:,:].reshape((bs, 778,2)).detach().cpu().numpy()
	v3d = x3d[:, 21:,:].reshape((bs, 778,3)).detach().cpu().numpy()
	j2d = x2d[:, :21,:].reshape((bs, 21,2)).detach().cpu().numpy()
	j3d = x3d[:, :21,:].reshape((bs, 21,3)).detach().cpu().numpy()

	pck_util.add_sample(j2d_gt, j2d)
	pck_3d_util.add_sample(j3d_gt, j3d)

	if if_export_images:
		for idx in range(batch_size):
			fig = plt.figure(1)
			plt.subplot(3, batch_size, idx + 1)
			plt.imshow(inputs[idx, 0:3,:,:].cpu().permute(1,2,0))
			plot_hand(j2d_gt[idx,:,:], 'b*', 0.1)
			plot_hand(j2d[idx,:,:], 'r.', 1.0)
			
			ax2 = plt.subplot(3, batch_size, batch_size + idx + 1, projection='3d')
			plot_hand_3d(ax2, j3d_gt[idx,:,:], 'b*', 1.0)
			plot_hand_3d(ax2, j3d[idx,:,:], 'r.', 0.1)
			
			ax3 = plt.subplot(3, batch_size,  * 2 + idx + 1, projection='3d')
			display_hand_3d(v3d, j3d, mano_faces=faces, ax=ax3, alpha=0.2, 
				edge_color=(255 / 255, 50 / 255, 50 / 255))
		fig.set_size_inches(5 * batch_size, 5 * 3)
		plt.savefig(os.path.join(dataset_list[0], '%08d.png'%i))
		plt.close()

	if if_save_results:
		for idx in range(bs):
			save_name = '%08d_%02d_%02d_%d'%(frame_idx[idx], camera_idx[idx], people_idx[idx], hand_idx[idx])
			np.savetxt(os.path.join(dataset_list[0], 'v3d', save_name + '.v3d'), v3d[idx,:,:])
			np.savetxt(os.path.join(dataset_list[0], 'j3d', save_name + '.j3d'), j3d[idx,:,:])
			np.savetxt(os.path.join(dataset_list[0], 'j2d', save_name + '.j2d'), j2d[idx,:,:])

curve_2d = pck_util.finish()
curve_3d = pck_3d_util.finish()
np.savetxt(os.path.join(dataset_list[0], '2dpck.txt'), curve_2d, fmt='%.4f', delimiter=',')
np.savetxt(os.path.join(dataset_list[0], '3dpck.txt'), curve_3d, fmt='%.4f', delimiter=',')

plt.figure(1)
plt.title('2D PCK')
plt.plot(curve_2d[0:100], label='2dpck')
plt.legend(bbox_to_anchor=(0., 0.32, 1., 0.102), loc=1, borderaxespad=0.)
plt.savefig(os.path.join(dataset_list[0], '2dpck.png'))

plt.figure(2)
plt.title('3D PCK')
plt.plot(curve_3d[0:100], label='3dpck')
plt.legend(bbox_to_anchor=(0., 0.32, 1., 0.102), loc=1, borderaxespad=0.)  
plt.savefig(os.path.join(dataset_list[0], '3dpck.png'))



