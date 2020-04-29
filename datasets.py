import os
from PIL import Image
import numpy as np
import torch
from torch.utils import data

class HandTestSet(data.Dataset):
	def __init__(self, root, img_transform=None):
		self.data_dir = root
		self.img_transform = img_transform
								
	def __len__(self):
		return 3
		
	def __getitem__(self, index):
		imgs = [self.img_transform(Image.open(os.path.join(self.data_dir, '%d.png' % index)).convert('RGB') )]		
		for i in xrange(7):
			imgs.append(self.img_transform(Image.open(os.path.join(self.data_dir, '%d_%d.png' %(index,i))).convert('RGB')))
		imgs = torch.cat(imgs,dim=0)
		
		return imgs

class PanopticSet(data.Dataset):
	def __init__(self, root, dataset_list, image_size, original_size=224, data_par='train',
		img_transform=None, syn_heatmap=True):
		self.root = root
		self.dataset_list = dataset_list
		self.original_size = original_size
		self.image_size = image_size
		self.data_par = data_par
		self.num_total, self.num_list, self.file_name_list, self.frame_list = self.get_len()
		self.img_transform = img_transform
		self.syn_heatmap = syn_heatmap

	def __len__(self):
		return self.num_total

	def __getitem__(self, index):
		scene_index, scene_frame_index = self.get_scene_index(index)
		scene_frame_name = self.file_name_list[scene_index][scene_frame_index]
		data_dir = os.path.join(self.root, self.dataset_list[scene_index], 'cropped_hand')
		arr = scene_frame_name.split('_')
		frame_index = int(arr[0])
		camera_index = int(arr[1])
		people_index = int(arr[2])

		# gt
		gt_path = os.path.join(data_dir, 'gt', scene_frame_name[:-4] + '.txt')
		param = self.adjust_param(torch.Tensor(np.loadtxt(gt_path)), self.original_size)

		if self.syn_heatmap:
			# rgb
			rgb = torch.Tensor(self.img_transform(Image.open(os.path.join(
				data_dir, 'rgb', scene_frame_name)).convert('RGB')))
			# heatmaps
			heatmaps = torch.squeeze(self.compute_heatmaps_from_uv(
				torch.Tensor(np.reshape(np.expand_dims(param[63:105], 0), (1, 21, 2))), 
				self.image_size, self.image_size, 10))
			imgs = torch.cat((rgb, heatmaps), dim=0)
			# mask
			mask = torch.zeros((self.image_size, self.image_size))
		else:
			# rgb
			imgs = [self.img_transform(Image.open(os.path.join(
				data_dir, 'rgb', scene_frame_name)).convert('RGB'))]		
			# heatmaps
			for i in range(7):
				new_file_name = '%08d_%02d_%02d_%d_%02d.png' % (
					frame_index, camera_index, people_index, int(scene_frame_name[-5]), i)
				imgs.append(self.img_transform(Image.open(os.path.join(
					data_dir, 'heatmap', new_file_name)).convert('RGB')))
			imgs = torch.cat(imgs, dim=0)
			# mask
			original_mask = Image.open(os.path.join(data_dir, 'mask', scene_frame_name)).convert('L')
			mask = torch.squeeze(self.img_transform(original_mask))

		sample = {'image': imgs, 'param': param, 'mask': mask}
		return sample

	def get_len(self):
		total_num = 0
		num_list = []
		file_name_list = []
		for dataset_name in self.dataset_list:
			curr_num = 0
			curr_file_name_list = []
			file_names = os.listdir(os.path.join(self.root, dataset_name, 'cropped_hand', 'rgb'))
			frame_list = np.loadtxt(os.path.join(self.root, dataset_name, 'cropped_hand', 
				'partition', self.data_par + '.txt'), dtype='i4')
			print(dataset_name, ' ', self.data_par, 'contains ', len(frame_list), ' frames.')
			file_names.sort()
			for file_name in file_names:
				frame_index = int(file_name.split('_')[0])
				hand_index = int(file_name.split('_')[3][0])
				if frame_index in frame_list and hand_index == 0:
					curr_file_name_list.append(file_name)
					curr_num += 1
			total_num += curr_num
			num_list.append(curr_num)
			file_name_list.append(curr_file_name_list)
			print(dataset_name, ' ', self.data_par, 'contains ', curr_num, ' images.')
		return total_num, num_list, file_name_list, frame_list

	def get_scene_index(self, index):
		idx = 0
		while idx < len(self.dataset_list):
			if index - self.num_list[idx] >= 0:
				index = index - self.num_list[idx]
				idx += 1
			else:
				return idx, index

	def adjust_param(self, param, original_size):
		# cam_K
		cam_K = np.reshape(param[105:114], (3,3))
		cam_K[0, 0] *= self.image_size * 1.0 / original_size
		cam_K[0, 2] *= self.image_size * 1.0 / original_size
		cam_K[1, 1] *= self.image_size * 1.0 / original_size
		cam_K[1, 2] *= self.image_size * 1.0 / original_size

		# j2d
		param[63:105] *= self.image_size * 1.0 / original_size
		param[105:114] = np.reshape(cam_K, (9))
		return param

	def compute_heatmaps_from_uv(self, coord, original_dim, dim, sigma):
		coord = coord * dim / original_dim

		xx, yy = torch.meshgrid([torch.arange(0,dim), torch.arange(0,dim)])

		coord_x = torch.ones((coord.shape[0], coord.shape[1], dim, dim)) * torch.unsqueeze(coord[:,:,1:2], 3).repeat(1,1,dim,dim)
		coord_y = torch.ones((coord.shape[0], coord.shape[1], dim, dim)) * torch.unsqueeze(coord[:,:,0:1], 3).repeat(1,1,dim,dim)

		xx = xx.unsqueeze(0).unsqueeze(0).repeat(coord.shape[0], coord.shape[1], 1, 1).to(dtype=torch.float) - coord_x
		yy = yy.unsqueeze(0).unsqueeze(0).repeat(coord.shape[0], coord.shape[1], 1, 1).to(dtype=torch.float) - coord_y

		dist = xx**2 + yy**2;

		scoremap = torch.exp(-dist / (sigma * sigma));
		return scoremap
