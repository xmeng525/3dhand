import os
import numpy as np
import torch
from matplotlib import pyplot as plt

def save_loss(train_loss, valid_loss, folder, model_name, name, ep):
	np.savetxt(os.path.join(folder, 'train_' + name + '_' + model_name + '_' + str(ep) + '.txt'), 
		train_loss, fmt='%.4f', delimiter=',')
	np.savetxt(os.path.join(folder, 'valid_' + name + '_' + model_name + '_' + str(ep) + '.txt'), 
		valid_loss, fmt='%.4f', delimiter=',')

def plot_loss(train_loss, valid_loss, folder):
	fig = plt.figure()
	plt_train_loss = plt.plot(train_loss, 'r', label="train_loss")
	plt_valid_loss = plt.plot(valid_loss, 'b', label="valid_loss")
	plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, borderaxespad=0.)
	plt.savefig(os.path.join(folder, 'loss.png'))

class OxfordLoss(torch.nn.Module):
	def __init__(self):
		super(OxfordLoss, self).__init__()
		self.a_2d = 1
		self.a_3d = 1
		self.a_mask = 100
		self.a_reg = 10
		self.N = 778

	def view(x):
		return x.view(x.size(0),-1)

	def mask_loss(self, mask, verts_2d):
		batch_size = verts_2d.shape[0]
		counter = 0
		verts_2d = verts_2d + 0.5
		verts_2d = torch.clamp(verts_2d, min=0, max=223)
		verts_2d = verts_2d.type(torch.LongTensor)
		
		for batch_idx in range(batch_size):
			in_mask = mask[batch_idx, verts_2d[batch_idx,:, 1], verts_2d[batch_idx, :, 0]]
			counter += torch.mean(in_mask)
		return 1.0 - counter / (batch_size * 1.0)

	def forward(self, j3d, j3d_gt, j2d, j2d_gt, v2d, param, mask):
		if j3d_gt is not None:
			loss_3d = torch.mean((j3d - j3d_gt)**2)
		else:
			loss_3d = 0
		loss_2d = torch.mean(torch.abs(j2d - j2d_gt))
		loss_mask = self.mask_loss(mask, v2d)

		if param is not None:
			loss_reg = torch.mean(param[:, 6:12]**2) + 10000 * torch.mean(param[:, 12:22] **2)
		else:
			loss_reg = 0
		loss = self.a_2d * loss_2d + self.a_3d * loss_3d + self.a_mask * loss_mask + self.a_reg * loss_reg
		# print('------------OxfordLoss--------------')
		# print('loss_2d = ', self.a_2d * loss_2d)
		# print('loss_3d = ', self.a_3d * loss_3d)
		# print('loss_mk = ', self.a_mask * loss_mask)
		# print('loss_rg = ', self.a_reg * loss_reg)
		return loss

class WeightedLoss(torch.nn.Module):
	def __init__(self):
		super(WeightedLoss, self).__init__()
		self.pose = 1.0
		self.betas = 10.0

	def forward(self, embedding, embedding_gt):		
		loss_pose = torch.mean((embedding[:,0:48] - embedding_gt[:,0:48])**2)
		loss_betas = torch.mean((embedding[:,48:58] - embedding_gt[:,48:58])**2)

		loss = self.pose * loss_pose + self.betas * loss_betas
		# print '-------------WeightedLoss-------------'
		# print 'loss_pose = ', self.pose * loss_pose
		# print 'loss_betas = ', self.betas * loss_betas
		return loss