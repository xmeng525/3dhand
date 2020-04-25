import numpy as np

class PCK():
	def __init__(self, max_dist):
		self.max_dist = max_dist
		self.total_joints = 0 
		self.accurate_joints = np.zeros((max_dist))

	def add_sample(self, prediction, GT):
		batch_size = GT.shape[0]
		for dist in range(0, self.max_dist):
			diff2 = (prediction - GT) ** 2
			diff_norm = np.sqrt(diff2[:, :,0] + diff2[:, :, 1])
			result = diff_norm < dist
			self.accurate_joints[dist] += result.sum()
		self.total_joints += batch_size * 21

	def finish(self):
		result = self.accurate_joints / self.total_joints
		self.total_joints = 0 
		self.accurate_joints = np.zeros((self.max_dist))
		return result

class PCK_3D():
	def __init__(self, max_dist, step=1):
		self.max_dist = max_dist
		self.step = step
		self.total_joints = 0 
		self.accurate_joints = np.zeros((int(self.max_dist / self.step)))

	def add_sample(self, prediction, GT):
		batch_size = GT.shape[0]
		for dist in range(0, self.max_dist, self.step):
			diff2 = (prediction - GT) ** 2
			diff_norm = np.sqrt(diff2[:, :,0] + diff2[:, :, 1] + diff2[:, :, 2])
			result = diff_norm < dist
			self.accurate_joints[dist] += result.sum()
		self.total_joints += batch_size * 21

	def finish(self):
		result = self.accurate_joints / self.total_joints
		self.total_joints = 0 
		self.accurate_joints = np.zeros((int(self.max_dist / self.step)))
		return result