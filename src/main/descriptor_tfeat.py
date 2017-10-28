
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import h5py

from util import *
from patch_types import *
from patches import compose_patches

import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
USE_CUDA = torch.cuda.is_available()

class TFeatCustom(nn.Module):
	"""TFeat model definition"""
	def __init__(self, input_channels=1):
		super().__init__()
		self.features = nn.Sequential(
			nn.Conv2d(input_channels, 32, kernel_size=8, stride=2),
			nn.Tanh(),
			nn.Conv2d(32, 64, kernel_size=6),
			nn.Tanh(),
		)
		self.classifier = nn.Sequential(
			nn.Linear(64 * 8 * 8, 128),
			nn.Tanh(),
		)

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

def create_net_for_mode(mode):
	n = TFeatCustom(patch_mode_to_channel_count[mode])
	init_weights(n)
	return n

def init_weights(net):
	def iw(m):
		if isinstance(m, nn.Conv2d):
			nn.init.xavier_uniform(m.weight.data, gain=np.sqrt(2.0))
			nn.init.constant(m.bias.data, 0.1)
	net.apply(iw)

def save_weights(net, file_path, extra_attrs = dict()):
	with h5py.File(file_path, 'w', libver='latest') as file:
		weights = file.create_group("weights")

		for w_name, w_value_tr in net.state_dict().items():
			w_value = w_value_tr.cpu().numpy()
			#print(w_name, w_value.shape, w_value.dtype)
			weights.create_dataset(w_name, data=w_value)

		for a_name, a_value in extra_attrs.items():
			file.attrs[a_name] = a_value

def load_weights(net, file_path):
	with h5py.File(file_path, 'r') as file:
		weight_dict = {
			w_name: torch.from_numpy(w_value.value)
			for w_name, w_value in file["weights"].items()
		}

		net.load_state_dict(weight_dict)

		attrs = dict(file.attrs.items())

	return attrs

class TFeatRunner:
	def __init__(self, model_file=None, net_type=PatchMode.INTENSITY, name=''):
		"""
		@param net_type: tanh or relu
		"""
		self.mode = net_type
		net_name = patch_mode_to_name[net_type]
		self.net = create_net_for_mode(net_type)

		if name:
			self.name = name
		else:
			self.name = 'tfeat_' + net_name
		
		if model_file:
			load_weights(self.net, model_file)

		if USE_CUDA:
			print('CUDA is on')
			self.net.cuda()
		
	def evaluate(self, patches, depths=None, normals=None):
		"""
		@param patches: shape [batch x 32 x 32]
		"""
		patches_full = compose_patches(patches, depths, normals, intensity_out_type=np.float32)

		var = Variable(torch.from_numpy(patches_full), volatile=True)

		if USE_CUDA:
			var = var.cuda()

		# evaluate network
		out_val = self.net(var).cpu()
		
		return out_val.data.numpy()

	def describe_patch_list(self, plist):
		
		if hasattr(plist, 'patch_array'):
			patches_32 = plist.patch_array
		else:
			print('Wrong size', plist.name)
	
		depths_32 = None
		normals_32 = None

		if self.mode == PatchMode.DEPTH:
			if hasattr(plist, 'depth_array'):
				depths_32 = plist.depth_array
			else:
				print('Depth net but no depth channel in patch list')
		elif self.mode == PatchMode.NORMALS:
			if hasattr(plist, 'normals_array'):
				normals_32 = plist.normals_array
			else:
				print('Normals net but no normals channels in patch list')

		vals = self.evaluate(patches_32, depths = depths_32, normals = normals_32)

		desc =  Description(self.name, plist, vals)
		frame_add_description(desc)
		return desc
