

import click
import numpy as np

import torch
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
USE_CUDA = torch.cuda.is_available()
print('CUDA =', USE_CUDA)

from patch_types import *
from descriptor_tfeat import *
from patches import compose_patches
import os, gc, re
from glob import glob
from tqdm import tqdm
import h5py
import tensorboard_logger

LOSS_MARGIN = 2.0
LOSS_ANCHORSWAP = True
CHECKPOINT_FORMAT = 'weights_{n:03d}.hdf5'

class DatasetScene:
	def __init__(self, in_file, mode=PatchMode.INTENSITY):
		self.in_file = in_file
		self.mode = mode

		self.track_lens = in_file["track_lengths"].value.reshape(-1)
		self.track_offsets = in_file["track_offsets"].value.reshape(-1)

		self.track_count = self.track_lens.shape[0]

		self.patches_intensity_link = in_file["intensity_patches"]
		self.patches_depth_link = in_file["depth_patches"]
		self.patches_normals_link = in_file["normals_patches"]

		self.patch_count = self.patches_intensity_link.shape[0]

def open_patch_set(file_path, **extra_args):
	f_in = h5py.File(file_path, 'r')#, #libver='latest')
	return DatasetScene(f_in, **extra_args)

def load_patches(ps, mode, patch_ids='ALL'):

	if patch_ids == 'ALL':
		intensities = ps.patches_intensity_link.value
	else:
		intensities = ps.patches_intensity_link[patch_ids]
	
	if mode == PatchMode.DEPTH:
		if patch_ids == 'ALL':
			depth = ps.patches_depth_link.value
		else:
			depth = ps.patches_depth_link[patch_ids]
	else:
		depth = None

	if mode == PatchMode.NORMALS:
		if patch_ids == 'ALL':
			normals = ps.patches_normals_link.value
		else:
			normals = ps.patches_normals_link[patch_ids]
	else:
		normals = None

	return compose_patches(intensities, depth, normals, intensity_out_type=np.float16)		

class TripletLoader:
	def __init__(self, dir_in, n_triplets, mode=PatchMode.INTENSITY):
		self.num_triplets = n_triplets
		self.mode = mode
		self.num_out_channels = patch_mode_to_channel_count[mode]

		scenes_paths = glob(os.path.join(dir_in, '*.hdf5'))
		scenes_paths.sort()

		self.scenes = [open_patch_set(path, mode=mode) for path in scenes_paths]
		self.num_scenes = len(self.scenes)

		total_patch_count = np.sum([sc.patch_count for sc in self.scenes])
		total_track_count = np.sum([sc.track_count for sc in self.scenes])
		print('Loaded ', total_track_count, ' tracks, ', total_patch_count, 'patches')

def load_all_scenes(tr):
	tr.scene_patches = list(tqdm(
		load_patches(sc, mode=tr.mode)
		for sc in tr.scenes
	))
	gc.collect()

def gen_pairs_from_tracks(lens):
	high = np.int32(1 << 30)
	num = lens.shape[0]
	lens = lens.reshape(-1)
	
	first = np.random.randint(high, size=num) % lens
	diff = np.random.randint(high, size=num) % (lens -1) + 1
	second = (first + diff) % lens
	
	return np.stack((first, second), axis=1)

def gen_positive_pairs(ps, num):
	chosen_tracks = np.random.randint(0, ps.track_count, size=num)
	chosen_track_lens = ps.track_lens[chosen_tracks]
	chosen_track_offsets = ps.track_offsets[chosen_tracks]

	pair_ids = gen_pairs_from_tracks(chosen_track_lens)
	pair_ids += chosen_track_offsets.reshape(-1, 1)
	return pair_ids

def gen_negative_examples(ps, num):
	return np.random.randint(0, ps.patch_count, size=num)

def gen_scene_pairs(num_scenes, num_triplets):
	first_scene = np.random.randint(num_scenes, size=num_triplets)
	second_scene = (first_scene + np.random.randint(num_scenes-1, size=num_triplets)+1) % num_scenes
	return np.stack((first_scene, second_scene), axis=1)

def gen_triplets(tr, num_triplets=None):
	if num_triplets is None:
		num_triplets = tr.num_triplets
		
	# from which scenes will patches be drawn
	# 
	trp_scenes = gen_scene_pairs(tr.num_scenes, num_triplets).astype(np.int32)
	
	trp_positive_pids = np.stack([gen_positive_pairs(sc, num_triplets) for sc in tr.scenes])
	trp_negative_pids = np.stack([gen_negative_examples(sc, num_triplets) for sc in tr.scenes])
		
	tr.trp_scenes = trp_scenes
	tr.trp_positive_pids = trp_positive_pids[trp_scenes[:, 0], range(num_triplets), :]
	tr.trp_negative_pids = trp_negative_pids[trp_scenes[:, 1], range(num_triplets)]

def get_batch(tr, start, batch_size):
	out_size = (batch_size, tr.num_out_channels, 32, 32)
	out_pos_1 = np.empty(out_size, dtype=np.float32)
	out_pos_2 = np.empty(out_size, dtype=np.float32)
	out_neg = np.empty(out_size, dtype=np.float32)

	scenes_map = tr.trp_scenes[start:start+batch_size, :]
	pos_pids = tr.trp_positive_pids[start:start+batch_size, :]
	neg_pids = tr.trp_negative_pids[start:start+batch_size]
	
	for scid in range(tr.num_scenes):
		scene_mask = (scenes_map == scid)
		patches = tr.scene_patches[scid]
		
		scene_mask_pos = scene_mask[:, 0]
		out_pos_1[scene_mask_pos, :, :, :] = patches[pos_pids[scene_mask_pos, 0], :, :, :]
		out_pos_2[scene_mask_pos, :, :, :] = patches[pos_pids[scene_mask_pos, 1], :, :, :]

		scene_mask_neg = scene_mask[:, 1]
		out_neg[scene_mask_neg, :, :, :] = patches[neg_pids[scene_mask_neg], :, :, :]
	
	return (out_pos_1, out_pos_2, out_neg)

class TfLog:
	def __init__(self, out_dir):
		tensorboard_logger.configure(out_dir)
		self.time = 0

	def write(self, name, value):
		tensorboard_logger.log_value(name, value, self.time)

	def advance_time(self):
		self.time += 1


NET_TYPES = {
	'intensity': PatchMode.INTENSITY, 
	'depth': PatchMode.DEPTH, 
	'normals': PatchMode.NORMALS,
}

@click.command()
@click.option('--net_type', type=click.Choice(list(NET_TYPES.keys())), default='intensity')
@click.option('--dataset_dir', type=click.Path(exists=True, dir_okay=True, file_okay=False, resolve_path=True))
@click.option('--out_dir', type=click.Path(exists=False, resolve_path=True))
@click.option('--batch_size', type=int, default=256)
@click.option('--epoch_size', type=int, default=1280000)
@click.option('--epochs', type=int, default=20)
@click.option('--learn_rate', type=float, default=0.1)
@click.option('--learn_rate_decay', type=float, default=1e-6)
@click.option('--weight_decay', type=float, default=1e-4)
def main(net_type, dataset_dir, out_dir, batch_size, epoch_size, epochs, learn_rate, learn_rate_decay, weight_decay):
	mode = NET_TYPES[net_type]

	# out dir
	os.makedirs(out_dir, exist_ok=True)
	print('Writing to ', out_dir)

	# load dset
	dataset = TripletLoader(dataset_dir, epoch_size, mode=mode)
	print('Reading patches from', dataset_dir)
	load_all_scenes(dataset)

	# create net
	net = create_net_for_mode(mode)

	# look for existing checkpoints
	checkpoint_files = glob(os.path.join(out_dir, '*.hdf5'))

	if len(checkpoint_files):
		checkpoint_files.sort()
		checkpoint_to_load = checkpoint_files[-1]

		print('Checkpoint: Resume from:', checkpoint_to_load)
		attrs = load_weights(net, checkpoint_to_load)

		init_epoch = attrs['epoch'] + 1
	else:
		print('Checkpoint: Start from beginning')
		init_epoch = 1

	if USE_CUDA:
		net = net.cuda()

	optimizer = create_optimizer(net, learn_rate, learn_rate_decay, weight_decay)

	tflog = TfLog(out_dir)

	for ep in range(epochs):
		train_epoch(net, dataset, optimizer, batch_size, ep+init_epoch, out_dir, log=tflog)
		gc.collect()


def create_optimizer(net, learn_rate, learn_decay, weight_decay):
	# setup optimizer
	
	optimizer = optim.SGD(net.parameters(), 
		lr=learn_rate,
		momentum=0.9, 
		dampening=0.9,
		weight_decay=weight_decay
	)
	#elif args.optimizer == 'adam':
	# optimizer = optim.Adam(net.parameters(), 
	# 	lr=learn_rate,
	# 	weight_decay=weight_decay
	# )

	optimizer.initial_learn_rate = learn_rate
	optimizer.learn_decay = learn_decay

	return optimizer

def learn_rate_decay(optimizer):
	"""
	Updates the learning rate given the learning rate decay.
	The routine has been implemented according to the original Lua SGD optimizer
	"""
	init_learn_rate = optimizer.initial_learn_rate
	learn_decay = optimizer.learn_decay

	for group in optimizer.param_groups:
		if 'step' not in group:
			group['step'] = 0
		group['step'] += 1
		group['lr'] = init_learn_rate / (1 + group['step'] * learn_decay)

def wrap_torch(value):
	val_torch = torch.from_numpy(value)
	if USE_CUDA:
		val_torch = val_torch.cuda()

	return Variable(val_torch)

def train_epoch(net, dset, optimizer, batch_size, epoch_index, out_dir, log=None):

	gen_triplets(dset)
	net.train()

	batch_num = dset.num_triplets // batch_size

	pbar = tqdm(range(batch_num))
	for batch_idx in pbar:
		batch_start = batch_size * batch_idx

		patches = get_batch(dset, batch_start, batch_size)
		pos_a, pos_b, neg = tuple(wrap_torch(p) for p in patches)

		# output loss on descriptors
		loss = F.triplet_margin_loss(
			net(pos_a), net(pos_b), net(neg),
			margin= LOSS_MARGIN, 
			swap= LOSS_ANCHORSWAP,
		) 

		# compute gradient and update weights
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		learn_rate_decay(optimizer)

		log.write('loss', loss.data[0])
		log.advance_time()

		if batch_idx % 100 == 0:
			pbar.set_description(
				'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch_index, batch_start, dset.num_triplets,
					100. * batch_idx / batch_num,
					loss.data[0]
				)
			)

	chkpt_file = os.path.join(out_dir, CHECKPOINT_FORMAT.format(n=epoch_index))
	save_weights(net, chkpt_file, extra_attrs = dict(epoch=epoch_index) )


if __name__ == '__main__':
	main()
