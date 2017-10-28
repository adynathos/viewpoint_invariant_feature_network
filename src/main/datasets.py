import numpy as np
import cv2, h5py
import os, operator, re, subprocess
import copy
from attrdict import AttrDict
from os.path import join as pp
from geometry import *

import quaternion

def quat_pose_to_transform(position, quat):
	"""
		Calculates the camera-to-world matrix
			p_w = R p_c + T
		Quat: x y z w
	"""

	# tx ty tz (3 floats) give the 
	# position of the optical center of the color camera 
	# with respect to the world origin 
	# as defined by the motion capture system

	# qx qy qz qw (4 floats) give the 
	# orientation of the optical center of the color camera in form of a unit quaternion 
	# with respect to the world origin as defined by the motion capture system.

	# quaternion contructor takes (w, x, y, z)
	quat_as_q = np.quaternion(quat[3], quat[0], quat[1], quat[2])
	rot_mat = quaternion.as_rotation_matrix(quat_as_q)

	return spatial_transform(
		t = position.reshape((3,1)),
		r = rot_mat,
	)

class RgbdFrame:
	""" 
	Frame with RGBD data, camera parameters and pose 
	RGBD:
		frame.photo
		frame.photo_gray_float
		frame.depth

	Camera intrinsic parameters:
		frame.intrinsic_mat

	Pose as 4x4 transformation matrix:
		frame.world_to_camera
		frame.camera_to_world
	"""
	def __init__(self, photo, depth, name='rgbd'):
		self.name = name
		self.photo = photo
		self.preprocess_depth(depth)

		self.photo_gray = cv2.cvtColor(self.photo, cv2.COLOR_RGB2GRAY)
		self.photo_gray_float = cv2.cvtColor(self.photo.astype(np.float32), cv2.COLOR_RGB2GRAY)

		self.patch_lists = dict()
		self.descriptions = dict()

	def set_intrinsic_mat(self, intrinsic_mat):
		self.focal = intrinsic_mat[0, 0]
		self.intrinsic_mat = intrinsic_mat

	def set_focal(self, focal):
		self.focal = focal
		self.intrinsic_mat = intrinsic_matrix(focal, (photo.shape[1], photo.shape[0]))

	def calc_projection(self):
		self.world_to_camera = spatial_inverse(self.camera_to_world)
		self.projection = self.intrinsic_mat @ self.world_to_camera[:3, :]

	def set_pose_quat_cam_to_world(self, rot_quat, pos):
		self.camera_to_world = quat_pose_to_transform(pos, rot_quat)
		self.calc_projection()

	def set_pose_matrix_cam_to_world(self, pose_transform_matrix):
		self.camera_to_world = pose_transform_matrix
		self.calc_projection()

	def preprocess_depth(self, depth):
		# attempt to remove spikes
		depth = cv2.medianBlur(depth, 5)

		# unify all invalid depth data as NAN
		depth[depth == 0] = np.nan

		self.depth = depth
		
class FrameSequence:
	""" A sequence of frames coming from the same scene """
	def __init__(self, frames = None, config = None, dset= None):
		self.config = config or AttrDict()
		self.title = ''
		self.frames = frames or []
		self.dataset = dset
		
	# def ensure_out_dir(self):
	# 	self.config.dir_out = pp(self.config.dir_out_base, self.config.dataset_name + '_' + self.config.sequence_name)
	# 	os.makedirs(self.config.dir_out, exist_ok=True)

def frame_clone(fr):
	"""
	Clone the frame for another detector
	"""
	cc = copy.copy(fr)
	cc.patch_lists = dict()
	cc.descriptions = dict()
	return cc

def seq_clone(seq):
	"""
	Clone the sequence for another detector
	"""
	cl = copy.copy(seq)
	cl.frames = [frame_clone(fr) for fr in seq.frames]
	return cl

class Dataset:
	def __init__(self, config = None):
		"""
		config:
			dir_datasets
		"""

		self.config = config or AttrDict()

	def __len__(self):
		return self.frame_count

	def get_sequence(self, initial_index, frame_count, stride=1):
		return None

	def get_sequence_random(self, frame_count=16, stride=4):
		init_idx = np.random.randint(0, len(self) - frame_count * stride)
		return self.get_sequence(init_idx, frame_count, stride=stride)

	def set_geometry(self, focal, img_res_xy):
		self.focal = focal
		self.intrinsic_mat = intrinsic_matrix(focal, img_res_xy)

	def name(self):
		return self.config.dataset_name

	def __repr__(self):
		return self.name()

	def get_single_frame(self, frame_idx):
		sq = self.get_sequence(frame_idx, 1)
		fr = sq.frames[0]
		del(sq.frames)
		return fr

###################################################################################################
# 7 Scenes
###################################################################################################

class Dataset7Scenes(Dataset):
	def __init__(self, config, scene_name, seq_index):
		config = AttrDict(config)
		config.dataset_scene = scene_name
		config.dataset_sequence = seq_index
		config.dataset_name = '7scenes_{sn}_{si:02d}'.format(sn=scene_name, si=seq_index)

		config.dir_input = pp(config.dir_datasets, '7scenes', scene_name, 'seq-{si:02d}'.format(si=seq_index))
		# each frame has 2 images and pose file
		self.frame_count = len(os.listdir(config.dir_input)) // 3

		# Principle point (320,240), Focal length (585,585).
		self.set_geometry(585, (640, 480))

		self.depth_scale_value_to_m = 1/1000.0

		super().__init__(config)

	def get_sequence(self, initial_index, frame_count, stride=1):

		idx_list = np.arange(frame_count, dtype=np.int) * int(stride) + int(initial_index)

		frames = []

		for idx in idx_list:
			name_base = pp(self.config.dir_input, 'frame-{fi:06d}.'.format(fi = idx))

			photo = cv2.imread(name_base + 'color.png', cv2.IMREAD_UNCHANGED)[:, :, (2, 1, 0)]
			if photo is None:
				raise Exception('Loaded image is None')

			depth = cv2.imread(name_base + 'depth.png', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
			depth = depth.astype(np.float32)
			depth *= self.depth_scale_value_to_m

			trans_mat = np.loadtxt(name_base + 'pose.txt')

			fr = RgbdFrame(
				name = str(idx),
				photo = photo,
				depth = depth,
			)
			fr.set_intrinsic_mat(self.intrinsic_mat)
			fr.set_pose_matrix_cam_to_world(trans_mat)
			frames.append(fr)

		cfg = AttrDict(self.config)
		cfg.sequence_name = '{si:06d}_{ei:06d}_{s:02d}'.format(si=idx_list[0], ei=idx_list[-1], s=stride)

		seq = FrameSequence(frames, config=cfg, dset=self)
		seq.focal = self.focal
		seq.intrinsic_mat = self.intrinsic_mat
		return seq

RE_7SC_SEQ = re.compile(r'sequence(\d+)')
RE_7SC_SEQ_DIR = re.compile(r'seq-(\d+)')

def load_split_file_7scenes(path):
	seq_ids = []

	with open(path, 'r') as f:
		for line in f:
			m = RE_7SC_SEQ.match(line)
			if m:
				seq_ids.append(int(m.group(1)))

	return seq_ids

def discover_7scenes(config, scenes_train, scenes_test, base_dir = None):
	base_dir = base_dir or pp(config.dir_datasets, '7scenes')

	scenes = [
		(subdir, pp(base_dir, subdir))
		for subdir in sorted(next(os.walk(base_dir))[1]) # list of dir names in base_dir
	]

	train_dsets = []
	test_dsets = []
	
	for scene_name, scene_path in scenes:		
		#train_ids = load_split_file_7scenes(pp(scene_path, 'TrainSplit.txt'))
		#test_ids = load_split_file_7scenes(pp(scene_path, 'TestSplit.txt'))

		seq_dirs = next(os.walk(scene_path))[1]
		seq_ids = [int(RE_7SC_SEQ_DIR.match(dir_name).group(1)) for dir_name in seq_dirs]
		seq_ids.sort()
		seqs = [Dataset7Scenes(config, scene_name, seq_id) for seq_id in seq_ids]

		if scene_name in scenes_train:
			train_dsets += seqs
		elif scene_name in scenes_test:
			test_dsets += seqs

	return dict(train=train_dsets, test=test_dsets)

###################################################################################################
# Synthetic
###################################################################################################

class DatasetSyntheticGen(Dataset):
	def __init__(self, config, dir_in):
		scene_name = os.path.basename(dir_in)
		config = AttrDict(config)
		config.dataset_scene = scene_name
		config.dataset_sequence = ''
		config.dataset_name = 'synthetic_' + scene_name

		config.dir_input = dir_in

		# each frame has: image, depth, normals, pose
		self.frame_count = len(os.listdir(config.dir_input)) // 4

		# fov = 70 deg
		self.intrinsic_from_fov(70., (1024, 768))

		super().__init__(config)

	def intrinsic_from_fov(self, fov_deg, res):
		# w*0.5 / f = tan(fov*0.5)
		# f = w*0.5 / tan(fov*0.5)

		whalf = res[0] * 0.5
		f = whalf / np.tan(np.radians(fov_deg)*0.5)
		self.set_geometry(f, res)

	def get_sequence(self, initial_index, frame_count, stride=1):

		idx_list = np.arange(frame_count, dtype=np.int32)*stride+initial_index
		idx_list = idx_list.astype(np.int32)
		dir_in = self.config.dir_input
		frames = []

		for idx in idx_list:
			name_base = pp(dir_in, '{fi:04d}'.format(fi = idx))

			photo_path = name_base + '_image0032.jpg'
			photo = cv2.imread(photo_path, cv2.IMREAD_UNCHANGED)

			if photo is None:
				print('File fail', photo_path)

			photo = photo[:, :, (2, 1, 0)]

			depth = cv2.imread(name_base + '_depth0032.exr', cv2.IMREAD_UNCHANGED)
			depth = depth[:, :, 0]
			depth[depth == 0] = np.nan
			depth[depth > 1e6] = np.nan

			trans_mat = np.loadtxt(name_base + '_pose.csv')

			fr = RgbdFrame(
				name = str(idx),
				photo = photo,
				depth = depth,
			)
			fr.set_intrinsic_mat(self.intrinsic_mat)
			fr.set_pose_matrix_cam_to_world(trans_mat)
			frames.append(fr)

		cfg = AttrDict(self.config)
		cfg.sequence_name = '{si:06d}_{ei:06d}_{s:02d}'.format(si=idx_list[0], ei=idx_list[-1], s=stride)

		seq = FrameSequence(frames, config=cfg, dset=self)
		seq.focal = self.focal
		seq.intrinsic_mat = self.intrinsic_mat
		return seq

def discover_synthetic_dsets(cfg, base_dir=None):
	base_dir = base_dir or pp(cfg.dir_datasets, 'synthetic')

	subdirs = os.listdir(base_dir)
	subdirs.sort()

	dss = []
	for subd in subdirs:
		ds = DatasetSyntheticGen(cfg, pp(base_dir, subd))
		dss.append(ds)

	return dss

class DatasetSynthRotationZ(Dataset):
	def __init__(self, config, input_image, focal=585, depth=1.0):
		config = AttrDict(config)
		config.dataset_name = 'synthetic_rotZ_' + os.path.splitext(os.path.basename(input_image))[0]

		self.source_image = cv2.imread(input_image, cv2.IMREAD_UNCHANGED)[:, :, (2, 1, 0)]
		self.source_image_size_v = np.array(self.source_image.shape[:2][::-1], dtype=np.int)

		min_wh = int(np.min(self.source_image_size_v) / np.sqrt(2))
		# to prevent cv::rgbd from crashing, make it multiple of block size
		min_wh = int(np.rint(32 * np.floor(min_wh / 32)))

		self.out_size = (min_wh, min_wh)
		self.set_geometry(focal, self.out_size)

		self.depth = np.ones(self.out_size, dtype=np.float32) * depth

		super().__init__(config)

	def get_sequence(self, initial_angle = 0, final_angle = np.pi*0.5, frame_count=16):

		angles = np.linspace(initial_angle, final_angle, num=frame_count)

		frames = []

		Al, Ar = patch_cutting_affine_matrix(0.5 * self.source_image_size_v, np.array(self.out_size))

		for angle in angles:
			R = rot_around_z(angle)

			fr = RgbdFrame(
				name = '{a:.2f}'.format(a=angle),
				photo = cv2.warpAffine(self.source_image, (Al @ R @ Ar)[:2, :], self.out_size, cv2.INTER_LINEAR, cv2.BORDER_REPLICATE),
				depth = self.depth,
			)
			fr.set_intrinsic_mat(self.intrinsic_mat)
			fr.set_pose_matrix_cam_to_world(spatial_transform(r=R.T))
			frames.append(fr)

		cfg = AttrDict(self.config)
		cfg.sequence_name = '{si:.2f}_{ei:.2f}_{c}'.format(si=angles[0], ei=angles[-1], c = frame_count)

		seq = FrameSequence(frames, config=cfg, dset=self)
		seq.focal = self.focal
		seq.intrinsic_mat = self.intrinsic_mat
		return seq

###################################################################################################
# Freiburg
# http://vision.in.tum.de/data/datasets/rgbd-dataset/download
###################################################################################################

class DatasetFreiburg(Dataset):
	"""
		freiburg3_long_office_household
		http://vision.in.tum.de/data/datasets/rgbd-dataset/download
	"""

	class TimeseriesCollection:
		""" 
			Images in Freiburg dataset are not indexed by numbers but by timestamps
			RGB and depth images have different timestamps, so we find the closest ones in time
		"""

		def __init__(self, timepoints):
			self.index = timepoints

		def get_nearby_point_idx(self, time_point):
			next_idx = np.searchsorted(self.index, time_point)

			if next_idx == 0:
				return 0
			if next_idx == self.index.shape[0]:
				return (self.index.shape[0] - 1)

			diff_to_next = self.index[next_idx] - time_point
			diff_to_prev = time_point - self.index[next_idx-1]

			return next_idx if diff_to_next < diff_to_prev else (next_idx-1)

	class TimeseriesImageCollection(TimeseriesCollection):

		def __init__(self, directory):
			self.directory = directory

			self.pairs = [
				(
					float(os.path.splitext(filename)[0]), 
					os.path.join(self.directory, filename),
				) 
				for filename in os.listdir(directory)
			]
			self.pairs.sort()
			
			index = np.fromiter(map(operator.itemgetter(0), self.pairs), dtype=np.float64)
			super().__init__(index)

		def get_nearby_filepath(self, time_point):
			return self.pairs[self.get_nearby_point_idx(time_point)][1]

		def get_sequence(self, initial_time_point, count, stride=1):
			idx = self.get_nearby_point_idx(initial_time_point)
			return self.pairs[idx : min(idx+count, len(self.pairs)-1) : stride]

	class TimeseriesArray(TimeseriesCollection):
		def __init__(self, timepts, values):
			super().__init__(timepts)
			self.values = values

		def get_nearby_value(self, time_point):
			return self.values[self.get_nearby_point_idx(time_point), :]

	def __init__(self, config):
		config = AttrDict(config)
		config.dataset_name = 'freiburg'
		config.dir_input = pp(config.dir_datasets, 'freiburg', 'rgbd_dataset_freiburg3_long_office_household')
		config.dir_input_img = pp(config.dir_input, 'rgb')
		config.dir_input_depth = pp(config.dir_input, 'depth')

		self.set_geometry(537.3, (640, 480))

		# "The depth images are scaled by a factor of 5000, 
		# i.e., a pixel value of 5000 in the depth image corresponds to a distance of 1 meter from the camera"
		self.depth_scale_value_to_m = 1/5000.0

		# Load images ordered by time
		self.ts_img = DatasetFreiburg.TimeseriesImageCollection(config.dir_input_img)
		self.ts_depth = DatasetFreiburg.TimeseriesImageCollection(config.dir_input_depth)

		# Load poses
		self.time_and_pose_array = np.loadtxt(pp(config.dir_input, 'groundtruth.txt'))
		self.ts_pose = DatasetFreiburg.TimeseriesArray(self.time_and_pose_array[:, 0], self.time_and_pose_array[:, 1:])

		super().__init__(config)

	def get_sequence(self, initial_time_point, frame_count, stride=1):
		rgb_sequence = self.ts_img.get_sequence(initial_time_point, frame_count, stride=stride)
		
		frames = []
		for time_point, rgb_filepath in rgb_sequence:
			depth_filepath = self.ts_depth.get_nearby_filepath(time_point)
			
			# Load rgb photo
			photo = cv2.cvtColor(cv2.imread(rgb_filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH), cv2.COLOR_BGR2RGB)
			
			# Load depth
			depth = cv2.imread(depth_filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
			# and apply scale
			depth = depth.astype(np.float32)
			depth *= self.depth_scale_value_to_m
			
			# Geometry transform
			pose = self.ts_pose.get_nearby_value(time_point)

			fr = RgbdFrame(
				name = str(time_point),
				photo = photo,
				depth = depth,
			)
			fr.set_intrinsic_mat(self.intrinsic_mat)
			fr.set_pose_quat_cam_to_world(rot_quat=pose[3:], pos=pose[:3])
			frames.append(fr)


		cfg = AttrDict(self.config)
		cfg.sequence_name = str(initial_time_point).replace('.', '') + '_' + str(frame_count) + '_' + str(stride)

		seq = FrameSequence(frames, config=cfg, dset=self)
		seq.focal = self.focal
		seq.intrinsic_mat = self.intrinsic_mat
		return seq

	DILATE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))	

	@classmethod
	def fix_holes(cls, depth_img_holed):
		depth_img_dilated = cv2.dilate(depth_img_holed, cls.DILATE_KERNEL, None, (-1, -1), 7)

		hole_mask = (depth_img_holed == 0.0)

		depth_img_fixed = depth_img_holed.copy()
		depth_img_fixed[hole_mask] = depth_img_dilated[hole_mask]
		
		return depth_img_fixed

###################################################################################################
# COLMAP reconstructions
###################################################################################################

class DatasetColmap(Dataset):
	def __init__(self, config, name, dirname):
		config = AttrDict(config)
		config.dataset_scene = name
		config.dataset_name = name

		config.dir_input = pp(config.dir_datasets, 'architectural', dirname)
		config.dir_input_imgs = pp(config.dir_input, 'images')
		self.paths_imgs = os.listdir(config.dir_input_imgs)
		self.paths_imgs.sort()

		config.dir_input_depth = pp(config.dir_input, 'depth_maps')
		self.paths_depths = os.listdir(config.dir_input_depth)
		self.paths_depths.sort()

		#config.dir_input_normals = pp(config.dir_input, 'normal_maps')
		#paths_normals = os.listdir(config.dir_input_normals)
		#paths_normals.sort()
		
		# each frame has 2 images
		self.frame_count = len(os.listdir(config.dir_input_imgs))

		super().__init__(config)

		self.load_cameras()

	def load_cameras(self):
		self.poses_cam_to_world = []

		# Example file:
		# # Camera list with one line of data per camera:
		# #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
		# # Number of cameras: 330
		# 330 SIMPLE_RADIAL 1404 936 967.328 702 468 -0.0558808
		# 329 SIMPLE_RADIAL 1404 936 950.169 702 468 -0.0521315
		# 328 SIMPLE_RADIAL 1404 936 965.562 702 468 -0.0542226
		# 327 SIMPLE_RADIAL 1404 936 952.185 702 468 -0.0531098
		intrinsics = dict()

		with open(pp(self.config.dir_input, 'cameras', 'cameras.txt'), 'r') as f_cameras:
			is_pose_line = True
			for line in f_cameras:
				if line[0] == '#':
					pass # commant
				else:
					#print(line)
					params = line.split()
					# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
					cam_id = int(params[0])

					# model, width, height -> not needed
					# 1 2 3

					# params[] = focal, center_x, center_y, distortion?
					focal = float(params[4])
					center_x = float(params[5])
					center_y = float(params[6])

					intrinsics[cam_id] = intrinsic_matrix(focal, (center_x*2, center_y*2))

		frame_defs = {}

		with open(pp(self.config.dir_input, 'cameras', 'images.txt'), 'r') as f_poses:
			is_pose_line = True
			for line in f_poses:
				if line[0] == '#':
					pass # commant
				elif is_pose_line:
					#print(line)
					params = line.split()
					# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
					# img id
					img_id = int(params[0])
					# qw qx qy qz
					quat_rot = np.array(list(map(float, params[1:5])))
					# but we want qx qy qz qw
					quat_rot = np.roll(quat_rot, -1)
					# tx ty tz
					t = np.array(list(map(float, params[5:8])))
					# camera_id
					cam_id = int(params[8])
					intrinsic_mat = intrinsics[cam_id]

					# file name
					file_name = params[9]

					pose_mat = quat_pose_to_transform(quat=quat_rot, position=t)
					pose_mat = spatial_inverse(pose_mat)

					frame_defs[img_id] = AttrDict(
						intrinsic_mat = intrinsic_mat,
						pose_cam_to_world = pose_mat,
					)

				is_pose_line = not is_pose_line

		self.frame_defs = list(frame_defs.items())
		self.frame_defs.sort()
		self.frame_defs = [fd for _, fd in self.frame_defs]

	def get_sequence(self, initial_index, frame_count, stride=1):

		idx_list = np.arange(frame_count)*stride+initial_index
		frames = []

		for idx in idx_list:
			photo = cv2.imread(pp(self.config.dir_input_imgs, self.paths_imgs[idx]), cv2.IMREAD_UNCHANGED)[:, :, (2, 1, 0)]

			depth = read_colmap_img_file(pp(self.config.dir_input_depth, self.paths_depths[idx]))
			depth = cv2.resize(depth, photo.shape[:2][::-1])
			depth[depth < 0] = np.nan

			fr = RgbdFrame(
				name = str(idx),
				photo = photo,
				depth = depth,
			)
			
			frame_def = self.frame_defs[idx]
			fr.set_intrinsic_mat(frame_def.intrinsic_mat)
			fr.set_pose_matrix_cam_to_world(frame_def.pose_cam_to_world)

			#fr.set_intrinsic_mat(self.intrinsic_mat)
			#fr.set_pose_matrix_cam_to_world(self.poses_cam_to_world[idx])
			#fr.params = poses[idx][2]
			frames.append(fr)

		cfg = AttrDict(self.config)
		cfg.sequence_name = '{si:06d}_{ei:06d}_{s:02d}'.format(si=idx_list[0], ei=idx_list[-1], s=stride)

		seq = FrameSequence(frames, config=cfg, dset=self)
		seq.focal = frames[0].focal #self.focal
		seq.intrinsic_mat = frames[0].intrinsic_mat #self.intrinsic_mat
		return seq

class DatasetSouthBuilding(DatasetColmap):
	def __init__(self, config):
		super().__init__(config, name='south-building', dirname='south-building-sml')
		self.set_geometry(1280, (1536, 1152))

class DatasetPersonHall(DatasetColmap):
	def __init__(self, config):
		super().__init__(config, name='person-hall', dirname='person-hall-sml')

class DatasetGerrardHall(DatasetColmap):
	def __init__(self, config):
		super().__init__(config, name='gerrard-hall', dirname='gerrard-hall-sml')

class DatasetGrahamHall(DatasetColmap):
	def __init__(self, config):
		super().__init__(config, name='graham-hall-exterior', dirname='graham-hall-exterior-sml')

###################################################################################################
# Saving and loading processed sequences
###################################################################################################

def save_rgbd_frames_to_hdf(sequence):
	file_path = pp(sequence.config.dir_out, sequence.config.dataset_name + '.hdf5')
	
	# hdf5 does not overwrite files
	if os.path.exists(file_path):
		os.remove(file_path)
	
	#print('Saving to', file_path)


	with h5py.File(file_path, libver='latest') as out:
				
		out.create_dataset('intrinsic', data=sequence.frames[0].intrinsic_mat)
		
		# Configs
		cfg_group = out.create_group("config")
		cfg_group.attrs["frame_count"] = len(sequence.frames)
		for key, val in sequence.config.items():
			cfg_group.attrs[key] = val

		# Frames
		for fid, frame in enumerate(sequence.frames):
			fgroup = out.create_group("frame_{n}".format(n=fid))
			
			fgroup.attrs['name'] = frame.name
			fgroup.attrs['focal'] = frame.focal

			fgroup.create_dataset("rgb", data=frame.photo)
			fgroup.create_dataset("depth", data=frame.depth)
			fgroup.create_dataset("pose_cam_to_world", data=frame.camera_to_world)

def load_sequence_from_hdf(file_path):
	#print('Loading from', file_path)
	
	with h5py.File(file_path, mode='r', libver='latest') as in_file:

		cfg_group = in_file["config"]
		cfg = AttrDict(cfg_group.attrs)
		#print('cfg', cfg)
		frame_count = cfg["frame_count"]
		intrinsic = in_file["intrinsic"].value

		frames = []

		for fid in range(frame_count):
			fgroup = in_file["frame_{n}".format(n=fid)]

			frame = RgbdFrame(
				name = fgroup.attrs['name'],
				photo = fgroup['rgb'].value[:, :, (2, 1, 0)],
				depth = fgroup['depth'].value,
			)

			try:
				frame.normals = fgroup['normals'].value
				frame.point_cloud = fgroup['points'].value
				frame.plane_map = fgroup['plane_map'].value
				frame.plane_coefficients = fgroup['plane_coefficients'].value[:, 0, :]

				frame.set_intrinsic_mat(intrinsic)
			except Exception as e:
				print('Frame', fid, '- No geometry information:', e)

			frames.append(frame)

		seq = FrameSequence(frames, config=cfg)
		seq.intrinsic_mat = intrinsic

	return seq

def load_plane_estimations(sequence):
	file_path = pp(sequence.config.dir_out, sequence.config.dataset_name + '.hdf5')

	#print('Loading from', file_path)
	
	with h5py.File(file_path, libver='latest') as in_file:

		for fid, frame in enumerate(sequence.frames):
			fgroup = in_file["frame_{n}".format(n=fid)]
			
			frame.normals = fgroup['normals'].value
			frame.point_cloud = fgroup['points'].value
			frame.plane_map = fgroup['plane_map'].value
			
			try:
				frame.plane_coefficients = fgroup['plane_coefficients'].value[:, 0, :]
			except Exception as e:
				print('Faulty planes')

def read_colmap_img_file(path):
	with open(path, 'rb') as f_in:
		
		# read header
		header = f_in.read(32)
		
		# find header size
		SEP = '&'
		SEP_B = ord(SEP)
		offset = 0
		seps = 3
		while seps > 0:
			if header[offset] == SEP_B:
				seps -= 1
			offset += 1
		
		# read header to get image dimensions
		header_str = header[:offset].decode()
		dims = [int(v) for v in header_str.split(SEP) if v]
		
		
		# move read position after header
		f_in.seek(offset)
		file_data = np.fromfile(f_in, dtype=np.float32)
		
		if dims[2] == 1:
			return file_data.reshape((dims[1], dims[0]))
		else:
			# the file shape is (3, H, W)
			file_data = file_data.reshape((dims[2], dims[1], dims[0]))
			# swap axes to (H, W, 3)
			return 	file_data.swapaxes(0, 2).swapaxes(0,1)
			
			shape = (dims[2], dims[1], dims[0]) if dims[2] > 1 else (dims[1], dims[0])
		
		return file_data.reshape(shape)

def displayable_depth(depth_map):
	vals = depth_map.flatten()

	vals = vals[np.logical_not(np.isnan(vals))]
	span_ends = np.percentile(vals, [10, 90])

	out = depth_map.copy()

	# disable warning:
	# RuntimeWarning: invalid value encountered in less
	old_warn_status = np.seterr(invalid='ignore')
	out[out < span_ends[0]] = span_ends[0]
	out[out > span_ends[1]] = span_ends[1]
	np.seterr(**old_warn_status)

	return out

def summarize_seq(seq, b_save=True):
	photo_0 = seq.frames[0].photo
	dep_0 = displayable_depth(seq.frames[0].depth)
	photo_n = seq.frames[-1].photo
	dep_n = displayable_depth(seq.frames[-1].depth)

	show_multidim(
		[[photo_0, photo_n, dep_0, dep_n]], 
		col_titles = ["frame 0", "frame N", "depth 0", "depth N"], 
		save=pp(seq.config.dir_out, '01_dataset_overview.jpg') if b_save else False,
	)

def random_walk_sequence(start, end, min_step, max_step):
	length = end - start

	steps = np.random.randint(min_step, max_step+1, size=length//min_step+1)

	offsets = np.cumsum(steps)
	cut_idx = np.searchsorted(offsets, length)
	offsets = offsets[:cut_idx]

	offsets += start

	return offsets

