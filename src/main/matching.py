from util import *
from geometry import *
from patch_types import *

import h5py
import gc
from statsmodels.stats.proportion import proportion_confint
from collections import deque
from multiprocessing import Pool
import subprocess

###################################################################################################
# FLANN
###################################################################################################
def create_flann():
	""" Init FLANN object """
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)
	return cv2.FlannBasedMatcher(index_params, search_params)
	
if not ('FLANN_INST' in globals()):
	FLANN_INST = create_flann()

def make_empty_match_result():
	pairs = np.zeros((0, 2), dtype=np.int32)
	distances = np.zeros(0, dtype=np.float32)
	return pairs, distances

def match_descriptors(d1, d2):
	"""
	Mutual FLANN match between descriptor arrays
	"""

	# arguments( "query" descriptors, "train" descriptors)

	# FLANN can crash with empty input
	if d1.shape[0] == 0 or d2.shape[0] == 0:
		return make_empty_match_result()

	try:
		m12 = FLANN_INST.knnMatch(d1, d2, k=2)
		m21 = FLANN_INST.knnMatch(d2, d1, k=2)
	except Exception as e:
		print('FLANN failure', e)
		return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.float32)

	matches_filtered = []
	pairs = []
	distances = []

	for idx, (first, second) in enumerate(m12):
		
		other_pt_idx = first.trainIdx
		other_pair = m21[other_pt_idx]
		other_pt_best_match = other_pair[0]
		other_pt_best_match_idx = other_pt_best_match.trainIdx

		if idx == other_pt_best_match_idx: # and match_passes_filter(first, second) and match_passes_filter(*other_pair):
			matches_filtered.append(first)
			pairs.append((first.queryIdx, first.trainIdx))
			distances.append(first.distance)

	if len(pairs) > 0:
		pairs = np.array(pairs, dtype=np.int32)
		distances = np.array(distances, dtype=np.float32)

		sorting_order = np.argsort(pairs[:, 0])
		pairs = pairs[sorting_order, :]
		distances = distances[sorting_order]
	else:
		pairs, distances = make_empty_match_result()

	return pairs, distances

def intersect_match_lists(descr_matches, gt_matches):
	"""
	returns indices of rows in gt_matches which were found in descr_matches

	pair arrays should be sorted by 1st column
	"""

	idx_gt = 0

	potential_correspondence_descr = np.where(np.isin(descr_matches[:, 0], gt_matches[:, 0]))[0]

	# source (left) point ids for correspondences that might be common
	src_pt_ids = descr_matches[potential_correspondence_descr, 0]

	# find those point ids in gt array
	potential_correspondence_gt = np.searchsorted(gt_matches[:, 0], src_pt_ids, side='left')
	
	# compare the dest (right) point ids
	# which of those potential correspondeces are confirmed
	potential_correspondence_valid = (descr_matches[potential_correspondence_descr, 1] == gt_matches[potential_correspondence_gt, 1])

	#print(potential_correspondence_descr, potential_correspondence_gt, potential_correspondence_valid)

	confirmed_correspondence_descr = potential_correspondence_descr[potential_correspondence_valid]
	confirmed_correspondence_gt = potential_correspondence_gt[potential_correspondence_valid]

	# test
	if np.any(descr_matches[confirmed_correspondence_descr, :] != gt_matches[confirmed_correspondence_gt, :]):
		print('Corr test FAIL')

	return confirmed_correspondence_gt.astype(np.int32)
	#descr_match_valid = np.zeros(descr_matches.shape[0], dtype=np.bool)
	#descr_match_valid[potential_correspondence_descr] = potential_correspondence_valid
	#return descr_match_valid

def match_frame_pair(frA, frB, desc_ids=None):
	"""
	finds groundtruth geometry matches, descriptor matches, and the overlap btw them
	"""
	gt_match = keypoints_match_geometry(frA, frB)
	gt_pairs = gt_match.pairs

	descr_matches = dict()

	if desc_ids is None:
		desc_ids = frB.descriptions.keys()

	for des_key in desc_ids:
		desB = frB.descriptions.get(des_key, None)
		desA = frA.descriptions.get(des_key, None)
		if desA and desB:
			des_pairs, _ = match_descriptors(desA.descriptors, desB.descriptors)

			des_pair_count = len(des_pairs)

			# which pairs from gt were detected in descriptor match
			detected_gt = intersect_match_lists(des_pairs, gt_pairs)

			descr_matches[des_key] = detected_gt
		else:
			print('Missing description', des_key)

	return AttrDict(
		gt_pairs = gt_pairs,
		gt_angles = gt_match.angles,
		descr_redetects = descr_matches,
	)

class AccuracyForDescriptor:
	def __init__(self, desc_id, bin_count):
		self.desc_id = desc_id
		self.num_gt = np.zeros(bin_count, dtype=np.int64)
		self.num_correct = np.zeros(bin_count, dtype=np.int64)

	def add_sample(self, gt_counts, correct_counts):
		self.num_gt += gt_counts
		self.num_correct += correct_counts


class AccuracyAccumulator:
	def __init__(self, bin_size=1.0*np.pi/180.0):
		bin_sq = np.arange(np.pi / bin_size + 1, dtype=np.float32)

		self.bin_size = bin_size
		self.bin_borders = bin_sq * bin_size
		self.bin_centers = (bin_sq[:-1] + 0.5) * bin_size
		self.bin_count = len(self.bin_centers)

		self.acc_for_descriptor = dict()

	def get_for_descriptor(self, desc_id):
		afd = self.acc_for_descriptor.get(desc_id, None)

		if afd is None:
			afd = AccuracyForDescriptor(desc_id, self.bin_count)
			self.acc_for_descriptor[desc_id] = afd

		return afd

RAD_TO_DEG = 180/np.pi

def acc_bin_centers_deg(acc):
	bin_centers_deg = acc.bin_centers * RAD_TO_DEG
	return bin_centers_deg

def acc_add_measurements(acc, pair_match_info):
	"""
	@param pair_match_info: result of match_frame_pair(frame_1, frame_2)
	"""

	angle_bins = np.digitize(pair_match_info.gt_angles, bins=acc.bin_borders)
	gt_count = np.bincount(angle_bins, minlength=acc.bin_count)[:acc.bin_count]

	for did, redetects in pair_match_info.descr_redetects.items():
		corr_count = np.bincount(angle_bins[redetects], minlength=acc.bin_count)[:acc.bin_count]

		afd = acc.get_for_descriptor(did)
		afd.add_sample(gt_count, corr_count)

def acc_save(acc, file_path):
	os.makedirs(os.path.dirname(file_path), exist_ok=True)
	ensure_file_removed(file_path)

	with h5py.File(file_path, 'w', libver='latest') as file:
		file.attrs['bin_size'] = acc.bin_size

		for desc_id, afd in acc.acc_for_descriptor.items():
			path = 'acc/{extr}/{desc}'.format(extr=desc_id[0], desc=desc_id[1])

			g = file.create_group(path)
			g['num_gt'] = afd.num_gt
			g['num_correct'] = afd.num_correct

def acc_load(file_path):
	with h5py.File(file_path, 'r') as file:
		bin_size = file.attrs['bin_size']
		acc_obj = AccuracyAccumulator(bin_size=bin_size)

		acc_g = file['acc']

		for det_id, det_g in acc_g.items():
			for desc, desc_g in det_g.items():
				desc_id = (det_id, desc)

				afd = acc_obj.get_for_descriptor(desc_id)
				afd.add_sample(desc_g['num_gt'].value, desc_g['num_correct'].value)

		return acc_obj

def acc_merge(accA, accB):

	if accA.bin_count != accB.bin_count:
		print('Different number of bins, A:', accA.bin_count, ' B:', accB.bin_count)
		return

	for did, afd in accB.acc_for_descriptor.items():
		accA.get_for_descriptor(did).add_sample(afd.num_gt, afd.num_correct)

def acc_merge_list(acc_list):
	acc_total = AccuracyAccumulator()

	for acc in acc_list:
		acc_merge(acc_total, acc)

	return acc_total

def merge_accs_from_dir(dir_path):
	files = os.listdir(dir_path)
	files = [os.path.join(dir_path, f) for f in files if os.path.splitext(f)[1] == '.hdf5']
	print('Merge', len(files), 'files')
	return acc_merge_list([acc_load(fn) for fn in files])

def undo_mask(vals, mask):
	out = np.empty(mask.shape, vals.dtype)
	out[:] = np.nan
	out[mask] = vals
	return out

def sum_array_shards(ar, shard_size):
	shard_rows = ar.reshape((-1, shard_size))
	shard_sums = np.sum(shard_rows, axis=1)
	return shard_sums


def afd_calc_accuracies(afd, shard=1):
	num_gt = afd.num_gt
	num_correct = afd.num_correct

	if shard > 1:
		num_gt = sum_array_shards(num_gt, shard)
		num_correct = sum_array_shards(num_correct, shard)

	valid_mask = (num_gt > 0)
	count_gt = num_gt[valid_mask]
	count_desc = num_correct[valid_mask]

	zero_mask = count_desc == 0

	interval_center = count_desc / count_gt
	interval_bot, interval_top = proportion_confint(count_desc, count_gt, 0.05, 'wilson')

	interval_top[zero_mask] = 0
	interval_top[interval_top > 1] = 1

	interval_bot[interval_bot < 0] = 0
	interval_bot[zero_mask] = 0
	
	# out = center, error_top, error_bot

	return tuple(map(
		lambda v: undo_mask(v, valid_mask), 
		(
			interval_center,
			interval_top - interval_center,
			interval_center - interval_bot
		)
	))

PATCH_NAMES = {
	'flat': 'original patches',
	'unwarp': 'unwarped-after-detection',
	'unwarp_det': 'unwarped-before-detection',
}

# def DESC_NAMES(desc_id):


def plot_acc(acc, desc_ids = None, shard = 1, plot=None, prefix='', save=None, eval_ds=None):
	#result_values = total_accuracies(acc)
	bin_centers = acc_bin_centers_deg(acc)

	if shard > 1:
		# average center across shard
		bin_centers = sum_array_shards(bin_centers, shard) / shard

	if desc_ids is None:
		desc_ids = sorted(acc.acc_for_descriptor.keys())

	if plot is None:
		fig = plt.figure(figsize=(8, 4))
		plot = fig.add_subplot(1, 1, 1)
		#fig.tight_layout()

		if eval_ds:
			plot.set_title('Eval dset: ' + eval_ds)


	for did_with_name in desc_ids:
		#plot.plot(bin_centers, result_values, '-o', label=desc_id_name(did))
		#bot, mid, top = result_values[did]

		did = did_with_name[:2]

		if len(did_with_name) > 2:
			name = did_with_name[2]
		else:
			name = prefix + desc_id_name(did)

		afd = acc.get_for_descriptor(did)
		interval_center, err_top, err_bot = afd_calc_accuracies(afd, shard=shard)

		plot.errorbar(bin_centers, interval_center, yerr=(err_bot, err_top), fmt='--.', label=name,
			capsize=4, linewidth=1)

	plot.set_xlabel('View angle change [degrees]')
	plot.set_ylabel('Fraction of ground truth pairs\nmatched by feature descriptor')
	
	#plot.xaxis.set_ticks(bin_centers[np.logical_not(np.isnan(interval_center))])
	plot.legend()

	if save:
		save_plot(fig, save)

	return plot

GT_LABELS = {
	('flat', 'sift'): 'Standard detector',
	('unwarp_det', 'sift'): 'Unwarped detector',
}

def plot_acc_gt_single(acc, desc_id = None, shard=1, plot=None, fmt='.'):
	bin_centers = acc_bin_centers_deg(acc)

	if desc_id is None:
		desc_id = next(iter(acc.acc_for_descriptor))
	
	afd = acc.get_for_descriptor(desc_id)

	if shard > 1:
		# average center across shard
		bin_centers = sum_array_shards(bin_centers, shard) / shard
		num_gt = sum_array_shards(afd.num_gt, shard)
	else:
		num_gt = afd.num_gt
	
	mask_nonzero = num_gt > 0

	if plot is None:
		fig = plt.figure()
		plot = fig.add_subplot(1, 1, 1)
		#fig.tight_layout()

	plot.semilogy(bin_centers[mask_nonzero], num_gt[mask_nonzero], fmt, label=GT_LABELS[desc_id])
	#plot.xaxis.set_ticks(bin_centers[mask_nonzero])

	return plot

def plot_acc_gt(acc_result, shard=10, save=None):
	gt_plot = plot_acc_gt_single(acc_result, shard=shard, desc_id=('flat', 'sift'), fmt='s')
	#gt_plot = plot_acc_gt(acc_result, shard=shard, desc_id=('unwarp', 'sift'), plot=gt_plot)
	gt_plot = plot_acc_gt_single(acc_result, shard=shard, desc_id=('unwarp_det', 'sift'), plot=gt_plot, fmt='o')

	gt_plot.set_xlabel('View angle change [degrees]')
	gt_plot.set_ylabel('Number of ground truth matches')

	gt_plot.legend()

	if save:
		save_plot(gt_plot.figure, save)

###################################################################################################
# SLIDING WINDOW EVALUATION
###################################################################################################
		
def windowed_matching(dset, indices, describe_frame_func, win_size=4, acc_save_path=None, out_dir=None, threading=True):

	acc = AccuracyAccumulator()
	window = deque()

	pbar = ProgressBar(len(indices))
	
	if acc_save_path is None and out_dir is not None:
		acc_save_path = pp(out_dir, dset.name() + '.hdf5')

	with thread_Pool(win_size) as pool:
	
		for fr_id in indices:
			new_frame = dset.get_single_frame(fr_id)

			describe_frame_func(new_frame)

			if threading:
				mobjs = pool.imap(
					lambda w_fr: match_frame_pair(w_fr, new_frame),
					window,
				)
			else:
				mobjs = [match_frame_pair(w_fr, new_frame) for w_fr in window]

			for mobj in mobjs:
				acc_add_measurements(acc, mobj)
			
			if acc_save_path:
				acc_save(acc, acc_save_path)

			window.append(new_frame)
			if len(window) > win_size:
				window.popleft()
			
			pbar += 1
			
			gc.collect()
	
	return acc

def windowed_matching_multi(dset, indices, describe_frame_func, win_size=4, acc_save_path=None, out_dir=None, threading=True):
	"""
	Same as windowed_matching but describe_frame_func returns a list of frame variants [flat_det, unwarp_det]
	"""

	acc = AccuracyAccumulator()
	window = deque()
		
# 	def match_task(src_fr, dest_fr, acc):
# 		desc_m = match_frame_pair(src_fr, dest_fr)
# 		acc.accumulate_frame_pair(desc_m)

	pbar = ProgressBar(len(indices))
	
	if acc_save_path is None and out_dir is not None:
		acc_save_path = pp(out_dir, dset.name() + '.hdf5')

	with thread_Pool(win_size) as pool:
	
		for fr_id in indices:
			new_frame = dset.get_single_frame(fr_id)

			new_frame_variants = describe_frame_func(new_frame)

			for var_idx, fr in enumerate(new_frame_variants):
				same_var_from_window = [w[var_idx] for w in window]

				if threading:
					mobjs = pool.imap(
						lambda w_fr: match_frame_pair(w_fr, fr),
						same_var_from_window,
					)
				else:
					mobjs = [match_frame_pair(w_fr, new_frame) for w_fr in same_var_from_window]

				for mobj in mobjs:
					acc_add_measurements(acc, mobj)
			
			if acc_save_path:
				acc_save(acc, acc_save_path)

			window.append(new_frame_variants)
			if len(window) > win_size:
				window.popleft()
			
			pbar += 1
			
			gc.collect()
	
	return acc

def describe_basic(fr):
	frame_detect_planes(fr, b_normals=True)
	frame_all_convex_hulls(fr)
	frame_detect_sift_flat(fr, b_cut_patches=True)


def evaluation_task(taskspec):

	dset = taskspec.dset
	jump_range = taskspec.jump_range
	fixed_seq_len = taskspec.fixed_seq_len
	out_dir = taskspec.out_dir
	describe_func = taskspec.describe_func
	win_size = taskspec.win_size
	threading = taskspec.threading
	multidet = taskspec.multidet
	dset_max_frame_count = taskspec.dset_max_frame_count
	options = taskspec.options


	if jump_range is not None:
		fr_ids = random_walk_sequence(0, len(dset), jump_range[0], jump_range[1])
		if dset_max_frame_count is not None and len(fr_ids) > dset_max_frame_count:
			fr_ids = fr_ids[:dset_max_frame_count]
	else:
		fr_ids = range(fixed_seq_len)

	mfunc = windowed_matching_multi if multidet else windowed_matching

	return mfunc(
		dset,
		fr_ids,
		out_dir = out_dir,
		describe_frame_func = describe_func,
		win_size = win_size,
		threading=threading,
		**options
	)

def evaluation_global(dset_list, out_dir, describe_func, jump_range=None, fixed_seq_len=None, proc_count=4, win_size=4, multiproc=True, multidet=False, dset_max_frame_count=None, **options):
	
	if jump_range is None and fixed_seq_len is None:
		print('Specify either jump_range for random walk or fixed_seq_len for prefix sequence')

	taskspecs = [
		AttrDict(
			dset = dset,
			jump_range = jump_range,
			fixed_seq_len = fixed_seq_len,
			out_dir = out_dir,
			describe_func = describe_func,
			win_size = win_size,
			options = options,
			threading = not multiproc,
			multidet = multidet,
			dset_max_frame_count = dset_max_frame_count,
		)
		for dset in dset_list
	]

	pbar = ProgressBar(len(dset_list))
	accs = []

	if multiproc:
		with Pool(proc_count) as P:
			for acc in P.imap(evaluation_task, taskspecs, chunksize=1):
				pbar += 1
				accs.append(acc)
	else:
		for ts in taskspecs:
			acc = evaluation_task(ts)
			pbar += 1
			accs.append(acc)


	return accs
