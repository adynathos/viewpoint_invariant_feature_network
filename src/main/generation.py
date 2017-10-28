from util import *
from patch_types import *

import h5py
import shutil, traceback, gc
from multiprocessing import Pool

def dset_get_random_seq(dset, seq_n, seq_total, frame_count, stride):
	dset_size = len(dset)
	print(dset_size)
	step = dset_size // seq_total
	seq_len = (1+frame_count)*stride
	
	start_idx = seq_n * step + np.random.randint(0, step - seq_len)
	seq = dset.get_sequence(start_idx, frame_count, stride=stride)

	return seq

def dset_get_nth_seq(dset, seq_n, seq_total, frame_count, stride):
	dset_size = len(dset)
	step = dset_size // seq_total
	seq_len = (1+frame_count)*stride
	
	# try:
	start_idx = seq_n * step
	seq = dset.get_sequence(start_idx, frame_count, stride=stride)

	# except Exception as e:
		# print('Sequence failed', dset.config.dataset_name, ':', type(e), e)
	
	return seq

def build_tracks(seq, min_track_length = 2):
	
	pt_classes = [
		[(pt_id, 0)]# (kpoint index, frame index)
		for pt_id in range(seq.frames[0].pt_count)
	]
	
	for idx, fr in enumerate(seq.frames[1:]):
		fr_id = idx+1
		
		for src_idx, dest_idx in fr.gt_to_first.pairs:
			# pair is (idx in fr0, idx in current frame)
			pt_classes[src_idx].append((dest_idx, fr_id))
	
	# remove classes with 1 pt only
	pt_classes = [pl for pl in pt_classes if len(pl) >= min_track_length]
	
	return pt_classes

def build_patches(pt_classes, p_lists):
	b_depth = hasattr(p_lists[0], 'depth_array')
	
	dest_rgb = []
	
	src_rgb = [pl.patch_array for pl in p_lists]
		
	if b_depth:
		dest_depth = []
		dest_normals = []
		
		src_depth = [pl.depth_array for pl in p_lists]
		src_normals = [pl.normals_array for pl in p_lists]
		
	for ptc in pt_classes:
		for (pt_idx, fr_idx) in ptc:
			dest_rgb.append(src_rgb[fr_idx][pt_idx, :, :])
			
			if b_depth:
				dest_depth.append(src_depth[fr_idx][pt_idx, :, :])
				dest_normals.append(src_normals[fr_idx][pt_idx, :, :])
	
	if len(dest_rgb) == 0:
		return None

	dest_rgb = np.stack(dest_rgb).astype(np.uint8)
	
	result = {
		"intensity": dest_rgb
	}
	
	if b_depth:
		result['depth'] = np.stack(dest_depth).astype(np.float16)
		result['normals'] = np.stack(dest_normals).astype(np.float16)
	
	return result


def build_patch_file_name(scene_name, seq_index=0, plist_name=''):
	base_name = '{sc}_{sqi}.hdf5'.format(sc=scene_name, sqi=seq_index)
	
	if plist_name:
		return os.path.join(plist_name, base_name)
	else:
		return base_name

def init_patch_file(hdf_file, scene_name='', seq_index=0, plist_name=''):

	hdf_file.attrs["scene"] = scene_name
	hdf_file.attrs["dataset_sequence"] = seq_index
	hdf_file.attrs["patch_list_name"] = plist_name
	
	hdf_file.create_dataset('track_lengths', (0, 1), maxshape=(None, 1), dtype=np.int32)
	hdf_file.create_dataset('track_offsets', (0, 1), maxshape=(None, 1), dtype=np.int32)
	
	hdf_file.create_dataset('intensity_patches', (0, 32, 32), maxshape=(None, 32, 32), dtype=np.uint8)
	hdf_file.create_dataset('depth_patches', (0, 32, 32), maxshape=(None, 32, 32), dtype=np.float32)
	hdf_file.create_dataset('normals_patches', (0, 32, 32, 3), maxshape=(None, 32, 32, 3), dtype=np.float32)

def append_patches_to_file(hdf_file, track_lengths, patches_intensity, patches_depth=None ,patches_normals=None):
	
	ds_track_lengths = hdf_file['track_lengths']
	ds_track_offsets = hdf_file['track_offsets']
	ds_patches_intensity = hdf_file['intensity_patches']
	
	num_existing_tracks = ds_track_lengths.shape[0]
	num_existing_patches = ds_patches_intensity.shape[0]
	
	num_new_tracks = len(track_lengths)
	num_all_tracks = num_existing_tracks + num_new_tracks
	
	num_new_patches = patches_intensity.shape[0]
	num_all_patches = num_existing_patches + num_new_patches
	
	# add track lengths
	ds_track_lengths.resize(num_all_tracks, axis=0)
	ds_track_lengths[-num_new_tracks:] = track_lengths.reshape(-1, 1)
	
	# add track offsets
	new_track_offsets = np.cumsum(track_lengths)
	new_track_offsets = np.roll(new_track_offsets, 1)
	new_track_offsets[0] = 0
	new_track_offsets += num_existing_patches
	
	ds_track_offsets.resize(num_all_tracks, axis=0)
	ds_track_offsets[-num_new_tracks:] = new_track_offsets.reshape(-1, 1)
	
	# add new patches
	ds_patches_intensity.resize(num_all_patches, axis=0)
	ds_patches_intensity[-num_new_patches:, :, :] = patches_intensity
	
	if patches_depth is not None:
		ds_patches_depth = hdf_file['depth_patches']
		ds_patches_depth.resize(num_all_patches, axis=0)
		ds_patches_depth[-num_new_patches:, :, :] = patches_depth
	
	if patches_normals is not None:
		ds_patches_normals = hdf_file['normals_patches']
		ds_patches_normals.resize(num_all_patches, axis=0)
		ds_patches_normals[-num_new_patches:, :, :, :] = patches_normals

def gen_patches_from_sequence(sq, plist_ids, min_track_length=2):
	
	try:
		# detector
		if 'flat' in plist_ids or 'unwarp' in plist_ids:
			loop(partial(frame_detect_sift_flat, b_cut_patches=True), sq.frames)
			loop(partial(frame_describe_unwarp, b_describe=False, b_cut_patches=True), sq.frames)
		elif 'unwarp_det' in plist_ids:
			loop(frame_detect_sift_unwarp, sq.frames)

		# geometry matching
		seq_find_groundtruth_matches_to_first_frame(sq)

		tracks = build_tracks(sq, min_track_length=min_track_length)
		#pt_labs = build_labels(pt_clss)
		
		patch_collections = dict()
		for plist_id in plist_ids:
			patch_collections[plist_id] = build_patches(tracks, get_patch_lists(sq.frames, [plist_id]))

		return tracks, patch_collections
	
	except Exception as e:
		print('Gen for seq ', sq.config.sequence_name, 'failed:', e)
		traceback.print_exc()
		
		return (None, None)
		#return ([], {plid: None for plid in plist_ids})

# def jobs_for_dsets(dset_list, seq_per_dset):
# 	return [(ds, sq_n, seq_per_dset) for ds in dset_list for sq_n in range(seq_per_dset)]

def write_tracks_to_file(tracks, patch_collections, file_handles_by_plist):
	# failed sequences will return (None, None)
	if (tracks is not None) and len(tracks) > 0:
		track_lens = np.array([len(tr) for tr in tracks], dtype=np.int32)
		#print('Track len avg', np.average(track_lens))
		
		for plist_id in patch_collections.keys():
			of = file_handles_by_plist[plist_id]
			pcol = patch_collections[plist_id]
			
			append_patches_to_file(
				of,
				track_lens,
				pcol['intensity'],
				pcol.get('depth', None),
				pcol.get('normals', None),
			)
			of.flush()


def process_dataset(dset, out_dir, plist_ids=['flat'], min_track_length=2, seq_per_dset = 10, frames_per_seq=16, stride=2):
		
	out_files = {}
	
	b_flat = 'flat' in plist_ids
	b_unw = 'unwarp' in plist_ids
	b_unw_det = 'unwarp_det' in plist_ids

	plist_ids_flatdet = []
	plist_ids_unwdet = []
	if b_flat:
		plist_ids_flatdet.append('flat')
	if b_unw:
		plist_ids_flatdet.append('unwarp')
	if b_unw_det:
		plist_ids_unwdet.append('unwarp_det')

	for plist_id in plist_ids:
		out_name = build_patch_file_name(dset.config.dataset_scene, dset.config.get('dataset_sequence', ''), plist_id)
		out_path = os.path.join(out_dir, out_name)
		os.makedirs(os.path.dirname(out_path), exist_ok=True)
		ensure_file_removed(out_path)
		out_file = h5py.File(out_path, libver='latest')
		init_patch_file(out_file, dset.config.dataset_scene, dset.config.get('dataset_sequence', 0), plist_id)
		
		out_files[plist_id] = out_file
	
	pbar = ProgressBar(seq_per_dset)
	for seq_n in range(seq_per_dset):
		try:
			seq = dset_get_nth_seq(dset, seq_n, seq_per_dset, frame_count=frames_per_seq, stride=stride)
			seq_detect_planes(seq, b_normals=True)

			if b_unw_det:
				seq_unwdet = seq_clone(seq)
				tracks, patch_collections = gen_patches_from_sequence(seq_unwdet, plist_ids=plist_ids_unwdet, min_track_length=min_track_length)
				write_tracks_to_file(tracks, patch_collections, out_files)
				del seq_unwdet

			if b_flat or b_unw:
				tracks, patch_collections = gen_patches_from_sequence(seq, plist_ids=plist_ids_flatdet, min_track_length=min_track_length)
				write_tracks_to_file(tracks, patch_collections, out_files)

			#shutil.rmtree(seq.config.dir_out)
			del seq
			del patch_collections
		except Exception as e:
			print('Sequence ', seq.config.sequence_name, 'failed:', e)
			traceback.print_exc()

		gc.collect()
		pbar += 1
	
	for f in out_files.values():
		f.close()

def gen_patches_main(dset_list, out_dir, plist_ids, proc_count=4, **options):
	
	task = partial(process_dataset, plist_ids=plist_ids, out_dir=out_dir, **options)
	pbar = ProgressBar(len(dset_list))
	
	if proc_count > 1:
		with Pool(proc_count) as P:
			for res in P.imap(task, dset_list, chunksize=1):
				pbar += 1
	else:
		for ds in dset_list:
			task(ds)
			pbar += 1

def show_patch_file(file_path, trid=None, pids=None, save=None, b_depth=True):
	with h5py.File(file_path, 'r') as f_in:
		
		print('Scene:', f_in.attrs['scene'], f_in.attrs['dataset_sequence'])
		print('Extraction method:', f_in.attrs["patch_list_name"])

		track_lens = f_in['track_lengths']
		track_offsets = f_in['track_offsets']
		
		print('Track count:', track_lens.shape[0])
		print('Average track length:', np.mean(track_lens))
		
		patches_intensity_h5 = f_in['intensity_patches']
		patches_depth_h5 = f_in['depth_patches']
		patches_normals_h5 = f_in['normals_patches']
		
		print('Patch count:', patches_intensity_h5.shape[0])

		if trid is not None and trid < track_lens.shape[0]:
		
			tr_len = track_lens[trid, 0]
			pt_ids = np.arange(tr_len) + track_offsets[trid, 0]
		
		else:

			pt_ids = pids

		rows = [
			[patches_intensity_h5[ptid] for ptid in pt_ids]
		]


		if b_depth:
			if patches_depth_h5.shape[0]:
				rows.append([patches_depth_h5[ptid].astype(np.float32) for ptid in pt_ids])

			if patches_normals_h5.shape[0]:
				rows.append([patches_normals_h5[ptid].astype(np.float32) * 0.5 + 0.5 for ptid in pt_ids])

		fig = show_multidim(rows, save=save)

def merge_patch_files(in_file_paths, out_file_path):
	ensure_file_removed(out_file_path)
	os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
	
	with h5py.File(out_file_path, libver='latest') as out_file:
		b_init = True
		
		for in_path in in_file_paths:
			with h5py.File(in_path, 'r', libver='latest') as in_file:
				sc = in_file.attrs["scene"]
				pln = in_file.attrs["patch_list_name"]
				
				if b_init:
					init_patch_file(out_file, sc, 0, pln)
					b_init = False
				
				ds_depth = in_file['depth_patches']
				ds_normals = in_file['normals_patches']
				
				append_patches_to_file(
					out_file,
					in_file["track_lengths"].value,
					in_file["intensity_patches"].value,
					patches_depth = ds_depth.value if ds_depth.shape[0] > 0 else None,
					patches_normals = ds_normals.value if ds_normals.shape[0] > 0 else None,
				)

def merge_scene(from_dir, sc_name, out_name):
	merge_regexp = pp(from_dir, sc_name + '*')
	#merge_regexp = pp(DIR_PATCH_OUT, 'train_01', 'flat', 'chess*')
	merge_paths = glob.glob(merge_regexp)
	merge_paths.sort()
	merge_patch_files(merge_paths, out_name)
