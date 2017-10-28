from util import * 
from geometry import *
from keypoint_projection import *
from cyvlfeat.sift import sift as vlfeat_sift

DETECTOR_PEAK_THR = 1

###################################################################################################
# Original frames
###################################################################################################

def frame_detect_sift_flat(frame, b_cut_patches=True):
	"""
	SIFT detector & descriptor on unprocessed frames
	"""
	sift_pts, sift_descs = vlfeat_sift(
		frame.photo_gray_float,
		peak_thresh = DETECTOR_PEAK_THR,
		compute_descriptor = True,
		float_descriptors = True,
	)

	frame.kpt_locs = sift_pts[:, (1, 0)] # reorder columns to get (X, Y)
	frame.kpt_sizes = sift_pts[:, 2].reshape((-1, 1))
	frame.kpt_orientations = sift_pts[:, 3].reshape((-1, 1))
	frame.original_sift_descriptors = sift_descs
	frame.pt_count = sift_pts.shape[0]

	frame_derive_keypoint_geometry(frame)

	if b_cut_patches:
		patches_rgb, patches_depth, patches_normal = cut_patches(
			frame.photo_gray_float,
			frame.kpt_locs,
			frame.kpt_sizes,
			frame.kpt_orientations,
			depth_and_normals = (frame.depth, frame.normals)
		)

		plist = PatchList('flat', patches_rgb)
		plist.set_depth(patches_depth)
		plist.set_normals(patches_normal)
	else:
		plist = PatchList('flat', [])

	frame_add_patch_list(frame, plist)

	desc = Description('sift', plist, frame.original_sift_descriptors)
	frame_add_description(desc)

def seq_detect_sift_flat(seq, b_cut_patches=True, b_progress=False):
	parallel_process(
		partial(frame_detect_sift_flat, b_cut_patches=b_cut_patches),
		seq.frames,
		disp_progress = b_progress,
	)

###################################################################################################
# Unwarp before detection
###################################################################################################
z = np.zeros((0, 1), dtype=np.float32)
EMPTY_DET_AND_DESC = (np.zeros((0, 2), dtype=np.float32), z, z, np.zeros((0, 128), dtype=np.float32), [])

class PlaneException(Exception):
	pass

MIN_NORMAL_Z = 0.2
MIN_KPT_PERCENTAGE = 0.003
MIN_SURFACE_RELATIVE = 0.007
MAX_PLANE_COUNT = 16

def select_planes_by_surface(frame, min_surface_fraction=MIN_SURFACE_RELATIVE):
	
	idx_planes_selected = [
		idx for idx in frame.planes_by_surface
		if (-frame.plane_coefficients[idx, 2]) > MIN_NORMAL_Z and frame.planes[idx].area > min_surface_fraction
	]
	if len(idx_planes_selected) > MAX_PLANE_COUNT:
		idx_planes_selected = idx_planes_selected[:MAX_PLANE_COUNT]
	
	#print('selected planes', len(idx_planes_selected), '/', plane_count)
	#print([surface[idx] for idx in idx_planes_selected[:5]])
	return idx_planes_selected

def plane_unwarp_detect_and_describe(plane_obj):
	"""
	SIFT detector & descriptor on an unwarped plane
	"""
	frame = plane_obj.frame
	plane_id = plane_obj.plane_id
	unw_img = plane_unwarp(plane_obj)

	if unw_img is None:
		return EMPTY_DET_AND_DESC

	try:
		vl_pts_detected, initial_descs = vlfeat_sift(
			unw_img, 
			peak_thresh = DETECTOR_PEAK_THR,
			compute_descriptor = True,
			float_descriptors = True,
		)

		if vl_pts_detected.shape[0] == 0:
			return EMPTY_DET_AND_DESC

		Hinv = np.linalg.inv(plane_obj.homography)
		glob_locs, glob_sizes, glob_orients = homography_apply_to_keypoints(
			Hinv,
			vl_pts_detected[:, (1, 0)],
			vl_pts_detected[:, 2],
			vl_pts_detected[:, 3],
		)

		# filter valid points: inside frame image & lie on the right plane
		glob_locs_int = np.rint(glob_locs).astype(np.int32)

		frame_size = img_size(frame.photo)
		glob_patch_rads = 6 * glob_sizes

		valid_pt_mask = (glob_patch_rads <= glob_locs_int[:, 0]) & (glob_locs_int[:, 0] < frame_size[0]-glob_patch_rads)
		valid_pt_mask &= (glob_patch_rads <= glob_locs_int[:, 1]) & (glob_locs_int[:, 1] < frame_size[1]-glob_patch_rads)
		valid_pt_ids = np.where(valid_pt_mask)[0]

		glob_locs_int_in_frame = glob_locs_int[valid_pt_ids, :]
		glob_plane_ids = frame.plane_map[glob_locs_int_in_frame[:, 1], glob_locs_int_in_frame[:, 0]]

		valid_pt_ids = valid_pt_ids[ np.where(glob_plane_ids == (plane_id+1))[0] ]

		if len(valid_pt_ids) == 0:
			return EMPTY_DET_AND_DESC

		# ---
		glob_locs = glob_locs[valid_pt_ids, :]
		glob_sizes = glob_sizes[valid_pt_ids].reshape((-1, 1))
		glob_orients = glob_orients[valid_pt_ids].reshape((-1, 1))

		# calculate descriptors
		vl_defs = vl_pts_detected[valid_pt_ids, :]
		init_descs = initial_descs[valid_pt_ids, :]

		# cut patches
		patches = cut_patches(
			unw_img,
			vl_defs[:, (1, 0)],
			vl_defs[:, 2],
			vl_defs[:, 3],
		)

		return (glob_locs, glob_sizes, glob_orients, init_descs, patches)
	except Exception as e:
		print('Detect and describe:', e)
		return EMPTY_DET_AND_DESC

def frame_detect_sift_unwarp(frame, b_cut_patches=True):
	
	plane_ids = select_planes_by_surface(frame)
	det_outs = []

	planes_cancelled = 0
	for plid in plane_ids:
		try:
			out = plane_unwarp_detect_and_describe(frame.planes[plid])
			if len(out[0]) > 0:
				det_outs.append(out)
		except PlaneException as e:
			# print('Cancel plane:', e)
			planes_cancelled += 1

	if planes_cancelled > 0:
		print('Planes cencelled:', planes_cancelled, '/', len(plane_ids))

	if not det_outs:
		frame.pt_count = 0
		frame.kpt_locs = np.zeros((0, 2), dtype=np.float32)
		frame.kpt_sizes = np.zeros(0, dtype=np.float32)
		frame.kpt_orientations = np.zeros(0, dtype=np.float32)
		frame.original_sift_descriptors = np.zeros((0, 128), dtype=np.float32)

	def merge(tuple_idx):
		return np.concatenate([d[tuple_idx] for d in det_outs], axis=0)

	frame.kpt_locs = merge(0)
	frame.kpt_sizes = merge(1)
	frame.kpt_orientations = merge(2)
	frame.original_sift_descriptors = merge(3)
	frame.pt_count = frame.kpt_locs.shape[0]

	# patches
	patch_sets = [d[4] for d in det_outs]
	patch_sets_merged = [p for pl in patch_sets for p in pl]

	plist = PatchList('unwarp_det', patch_sets_merged)
	frame_add_patch_list(frame, plist)

	desc = Description('sift', plist, frame.original_sift_descriptors)
	frame_add_description(desc)

	frame_derive_keypoint_geometry(frame)

def seq_detect_sift_unwarp(seq):
	parallel_process(frame_detect_sift_unwarp, seq.frames)

###################################################################################################
# Unwarp after detection
###################################################################################################

def extract_vlfeat_best_matches(kpt_defs, vlfeat_kpt_outputs):
	"""
	@param kpt_defs: argument given to vlfeat_sift's `frames`
	@param vlfeat_kpt_outputs: point definitions detected by vlfeat_sift

	Both in format [y, x, size, orient]

	@return array of indices of matches points

	vlfeat's sift returns points in different order than the `frames` argument specified
	and it with the `force_orientations` option, it may produce multiple alternative orientations for one point.

	For each point in `kpt_defs` this function will find the index of the point in vlfeat's output
	such that they have the same position and the difference of rotations is the smallest amongst the alternatives.
	"""

	# dictionary of point indices, key = tuple(int(x*10), int(y*10))
	vlfeat_pts_by_position = dict()

	def pt_to_key(pt):
		return (int(pt[1]*10), int(pt[0]*10))

	# store detected points:
	for det_pt_idx in range(vlfeat_kpt_outputs.shape[0]):
		key = pt_to_key(vlfeat_kpt_outputs[det_pt_idx, :])
		alt_list = vlfeat_pts_by_position.get(key, None)
		if not alt_list:
			alt_list = []
			vlfeat_pts_by_position[key] = alt_list

		alt_list.append(det_pt_idx)

	# find best matches
	matches = np.zeros(kpt_defs.shape[0], dtype=np.int32)

	for src_pt_idx in range(kpt_defs.shape[0]):
		key = pt_to_key(kpt_defs[src_pt_idx, :])

		alt_list = vlfeat_pts_by_position.get(key, None)
		if not alt_list:
			#print('Missing detections for pt ', src_pt_idx)
			matches[src_pt_idx] = -1
		else:
			if len(alt_list) == 1:
				matches[src_pt_idx] = alt_list[0]
			else:
				orient = kpt_defs[src_pt_idx, 3]
				best_choice = np.argmin([
					angular_distance_abs(orient, vlfeat_kpt_outputs[det_idx, 3])
					for det_idx in alt_list
				])
				matches[src_pt_idx] = alt_list[best_choice]

	return matches

def unwarp_describe_existing_points(plane_obj, b_cut_patches=True, b_describe=True, b_demo_proj=False):
	"""
	Cut patches and calculate SIFT descriptors for existing keypoints on unwarped planes
	"""
	frame = plane_obj.frame
	unw_img = plane_unwarp(plane_obj)

	pt_indices_on_planes = np.where(frame.kpt_plane_ids == plane_obj.plane_id)[0]
	
	if unw_img is None or len(pt_indices_on_planes) == 0:
		# no points on plane
		return np.zeros(0, dtype=np.int32), np.zeros((0, 128), dtype=np.float32), []

	src_locs = frame.kpt_locs[pt_indices_on_planes, :]
	src_orients = frame.kpt_orientations[pt_indices_on_planes, :]
	src_sizes = frame.kpt_sizes[pt_indices_on_planes, :]
	
	pts_dest, pts_sizes_dest, pts_orients_dest = homography_apply_to_keypoints(
		plane_obj.homography, src_locs, src_sizes, src_orients
	)

	if b_demo_proj:
		return pt_indices_on_planes, pts_dest, pts_sizes_dest, pts_orients_dest
	
	if (np.min(pts_dest) < 4 
		or np.max(pts_dest[:, 0]) > plane_obj.out_size[0]-4
		or np.max(pts_dest[:, 1]) > plane_obj.out_size[1]-4
	):
		raise PlaneException('kpts projected outside of plane ' + str(pts_dest.shape[0]))

	if b_describe:
		
		vl_kpt_defs = np.concatenate((
				pts_dest[:, ::-1], # positions as [Y, X] 
				pts_sizes_dest.reshape(-1, 1), # sizes
				pts_orients_dest.reshape(-1, 1), # orientations
			), 
			axis=1,
		)

		try:
			det_pts, det_descs = vlfeat_sift(unw_img, 
				frames = vl_kpt_defs,
				compute_descriptor = True,
				force_orientations = True,
				float_descriptors = True,
			)
			det_indices = extract_vlfeat_best_matches(vl_kpt_defs, det_pts)

			good_subset = (det_indices > -1)
			good_indices = det_indices[good_subset]

			wrong_count = good_subset.shape[0] - np.count_nonzero(good_subset)
			if wrong_count > 0:
				print('Unaligned sift descriptors', wrong_count)

			descriptors = np.zeros((vl_kpt_defs.shape[0], 128), dtype=np.float32)
			descriptors[good_subset] = det_descs[good_indices]

		except Exception as e:
			print('vlfeat_sift failure', e)
			descriptors = np.zeros((pts_dest.shape[0], 128), dtype=np.float32)

	else:
		descriptors = 0

	if b_cut_patches:
		patches = cut_patches(
			plane_obj.unwarped_img,
			pts_dest,
			pts_sizes_dest,
			pts_orients_dest,
		)
	
	return pt_indices_on_planes, descriptors, patches

def frame_describe_unwarp(frame, b_describe=True, b_cut_patches=True):
	"""
	Calculate unwarp-after-detection patches and descriptors
	"""

	sel_pl_ids = select_planes_by_surface(frame)
	sel_pls = [frame.planes[plid] for plid in sel_pl_ids]

	kpt_described = np.zeros(frame.pt_count, dtype=np.bool)
	kpt_descriptors = np.zeros((frame.pt_count, 128), dtype=np.float32)
	kpt_patches = list(frame.patch_lists['flat'].patches)

	planes_cancelled = 0
	for pl in sel_pls:
		try:
			res = unwarp_describe_existing_points(pl, b_describe=b_describe, b_cut_patches=b_cut_patches)
			pt_indices_on_planes, det_descs, patches = res

			if len(pt_indices_on_planes) > 0:
				kpt_described[pt_indices_on_planes] = True
				kpt_descriptors[pt_indices_on_planes, :] = det_descs

				for idx, patch in zip(pt_indices_on_planes, patches):
					kpt_patches[idx] = patch

		except PlaneException as e:
			planes_cancelled += 1

	if planes_cancelled > 0:
		print('Planes cencelled:', planes_cancelled, '/', len(sel_pls))
	
	missing = np.logical_not(kpt_described)
	
	#print('Missing pts', np.count_nonzero(missing), '/', frame.pt_count)
	if b_describe:
		kpt_descriptors[missing] = frame.original_sift_descriptors[missing]
	
	plist = PatchList('unwarp', kpt_patches)
	frame_add_patch_list(frame, plist)

	if b_describe:
		desc = Description('sift', plist, kpt_descriptors)
		frame_add_description(desc)
