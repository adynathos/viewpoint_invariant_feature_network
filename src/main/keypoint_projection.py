from util import * 
from geometry import *
from scipy.spatial import cKDTree as KDTree

###################################################################################################
# Keypoint representation
###################################################################################################
def frame_filter_detections(frame, indices):
	frame.kpt_locs = frame.kpt_locs[indices, :]
	frame.kpt_sizes = frame.kpt_sizes[indices, :]
	frame.kpt_orientations = frame.kpt_orientations[indices, :]
	frame.original_sift_descriptors = frame.original_sift_descriptors[indices, :]
	frame.pt_count = frame.kpt_locs.shape[0]

def frame_derive_keypoint_geometry(frame):
	"""
	Derives depths, cam-space positions, plane_ids of keypoints.
	Removes keypoints which miss this info.
	"""

	# Filter out points not lying on planes
	locs_int = np.rint(frame.kpt_locs).astype(np.int32)
	valid = frame.plane_map[locs_int[:, 1], locs_int[:, 0]] > 0
	valid &= np.logical_not(np.isnan(frame.depth[locs_int[:, 1], locs_int[:, 0]]))
	frame_filter_detections(frame, valid)

	# Derive info
	locs_int = np.rint(frame.kpt_locs).astype(np.int32)
	locs_indexer = (locs_int[:, 1], locs_int[:, 0])

	frame.kpt_depths = frame.depth[locs_indexer]
	frame.kpt_plane_ids = frame.plane_map[locs_indexer] - 1

def frame_get_kpt_normals(frame, pt_ids):
	return frame.plane_coefficients[frame.kpt_plane_ids[pt_ids], :3]

###################################################################################################
# Applying transformations to keypoints
###################################################################################################

def kptproj_prepare_vectors(pt_locs, pt_sizes, pt_orientations):
	"""
	Produce the helper points for keypoint projection
	"""

	# project 3 pts:
	# center, center + (cos, sin), center + (sin, -cos)
	# measure angle and size

	pt_locs = pt_locs.reshape((-1, 2))
	pt_sizes = pt_sizes.reshape((-1, 1))
	pt_orientations = pt_orientations.reshape((-1, 1))

	pt_count = pt_locs.shape[0]
	pt_cos = np.cos(pt_orientations) * pt_sizes
	pt_sin = np.sin(pt_orientations) * pt_sizes

	cps_src = np.repeat(pt_locs, 3, axis=0)
	cps_src[1::3, :] += np.concatenate((pt_cos, pt_sin), axis=1)
	cps_src[2::3, :] += np.concatenate((pt_sin, -pt_cos), axis=1)

	return cps_src

def kptproj_interpret_projected_vectors(cps_dest):
	"""
		Given helper points, created by `kptproj_prepare_vectors` and then transformed,
		returns the keypoint locations, sizes, orientations of transformed keypoints.

		The `cps_dest` array is destroyed in the process.
	"""

	pt_count = cps_dest.shape[0] // 3
	if pt_count == 0:
		return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

	cps_dest = cps_dest.reshape((pt_count, 3, 2))

	# make vectors relative to centers
	pt_centers_dest = cps_dest[:, 0, :]
	cps_dest[:, 1, :] -= pt_centers_dest
	cps_dest[:, 2, :] -= pt_centers_dest

	# sizes = average btw projected vectors
	cps_distances = np.linalg.norm(cps_dest[:, (1, 2), :], axis=2)
	pt_sizes_dest = np.average(cps_distances, axis=1)

	# orientations = angle of projected orintation vector
	pt_orientations_dest = np.arctan2(cps_dest[:, 1, 1], cps_dest[:, 1, 0])

	return (pt_centers_dest, pt_sizes_dest, pt_orientations_dest)


###################################################################################################
# Homography transformation
###################################################################################################
def homography_apply_to_keypoints(H, pt_locs, pt_sizes, pt_orientations):
	
	cps_src = kptproj_prepare_vectors(pt_locs, pt_sizes, pt_orientations)

	cps_dest = homography_apply_rowvec(H, cps_src)

	return kptproj_interpret_projected_vectors(cps_dest)

###################################################################################################
# Spatial transformation
###################################################################################################

def image_points_to_rays(pt_positions, intrinsic):
	intrinsic_inv = np.linalg.inv(intrinsic)

	pts3_colvec = extend_with_neutral_row(pt_positions.T)
	rays = (intrinsic_inv @ pts3_colvec).T

	return rays

def build_point_cloud_for_projection(src_frame):
	"""
	Builds a point cloud of feature points in src_frame
	which will be used for ground-truth projections of these keypoints to other frames
	"""
	helper_points = kptproj_prepare_vectors(src_frame.kpt_locs, src_frame.kpt_sizes, src_frame.kpt_orientations)

	locs_int = np.rint(helper_points).astype(np.int32)

	locs_int[:, 0] = np.clip(locs_int[:, 0], 0, src_frame.photo.shape[1]-1)
	locs_int[:, 1] = np.clip(locs_int[:, 1], 0, src_frame.photo.shape[0]-1)

	# extract depths in those points
	depths = src_frame.depth[locs_int[:, 1], locs_int[:, 0]]

	# but some depths may be missing, fill them with depth from original points
	# which should have depth, because we disard points without depth
	invalid_depth_idx = np.where(np.isnan(depths))[0]
	depths[invalid_depth_idx] = depths[invalid_depth_idx - invalid_depth_idx % 3]

	if np.any(np.isnan(depths)):
		print('Depths still wrong in build_point_cloud_for_projection', np.where(np.isnan(depths))[0])

	helper_rays = image_points_to_rays(helper_points, src_frame.intrinsic_mat)

	helper_cloud = helper_rays * depths.reshape(-1, 1)

	src_frame.kpt_rays = helper_rays[::3, :]
	src_frame.kpt_proj_cloud = helper_cloud

	return helper_cloud

MATCH_DISTANCE = 5
MATCH_SIZE_DIFF_RELATIVE = 2**0.25
MATCH_ANGLE_DIFF = 0.25*np.pi 

def keypoints_match_geometry(src_frame, dest_frame):
	"""
	Finds keypoint matches based on scene geometry

	@return pairs, angles
	`pairs[n] = (src_pt_n, dest_pt_n)`
	`angles[n]` = angle btw rays of src_pt_n and dest_pt_n
	"""

	if hasattr(src_frame, 'kpt_proj_cloud'):
		helper_cloud = src_frame.kpt_proj_cloud
	else:
		helper_cloud = build_point_cloud_for_projection(src_frame)

	# spatial and perspective projection src_frame -> dest_frame
	spatial_mat = dest_frame.world_to_camera @ src_frame.camera_to_world 
	full_projection_mat = dest_frame.intrinsic_mat @ spatial_mat[:3, :]
	
	# project and retrieve keypoints
	helper_projected = projection_apply_rowvec(full_projection_mat, helper_cloud)
	proj_pts, proj_sizes, proj_orientations = kptproj_interpret_projected_vectors(helper_projected)

	# find pairs of neighbours in radius of MATCH_DISTANCE
	tree_proj = KDTree(proj_pts)
	tree_dest = KDTree(dest_frame.kpt_locs)
	match_suggestions = tree_dest.query_ball_tree(tree_proj, r=MATCH_DISTANCE)
	matches = []

	# kpt_matched_id[n] = id of point in src_frame which matches n
	#frame.kpt_matched_id = np.zeros(frame.pt_count, dtype=np.int32)
	#frame.kpt_matched_id[:] = -1 # no match

	#print(src_frame.pt_count, dest_frame.pt_count, len(match_suggestions))

	for dest_pt_idx, suggestions in enumerate(match_suggestions):

		dest_pt_size = dest_frame.kpt_sizes[dest_pt_idx]

		for src_pt_idx in suggestions:
			proj_size = proj_sizes[src_pt_idx]

			if ((max(dest_pt_size/proj_size, proj_size/dest_pt_size) < MATCH_SIZE_DIFF_RELATIVE)
				and
				(angular_distance_abs(dest_frame.kpt_orientations[dest_pt_idx], proj_orientations[src_pt_idx]) < MATCH_ANGLE_DIFF)
			):			
				matches.append((src_pt_idx, dest_pt_idx))
				#frame.kpt_matched_id[pt_idx] = src_pt_idx
				break

	if len(matches) == 0:
		return AttrDict(
			pairs = np.zeros((0, 2), dtype=np.int32), 
			angles = np.zeros(0, dtype=np.float32),
		)
	else:
		# store matched pairs
		match_pairs = np.array(matches, dtype=np.int32)
		# sort by src_point_id
		match_pairs = match_pairs[np.argsort(match_pairs[:, 0]), :]
	
		view_angle_changes = derive_keypoint_view_angle_change(src_frame, dest_frame, match_pairs)

		return AttrDict(
			pairs = match_pairs, 
			angles = view_angle_changes,
		)

def seq_find_groundtruth_matches_to_first_frame(seq):

	src_frame = seq.frames[0]
	rest_frames = seq.frames[1:]

	def step(frame):
		gt_info  = keypoints_match_geometry(src_frame, frame)
		frame.gt_to_first = gt_info

	parallel_process(step, rest_frames, threading=False, disp_progress=False)
 

###################################################################################################
# View angle change
###################################################################################################
def derive_keypoint_view_angle_change(src_frame, dest_frame, pairs):
	"""
	Returns angles such that:
	angles[match_id] = 
		angle_rad btw keypoint rays, for pt pair in pairs[match_id]

	"""

	# extract rays of matching points
	matched_pt_ids_src = pairs[:, 0]
	matched_pt_ids_dest = pairs[:, 1]
	
	kpt_rays_src = src_frame.kpt_rays[matched_pt_ids_src]
	kpt_rays_dest = image_points_to_rays(dest_frame.kpt_locs[matched_pt_ids_dest, :], dest_frame.intrinsic_mat)

	# rotation-only transform btw frames, to rotate the rays to dest_frame's system
	R_mat = dest_frame.world_to_camera[:3, :3] @ src_frame.camera_to_world[:3, :3]
	kpt_rays_src_rotated = (R_mat @ kpt_rays_src.T).T

	# normalize and dot product
	kpt_rays_src_rotated /= np.linalg.norm(kpt_rays_src_rotated, axis=1).reshape(-1, 1)
	kpt_rays_dest /= np.linalg.norm(kpt_rays_dest, axis=1).reshape(-1, 1)
	dot_products = np.sum(kpt_rays_src_rotated * kpt_rays_dest, axis=1)

	# retrieve angles from dot products
	angles = np.abs(np.arccos( dot_products ))
	return angles
