from util import *
from geometry import *
from patch_types import *
import random

# magnification=3
# The scale of the keypoint is multiplied by this factor to obtain the width (in pixels) of the spatial bins. 
# For instance, if there are there are 4 spatial bins along each spatial direction, 
# the ``side`` of the descriptor is approximately ``4 * magnification``.
PATCH_RADIUS_FACTOR = 3 * 2 # multiply this by keypoint size ("scale") to get actual size

class PatchList:
	def __init__(self, name, patches):
		self.name = name
		self.patches = patches
		self.descriptions = dict()

		if patches:
			self.patch_array = np.stack(self.patches)

	def __repr__(self):
		return 'PatchList<{n}>'.format(n=self.name)

	def set_depth(self, depth_patch_list):
		self.depth_array = np.stack(depth_patch_list)

	def set_normals(self, normal_patch_list):
		self.normals_array = np.stack(normal_patch_list)

def frame_add_patch_list(frame, plist):

	if not hasattr(frame, 'patch_lists'):
		frame.patch_lists = AttrDict()

	plist.frame = frame
	frame.patch_lists[plist.name] = plist

class Description:
	def __init__(self, name, plist, descriptors):
		self.name = name
		self.patch_list = plist
		self.descriptors = descriptors

	def get_frame(self):
		return self.patch_list.frame

	def __repr__(self):
		return "Description<{dn}>(PatchList<{pn}>)".format(dn=self.name, pn=self.patch_list.name)

def desc_id_name(did):
	return "{dn}({pn})".format(dn=did[1], pn=did[0])

def frame_add_description(desc):
	desc.patch_list.descriptions[desc.name] = desc
	desc.patch_list.frame.descriptions[(desc.patch_list.name, desc.name)] = desc


def frame_build_description_from_descriptors(frame, pl_name, desc_name, descriptors):

	orig_plist = PatchList(pl_name, [])
	frame_add_patch_list(frame, orig_plist)

	orig_desc = Description(desc_name, orig_plist, descriptors)
	frame_add_description(orig_desc)


def frame_build_original_sift_description(frame):
	frame_build_description_from_descriptors(frame, 'original', 'sift', frame.original_sift_descriptors)


def patch_file_name(fr_id, p_id):
	return '{p:04d}_{f:04d}'.format(f=fr_id, p=p_id)

def get_patch_lists(frames, plist_categories):
	return [fr.patch_lists[pn] for fr in frames for pn in plist_categories]

def cut_patches(img, pt_locs, pt_sizes, pt_orients, out_size=32, size_multiplier=PATCH_RADIUS_FACTOR, depth_and_normals=None):
	"""
	@param size_multiplier: multiply by pt_sizes to get patch radius
	"""

	pt_count = pt_locs.shape[0]
	pt_sizes = pt_sizes.reshape(-1)
	pt_orients = pt_orients.reshape(-1)

	out_shape = np.array([out_size, out_size], dtype=np.int32)
	out_shape_tpl = tuple(out_shape)

	patch_diams = pt_sizes * (size_multiplier * 2)
	scales = out_size  / patch_diams

	scale_mat = np.eye(3, dtype=np.float32)

	patches = []

	b_depth = depth_and_normals is not None

	if b_depth:
		img_depths, img_normals = depth_and_normals
		patches_depths = []
		patches_normals = []
		neutral_normal = np.array([0, 0, -1], dtype=np.float32)
		patch_normals_shape = (out_size, out_size, 3)
		patch_stacked_normals_shape = (out_size*out_size, 3)

	for pt_idx in range(pt_count):

		Al, Ar = patch_cutting_affine_matrix(pt_locs[pt_idx, :], out_shape)
		R = rot_around_z(-pt_orients[pt_idx])

		scale_mat[0, 0] = scale_mat[1, 1] = scales[pt_idx]

		A = Al @ R @ scale_mat @ Ar

		patches.append(
			cv2.warpAffine(img, A[:2, :], out_shape_tpl, borderMode=cv2.BORDER_REFLECT101)
		)

		if b_depth:
			pd = cv2.warpAffine(img_depths, A[:2, :], out_shape_tpl, borderMode=cv2.BORDER_REFLECT101)

			# missing depth values
			invalid_mask = np.isnan(pd)

			# replace nans with mean
			if np.any(invalid_mask):
				avg_d = np.mean(pd[np.logical_not(invalid_mask)])
				pd[invalid_mask] = avg_d

			# subtract center from depth
			pd -= pd[pd.shape[0] // 2, pd.shape[1]//2]
			
			# extract normals
			pn = cv2.warpAffine(img_normals, A[:2, :], out_shape_tpl, borderMode=cv2.BORDER_REFLECT101)
			pn[invalid_mask] = neutral_normal

			# rotate normals
			pn_stack = pn.reshape(patch_stacked_normals_shape)
			pn_stack = (R.astype(np.float32) @ pn_stack.T).T
			pn = pn_stack.reshape(patch_normals_shape)

			patches_depths.append(pd)
			patches_normals.append(pn)

	if b_depth:
		return (patches, patches_depths, patches_normals)		
	else:
		return patches

def compose_patches(intensities, depth=None, normals=None, intensity_out_type = np.float32):
	"""
	Prepare patches in a format input for torch
	"""

	# reshape to (n, channels=1, H, W)
	intensities = intensities.reshape((intensities.shape[0], 1, intensities.shape[1], intensities.shape[2]))
	
	# convert and normalize
	intensities = intensities.astype(intensity_out_type)
	intensities *= (1 / 255)
	intensities -= np.mean(intensities, dtype=np.float64)
	intensities /= np.std(intensities, dtype=np.float64)

	if depth is not None:
		depth = depth.reshape((depth.shape[0], 1, depth.shape[1], depth.shape[2]))
		
		depth[np.isnan(depth)] = 0

		return np.concatenate((intensities, depth), axis=1)

	elif normals is not None:
		# (n, H, W, 3) -> (n, 3, H, W) for torch
		normals = np.moveaxis(normals, 3, 1)

		return np.concatenate((intensities, normals), axis=1)
	else:
		return intensities

###################################################################################################
# Display
###################################################################################################
def list_planes(frame, b_surface=False, b_sort_patches=False):
	plane_count = frame.plane_coefficients.shape[0]
	coeffs = frame.plane_coefficients
	counts = np.zeros(plane_count, np.int32)
	avg_patch_area = np.zeros(plane_count, np.float32)

	surface = np.zeros(plane_count, np.int32)
	if b_surface:
		for plane_id in range(plane_count):
			surface[plane_id] = np.count_nonzero(frame.plane_map == (plane_id+1))

		surface = surface / (frame.plane_map.shape[0] * frame.plane_map.shape[1])

	for pnid in frame.kpt_plane_ids:
		counts[pnid] += 1 # replace with np.bincount
		avg_patch_area[pnid] += frame.kpt_sizes[pnid]

	avg_patch_area /= counts
	avg_patch_area *= 12
	avg_patch_area **= 2

	table = np.hstack((
		np.arange(counts.shape[0]).reshape((-1, 1)), # id
		counts.reshape((-1, 1) ),  # patch counts
		surface.reshape((-1, 1) ),
		coeffs[:, 2:3],
	))

	if b_sort_patches:
		table_idx = np.argsort(table[:, 1])[::-1]
	else:
		table_idx = np.arange(plane_count)

	np.set_printoptions(suppress=True)
	print('Plane ------------- patch count ------ surface ---- normal_z ---------------')
	print(table[table_idx, :])

def draw_patch_comparison(plists, patch_ids, col_names=None, save=None):
	fig = show_multidim([
			[pl.patches[pid] for pl in plists]
			for pid in patch_ids
		],
		figscale=(2, 2),
		col_titles = col_names or [pl.name for pl in plists],
		save = save,
	)
	return fig

def show_patches_from_plane(frame, plane_id, plist_names=None, plist_to_draw='unwarp', count=5, draw_pts_too=False):
	"""
	@param plist_names: patch lists, None to take all
	"""

	fr_idx = np.where(frame.kpt_plane_ids == plane_id )[0]
	print('Frame', frame.name, '- planes:')
	print(' Plane', plane_id, ' - ', frame.plane_coefficients[plane_id, :])
		
	if len(fr_idx) == 0:
		print('  No patches in plane')
		return
	
	chosen_idx = random.sample(list(fr_idx), min(fr_idx.shape[0], count))
	print(chosen_idx)

	if plist_names:
		plists = [frame.patch_lists[pn] for pn in plist_names]
	else:
		plists = list(frame.patch_lists.values())
		plist_names = [pl.name for pl in plists]
	
	if plist_to_draw:
		photo_with_patches = frame_draw_patches_on_photo(frame, plist_to_draw, chosen_idx)
		show([photo_with_patches, frame.plane_map == (plane_id+1)])
			
	draw_patch_comparison([frame.patch_lists[pln] for pln in plist_names], chosen_idx)

def get_points_from_each_plane(frame, per_plane=1):
	patch_id_lists = []

	plane_count = frame.plane_coefficients.shape[0]

	for pl_id in range(plane_count):
		pts_on_plane = np.where(frame.kpt_plane_ids == pl_id)[0]
		if len(pts_on_plane):
			idx = random.sample(range(len(pts_on_plane)), per_plane)
			patch_id_lists.append(pts_on_plane[idx])

	return np.hstack(patch_id_lists)


def show_plane_representatives(frame, plist_names=None, plist_to_draw='unwarp', per_plane=1):

	patch_ids = get_points_from_each_plane(frame, per_plane=per_plane)

	if not plist_names:
		plist_names = list(frame.patch_lists.keys())
		plist_names.sort()

	plists = [frame.patch_lists[pn] for pn in plist_names]

	if plist_to_draw:
		photo_with_patches = frame_draw_patches_on_photo(frame, plist_to_draw, patch_ids)
		show([photo_with_patches, frame.plane_map])

	draw_patch_comparison(plists, patch_ids)

