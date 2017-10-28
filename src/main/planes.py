from util import *
from geometry import *
import subprocess
 
PLANE_DISTANCE_THRESHOLD_RELATIVE = 0.05
PERCENTILE = [20, 80]

def get_span(vals):
	"""
	Given an array of values, finds the distance between the 20th and 80th percentile
	"""
	vals = vals.flatten()
	vals = vals[np.logical_not(np.isnan(vals))]
	span_ends = np.percentile(vals, PERCENTILE)
	return span_ends[1]-span_ends[0]

def get_threshold(pt_cloud):
	"""
	Calculates the plane-to-point distance threshold,
	equal to PLANE_DISTANCE_THRESHOLD_RELATIVE * (distance between the 20th and 80th percentile of pt coords)
	"""
	spx = get_span(pt_cloud[:, :, 0])
	spy = get_span(pt_cloud[:, :, 1])
	spz = get_span(pt_cloud[:, :, 2])

	spmax = max(spx, spy, spz)
	
	return PLANE_DISTANCE_THRESHOLD_RELATIVE * spmax

class PlaneDetector:
	"""
	Wrapper for the cv2.rgbd.detectPlanes plane-detection algorithm.
	* The function is not available in base OpenCV, it is exposed by OpenCV modifications included in this project
	* OpenCV's plane detection algorithm has a defect: it crashes when the dimensions of the image are not
		divisible by `block_size` (equal to 40).
		We work around that by padding the image to size divisible by 40.
		https://github.com/opencv/opencv_contrib/blob/3.2.0/modules/rgbd/include/opencv2/rgbd.hpp#L366
	"""
	def __init__(self, input_shape, block_size=40):
		if input_shape[0] % block_size != 0 or input_shape[1] % block_size != 0:
			self.resize = True
			
			self.new_size = (
				(1 + input_shape[0] // block_size) * block_size,
				(1 + input_shape[1] // block_size) * block_size,
				3,
			)
			
			self.pointcloud_ext = np.full(self.new_size, np.nan, dtype=np.float32)
			self.normals_ext = np.full(self.new_size, np.nan, dtype=np.float32)
			
		else:
			self.resize = False

	def apply(self, point_cloud, normals=None):
		if self.resize:
			orig_rs, orig_cs, _ = point_cloud.shape

			self.pointcloud_ext[:orig_rs, :orig_cs :] = point_cloud[:, :, :]
			point_cloud = self.pointcloud_ext

			if normals is not None:
				self.normals_ext[:orig_rs, :orig_cs, :] = normals[:, :, :]
				normals = self.normals_ext
			
			thr = get_threshold(point_cloud)

			pl_map, pl_coeffs = cv2.rgbd.detectPlanes(point_cloud, normals, distance_threshold=thr)
			pl_map = pl_map[:orig_rs, :orig_cs]

			return pl_map, pl_coeffs

		else:
			return cv2.rgbd.detectPlanes(point_cloud, normals)

class PlaneRegion:
	"""
	Represents a plane detected in the point cloud
		plane.frame - RGBDFrame in which it was detected
		plane.plane_id - index of plane in frame
		plane.coefficients - (a,b,c,d) such that ax+by+cz+d=0
	"""
	def __init__(self, frame, plane_id, coefs):
		self.frame = frame
		self.plane_id = plane_id
		self.coefficients = coefs

	def get_plane_mask(self):
		return self.frame.plane_map == (self.plane_id+1)

def frame_detect_planes(frame, b_normals=False, plane_detector=None):
	"""
	Detect planes in frame and store them in frame.planes
	"""
	if plane_detector is None:
		plane_detector = PlaneDetector(frame.depth.shape)

	K_mat = frame.intrinsic_mat.astype(np.float32)
	
	frame.point_cloud = cv2.rgbd.depthTo3d(frame.depth, K_mat)
	if b_normals:
		frame.normals = cv2.rgbd.calculateNormals(frame.point_cloud, K_mat)
	else:
		frame.normals = None
	
	pl_map, pl_coeffs = plane_detector.apply(frame.point_cloud, frame.normals)
	pl_map += 1
	pl_coeffs = pl_coeffs[:, 0, :]

	frame.plane_count = pl_coeffs.shape[0]
	frame.plane_map = pl_map
	frame.plane_coefficients = pl_coeffs

	frame.planes = [
		PlaneRegion(frame, plid, frame.plane_coefficients[plid])
		for plid in range(frame.plane_count)
	]

	return pl_map, pl_coeffs

#MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT,(9, 9))

def plane_calculate_convex_hull(plane_obj):
	"""
		Find convex hull of a plane mask and plane surface (pixels in plane),
		Mask preprocessed with opening operation
	"""
	frame = plane_obj.frame
	plid = plane_obj.plane_id
	plane_mask = (frame.plane_map == (plid+1))
	plane_mask_u8 = plane_mask.astype(np.uint8)
	
	# open to remove small noise points
	#plane_mask_u8 = cv2.morphologyEx(plane_mask_u8, cv2.MORPH_OPEN, MORPH_KERNEL)

	# area without noise points	
	area = np.count_nonzero(plane_mask_u8) / (frame.photo.shape[0] * frame.photo.shape[1])
	plane_obj.area = area
	plane_obj.valid = area > 0

	if plane_obj.valid:
		# find contours of the plane-mask, calculate convex hull of all contour points
		_, contour_list, hierarchy = cv2.findContours(plane_mask_u8, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
		all_contour_pts = np.concatenate(contour_list, axis=0)
		hull = cv2.convexHull(all_contour_pts)[:, 0, :]
	else:
		#print('Empty plane id =', plid)
		hull = np.zeros((0, 2), dtype=np.int32)
		
	plane_obj.hull = hull

	return hull, area

def frame_all_convex_hulls(frame):

	for pl in frame.planes:
		plane_calculate_convex_hull(pl)

	frame.plane_areas = np.array([pl.area for pl in frame.planes], dtype=np.float64)

	# planes ids ordered by the surface of their regions
	frame.planes_by_surface = np.argsort(frame.plane_areas)[::-1]

def seq_detect_planes(seq, b_normals=False):
	pl_det = PlaneDetector(seq.frames[0].depth.shape)

	for fr in seq.frames:
		frame_detect_planes(fr, b_normals=b_normals, plane_detector=pl_det)
		frame_all_convex_hulls(fr)

def frame_draw_plane_contour(plane_obj):
	frame = plane_obj.frame
	plane_mask = plane_obj.get_plane_mask()

	canvas = frame.photo.copy()
	#canvas[plane_mask, 0] = np.minimum(255, canvas[plane_mask, 0].astype(np.int16) + 50)
	return cv2.drawContours(canvas, [plane_obj.hull], 0, (0, 255, 0), thickness=3)

UNWARP_MARGIN = np.array([32, 32], dtype=np.int32)

def minmax(ar):
	return np.array((np.min(ar), np.max(ar)))

def intarray(v):
	return np.array(v, dtype=np.int32)

def calculate_image_unwarping_homography(normal, K, hull_points):
	"""
	@param hull_points convex hull of regions which should be visible on output

	@return (H, out_size):
		H = homography matrix
		out_size = needed output for unwarped image
	"""
	
	r3 = -normal
	R = np.eye(3)
	R[:, 0] = normal_to_both(vec(0, 1, 0), r3)
	R[:, 1] = normal_to_both(r3, R[:, 0])
	R[:, 2] = r3.flatten()
	
	t = vec(0, 0, 1)
	
	Rt = R.copy()
	Rt[:, 2:3] = t 
	
	H = K @ Rt
	H = np.linalg.inv(H)

	# now adjust the homography and output size to contain the output region
	# convex hull of the warped region
	H_initial = H
	hull_wr = homography_apply_rowvec(H_initial, hull_points)

	x_range = minmax(hull_wr[:, 0])
	y_range = minmax(hull_wr[:, 1])
	out_center = np.array((np.average(x_range), np.average(y_range)), dtype=np.float32)
	
	# requires size of image to contain the whole unwarped image
	out_size = np.array((x_range[1] - x_range[0], y_range[1] - y_range[0]), dtype=np.float32)
	
	# adjust homography such that output image is centered
	out_offset = out_size * 0.5 - out_center

	H_full = affine_translation(out_offset) @ H_initial
	
	# How to unwarp the image:
	# unwarped = cv2.warpPerspective(src_img, H_full, tuple(out_size))

	return H_full, out_size

def plane_unwarp(plane_obj, b_demo=False):
	"""
	Produce unwarped view of the plane and store it in:
		plane.unwarped_img
		plane.homography

	@param b_demo: draw the plane contour
	"""

	if not plane_obj.valid:
		return None

	if hasattr(plane_obj, 'unwarped_img') and not b_demo:
		# was already unwarped, use cached result
		return plane_obj.unwarped_img

	frame = plane_obj.frame

	normal = plane_obj.coefficients[:3].copy().reshape((3, 1))
	
	H, out_size = calculate_image_unwarping_homography(normal, frame.intrinsic_mat, plane_obj.hull) 

	desired_size = np.max(frame.photo.shape)
	adjust_scale = desired_size / np.max(out_size)
	new_size = out_size * adjust_scale
	
	H = affine_scale(adjust_scale, adjust_scale) @ H
	out_size = np.rint(new_size).astype(np.int32)
	
	# extend area with a margin
	H = affine_translation(UNWARP_MARGIN // 2) @ H
	out_size += UNWARP_MARGIN

	plane_obj.normal = normal
	plane_obj.homography = H
	plane_obj.out_size = out_size

	
	plane_obj.unwarped_img = cv2.warpPerspective(frame.photo_gray_float, H, tuple(out_size),
		None, cv2.INTER_LINEAR, cv2.BORDER_REFLECT101
	)

	if not b_demo:
		return plane_obj.unwarped_img
	else:
		unwarped_img_color = cv2.warpPerspective(frame.photo, H, tuple(out_size),
			None, cv2.INTER_LINEAR, cv2.BORDER_REFLECT101
		)
		hull_unw = homography_apply_rowvec(H, plane_obj.hull)
		unwarped_img_color = cv2.drawContours(unwarped_img_color, [hull_unw.astype(np.int32)], 0, (0, 255, 0), thickness=3)

		return unwarped_img_color

def frame_draw_unwarp_demo(frame, plane_by_size=0):
	"""
	Draw the image before and after unwarping, side by side
	"""
	plid = frame.planes_by_surface[plane_by_size]
	
	plane_obj = frame.planes[plid]
	contour_img = frame_draw_plane_contour(plane_obj)
	unw_img = plane_unwarp(plane_obj, b_demo=True)

	return [contour_img, unw_img]

