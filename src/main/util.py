
import numpy as np
import scipy as sp
import colorsys
from os.path import join as pp
from attrdict import AttrDict
from functools import partial
import os, glob, datetime, gc
import cv2

def mat_info(a, name=None):
	if name:
		print(name, a.dtype, a.shape, np.nanmin(a), np.nanmax(a))
	else:
		print(a.dtype, a.shape, np.nanmin(a), np.nanmax(a))

def byted_to_real(data):
	return data.astype(np.float32) / 255

def real_to_byted(data):
	c = data.copy()
	c -= c.min()
	c *= 255/c.max()
	return c.astype(np.uint8)

def nth_color(color_idx, color_count, scale=255, tp=int):
	"""
		Find `color_count` unique colors along the HSV circle
	"""
	# end at 300/360 so its red --> purple, 
	return tuple(tp(scale*v) for v in colorsys.hsv_to_rgb( 0.83 * color_idx / color_count, 1, 1))

def ensure_file_removed(filepath):
	if os.path.exists(filepath):
		os.remove(filepath)

def save_plot(fig, path):
	os.makedirs(os.path.dirname(path), exist_ok=True)
	fig.savefig(path)
	fn, ext = os.path.splitext(path)
	if ext == '.ps':
		cmd = ['/usr/bin/ps2pdf', path, fn + '.pdf']
		#print(cmd)
		subprocess.run(cmd) 

def load_image_any_ext(base_path):
	potential = glob.glob(base_path + '.*')
	if potential:
		return imread(potential[0])
	else:
		print('No image for', base_path)

###################################################################################################
# Thread-parallel loops
###################################################################################################
from multiprocessing.dummy import Pool as thread_Pool

if not ('POOL' in globals()):
	POOL = thread_Pool(8)

def parallel_process(func, tasks, threading=True, disp_progress=True, step_size=1):

	if disp_progress:
		from util_notebook import ProgressBar 
		pbar = ProgressBar(len(tasks))
	else:
		pbar = 0

	if threading:
		for progress in POOL.imap(func, tasks, chunksize=step_size):
			pbar += 1 
	else:
		for t in tasks:
			func(t)
			pbar += 1

###################################################################################################
# Drawing keypoints and matches
###################################################################################################

def to_cv_pt(v):
	""" Row vector to tuple of ints, which is used as input in some OpenCV functions """
	vi = np.rint(v).astype(np.int)
	return (vi.flat[0], vi.flat[1])

def kpts_to_cv(pt_positions, pt_sizes, pt_orientations):
	""" Arrays of keypoint positions, sizes and orientations to list of cv2.KeyPoint """
	sz = pt_sizes * 2
	angs = pt_orientations * (180/np.pi)

	return [
		cv2.KeyPoint(pt_positions[idx, 0], pt_positions[idx, 1], sz[idx], angs[idx])
		for idx in range(pt_positions.shape[0])
	]

def matches_to_cv(pair_array):
	""" Nx2 int array to list of cv2.DMatch """
	return [
		cv2.DMatch(row[0], row[1], 0)
		for row in pair_array
	]

def draw_keypoints(photo, pt_positions, pt_sizes, pt_orientations, color=None):
	""" Draw keypoints, with sizes and orientations , on an image """

	pt_objs = kpts_to_cv(pt_positions, pt_sizes, pt_orientations)
	return cv2.drawKeypoints(photo, pt_objs, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS, color=color)

def frame_draw_kpt_subset(frame, indices, img = None, color=(0, 255, 0)):
	return draw_keypoints(
		img if img is not None else frame.photo, 
		frame.kpt_locs[indices, :], 
		frame.kpt_sizes[indices, :], 
		frame.kpt_orientations[indices, :], 
		color=color
	)

def draw_keypoint_matches(frame_src, frame_dest, pairs):

	pts_cv_src = kpts_to_cv(frame_src.kpt_locs, frame_src.kpt_sizes, frame_src.kpt_orientations)
	pts_cv_dest = kpts_to_cv(frame_dest.kpt_locs, frame_dest.kpt_sizes, frame_dest.kpt_orientations)
	matches_cv = matches_to_cv(pairs)

	return cv2.drawMatches(
		frame_src.photo, pts_cv_src, frame_dest.photo, pts_cv_dest, matches_cv, None,
		#matchColor = (0, 255, 0),
		#singlePointColor = (25, 50, 255),
		flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
	)

