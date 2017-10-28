bl_info = {
	"name": "Synthetic Dataset Generator",
	"description": "Generate depth and normal views of scenes for computer vision research",
	"author": "Krzysztof Lis - adynathos@gmail.com",
	"version": (1, 0),
	"blender": (2, 78, 0),
	"location": "Render",
	"wiki_url": "...",
	"tracker_url": "...",
	"category": "Import-Export",
}

import bpy
import os
from os.path import join as pp
from enum import Enum
import mathutils as bmath
import math
import random
#import numpy as np

OP_ID = "rgbd.render"
ST_FINISHED = {'FINISHED'}
ST_INFO = {'INFO'}

#####################################################################
# Registration
#####################################################################

def add_to_menu(self, context) :
	self.layout.operator(OP_ID, icon = "PLUGIN")

def register():
	bpy.utils.register_module(__name__)

	bpy.types.INFO_MT_render.append(add_to_menu)

	print('RGBD - register')

	#bpy.types.WindowManager.matalogue_settings = bpy.props.PointerProperty(type=MatalogueSettings)

def unregister():
	#del bpy.types.WindowManager.matalogue_settings

	print('RGBD - UNregister')

	bpy.types.INFO_MT_render.remove(add_to_menu)

	bpy.utils.unregister_module(__name__)

if __name__ == "__main__":
	register()

#####################################################################
# PARAMETERS
#####################################################################

class RgbdAddonPreferences(bpy.types.AddonPreferences):
	bl_idname = __name__

	dir_out = bpy.props.StringProperty(
		name = "Output directory",
		subtype='FILE_PATH',
		description = "Output directory",
		default = 'out'
	)

	dir_tex = bpy.props.StringProperty(
		name = "Texture directory",
		subtype='FILE_PATH',
		description = "Texture directory",
		default = 'textures'
	)

	fov = bpy.props.FloatProperty(name='FOV [deg]', min=10, max=170, default=70, precision=1, description='Field of view of camera in degrees')
	pitch_max = bpy.props.FloatProperty(name='Pitch angle max [deg]', min=0, max=90, default=60, precision=1, description='Pitch angle range')
	frames_per_scene = bpy.props.IntProperty(name='Frames per scene', min=1, default=5)
	
	def draw(self, context):
		col = self.layout.column(align = True)
		col.prop(self, 'dir_out')
		col.prop(self, 'dir_tex')
		col.prop(self, 'fov')
		col.prop(self, 'pitch_max')
		col.prop(self, 'frames_per_scene')

def get_prefs(ctx):
	return ctx.user_preferences.addons[__name__].preferences

class RGBD_Panel(bpy.types.Panel):
	bl_label = "RGBD Rendering"
	bl_space_type = "PROPERTIES"
	bl_region_type = "WINDOW"
	bl_context = "render"

	def draw(self, context):
		pref = get_prefs(context)

		col = self.layout.column()

		col.prop(pref, 'dir_out')
		col.prop(pref, 'dir_tex')
		col.prop(pref, 'fov')
		col.prop(pref, 'pitch_max')
		col.prop(pref, 'frames_per_scene')

		col.separator()
		col.operator(OP_ID, text="Render RGBD", icon='PLUGIN')


#####################################################################
# OPERATION
#####################################################################

MAT_ROT_X_180 = bmath.Matrix.Rotation(math.pi, 4, 'X')

def bpy_mat_to_csv(mat):
	return '\n'.join([
		'	'.join(['{:.6f}'.format(val) for val in row])
		for row in mat
	])

def write_pose(camera, out_path):
	# transform of the camera object
	mat_camobj_to_world = camera.matrix_world
	
	# but Blender's camera looks into negative Z
	# to fix that, rotate by 180 around X
	mat_cam_to_world = mat_camobj_to_world * MAT_ROT_X_180

	# cam to world
	#mat_world_to_cam = mat_cam_to_world.inverted_safe()
	mat_csv = bpy_mat_to_csv(mat_cam_to_world)

	with open(out_path, 'w') as out_file:
		out_file.write(mat_csv)

def spherical_to_cartesian(r, pitch, yaw, roll=0.):
	mat_r = bmath.Matrix.Translation((0, 0, r))
	mat_roll = bmath.Matrix.Rotation(roll, 4, 'Z')
	mat_pitch = bmath.Matrix.Rotation(pitch, 4, 'X')
	mat_yaw = bmath.Matrix.Rotation(yaw, 4, 'Z')
	return mat_yaw * mat_pitch * mat_roll * mat_r

def move_camera_to_spherical(r, pitch, yaw, roll):
	mat_tr = spherical_to_cartesian(r, pitch, yaw, roll)
	print('Move camera to ', r, math.degrees(pitch), math.degrees(yaw), math.degrees(roll))

	scene = bpy.context.scene
	scene.camera.matrix_world = mat_tr

def get_node_graph():
	return bpy.context.scene.node_tree.nodes

def get_file_out_node():
	nodes = bpy.context.scene.node_tree.nodes
	return nodes['File Output']

def set_out_file_path_base(path):
	file_out_node = get_file_out_node()
	file_out_node.base_path = path

def set_out_file_index(idx):
	file_out_node = get_file_out_node()

	for slot in file_out_node.file_slots:
		# out names are in format
		#	####_image
		#	####_depth
		#	####_normals
		
		prev_idx, slot_name = slot.path.split('_')
		slot.path = '{idx:04d}_{name}'.format(idx=idx, name=slot_name)


wm = bpy.context.window_manager

# progress from [0 - 1000]
tot = 1000
wm.progress_begin(0, tot)
for i in range(tot):
    wm.progress_update(i)
wm.progress_end()

class RENDER_OT_RenderRgbd(bpy.types.Operator):
	bl_idname = OP_ID
	bl_label = "Render RGBD"
	bl_options = {'REGISTER'}

	def execute(self, context):

		scene = context.scene
		pref = get_prefs(context)

		# prepare rendering:
		# - set FOV
		context.scene.camera.data.angle_x = math.radians(pref.fov)
		

		# discover textures
		texture_files = os.listdir(pref.dir_tex)		
		texture_files.sort()
		texture_files = [pp(pref.dir_tex, tf) for tf in texture_files]

		if not texture_files:
			self.report({'ERROR'}, 'No textures in '+pref.dir_tex)
			return

		tex_img = None
		for tex_idx, tex_file in enumerate(texture_files):
			tex_file_path = os.path.abspath(tex_file)
			print('Scene', tex_idx, '-', tex_file_path)

			dir_scene = pp(pref.dir_out, 'scene_{s:03d}'.format(s=tex_idx))
			# - set out dir
			os.makedirs(dir_scene, exist_ok=True)
			set_out_file_path_base(dir_scene)

			# load and apply texture
			prev_tex_img = tex_img
			
			tex_img = bpy.data.images.load(tex_file_path, check_existing=True)
			bpy.data.materials['image_material'].texture_slots[0].texture.image = tex_img

			# remove previous image to free mem
			if prev_tex_img:
				prev_tex_img.user_clear()
				if not prev_tex_img.users:
					bpy.data.images.remove(prev_tex_img)

			# progress bar
			for fr_id in range(pref.frames_per_scene):

				# move camera
				if fr_id == 0:
					pitch = 0.
					yaw = 0.
					roll = 0.

				else:
					pitch = math.radians(random.uniform(0, pref.pitch_max))
					yaw = random.uniform(0, 2*math.pi)
					roll = random.uniform(0, 2*math.pi)

				move_camera_to_spherical(5., pitch, yaw, roll)

				# write camera pose to file
				write_pose(scene.camera, pp(dir_scene, '{n:04d}_pose.csv'.format(n=fr_id)))

				# set out file names
				set_out_file_index(fr_id)

				# render!
				bpy.ops.render.render()

		return ST_FINISHED
