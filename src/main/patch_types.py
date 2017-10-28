from enum import Enum

class PatchMode(Enum):
	INTENSITY = 1
	DEPTH = 2
	NORMALS = 3

patch_mode_to_channel_count = {
	PatchMode.INTENSITY: 1,
	PatchMode.DEPTH: 2,
	PatchMode.NORMALS: 4,
} 

patch_mode_to_name = {
	PatchMode.INTENSITY: 'int',
	PatchMode.DEPTH: 'depth',
	PatchMode.NORMALS: 'normals',
}