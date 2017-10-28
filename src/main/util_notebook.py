"""
	Initialization for jupyter-notebook files
""" 

from util import *
from scipy.misc import imsave, imread
import matplotlib as mpl
import matplotlib.pyplot as plt

from ipywidgets import FloatProgress, IntProgress, HBox, HTML
from IPython.display import display, Javascript

# Init display style
get_ipython().magic('matplotlib inline')
np.set_printoptions(suppress=True, linewidth=180)

# - disable scroll:
display(Javascript("""
	IPython.OutputArea.prototype._should_scroll = function(lines) {
		return false;
	}
"""))

# Init code formatting
get_ipython().run_cell_magic("javascript", "", """
	var cell = Jupyter.notebook.get_selected_cell()
	var config = cell.config
	var patch = {
		CodeCell: {
			cm_config: {
				indentWithTabs: true,
				indentUnit: 4,
			} //only change here.
		}
	}
	config.update(patch)
""")

# Display arrays:

# - communicate signs through colors
CMAP_POS = 'gray' # colormap for positive images
CMAP_NEG = 'bone' # colormap for negative images

CMAP_PN = 'bwr' # colormap for mixed-sign images

def show_array(ax, mat, extent_stdev=None, cmap=None):
	ax.axis('off')
	
	if cmap:
		ax.imshow(mat, cmap=cmap)
	else:
		# check if consistent sign
		minval = float(mat.min())
		maxval = float(mat.max())
		
		# mix of positive and negative
		if minval*maxval < 0:
			if extent_stdev is None:
				# extent uniformly around 0
				extent = np.maximum(-minval, maxval)
			else:
				extent = np.std(mat) * extent_stdev
				print('Extent stdev', extent)
		
			# use positive-negative colormap
			ax.imshow(mat, vmin= -extent, vmax= extent,
				cmap=CMAP_PN)
		# consistent sign
		else:
			ax.imshow(mat, cmap=CMAP_POS if maxval > 0 else CMAP_NEG)

def show_one(data, figsize=(15, 10), save=None, colorbar=False, **options):
	fig = plt.figure(figsize=figsize)
	
	if isinstance(data, list):
		size = len(data)

		for idx, submat in enumerate(data):
			ax = fig.add_subplot(1, size, idx+1)
			show_array(ax, submat, **options)
	else:
		ax = fig.add_subplot(1, 1, 1)
		show_array(ax, data, **options)

	fig.tight_layout()
	if colorbar:
		fig.colorbar()

	if save:
		fig.savefig(save)

def show(*args, **options):
	for a in args:
		show_one(a, **options)

def show_multidim(imgs, figscale=(4, 3), col_titles = None, row_titles = None, save=None, colorbar=False, **options):
	"""
	@param imgs: 2-d nested array
	"""

	rs = len(imgs)
	cs = len(imgs[0])

	fig = plt.figure(figsize = (figscale[0]*cs, figscale[1]*rs))

	idx = 1
	
	for r in range(rs):
		for c in range(cs):
			ax = fig.add_subplot(rs, cs, idx)
			ax.axis('off')
			show_array(ax, imgs[r][c], **options)

			if col_titles and r == 0:
				ax.set_title(col_titles[c])

			if row_titles and r != 0:
				ax.set_title(row_titles[r])

			idx += 1

	if colorbar:
		fig.colorbar()
	fig.tight_layout()


	if save:
		save_plot(fig, save)

	return fig

def print_time():
	print(datetime.datetime.now().isoformat(sep='_'))

# IPY Progress Bar
class ProgressBar:
	def __init__(self, goal):
		self.bar = IntProgress(min=0, max=goal, value=0)
		self.label = HTML()
		box = HBox(children=[self.bar, self.label])

		self.goal = goal
		self.value = 0
		self.template = '{{0}} / {goal}'.format(goal=goal)
		
		self.set_value(0)

		display(box)
		
	def set_value(self, new_val):
		self.value = new_val
		self.bar.value = new_val
		self.label.value = self.template.format(new_val)
	
		if new_val >= self.goal:
			self.bar.bar_style = 'success'
	
	def __iadd__(self, change):
		self.set_value(self.value + change)
		return self
