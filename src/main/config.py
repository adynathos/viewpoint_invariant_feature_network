
DIR_BASE = pp('..', '..')

DIR_DATASETS = pp(DIR_BASE, 'datasets')
DIR_OUT = pp(DIR_BASE, 'out')

DIR_OUT_MODELS = pp(DIR_OUT, 'models')
DIR_OUT_EVAL = pp(DIR_OUT, 'eval')
DIR_OUT_FIGURES = pp(DIR_OUT, 'figures')

CFG_BASE = AttrDict(
	dir_datasets = DIR_DATASETS,
	dir_out = DIR_OUT,
	dir_out_models = DIR_OUT_MODELS,
	dir_out_eval = DIR_OUT_EVAL,
	dir_out_figures = DIR_OUT_FIGURES,
)
