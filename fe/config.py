import numpy as np

CENTRALITY_WINDOW_SIZES = [10, 15]
CUTOFF_FREQUENCIES = [0.5, 1.5]
CENTRALITY_WINDOW_FUNS = (np.min, np.max, np.mean, np.std, np.median)
