import diConstants as di
import os

import numpy as np
import pandas as pd
from keras.models import Sequential, model_from_json
from scipy.stats.stats import pearsonr
from sklearn.metrics import precision_score
import rpy2.robjects as robjects
from rpy2.robjects.packages import STAP
import h5py

from subprocess import call

call('mkdir %s' % di.DATA_ROOT, shell=True)
call('mkdir %s' % di.MODELS_ROOT, shell=True)
call('mkdir %s' % di.RESULTS_ROOT, shell=True)

call('mkdir %s' % di.RAW_ROOT, shell=True)
call('mkdir %s' % di.MERGED_ROOT, shell=True)
call('mkdir %s' % di.SUBSAMPLED_ROOT, shell=True)
call('mkdir %s' % di.BIGWIGS_ROOT, shell=True)
call('mkdir %s' % di.INTERVALS_ROOT, shell=True)
call('mkdir %s' % di.NUMPY_ROOT, shell=True)
call('mkdir %s' % di.PEAK_BASE_DIR, shell=True)
call('mkdir -p %s' % di.PEAK_GAPPED_DIR, shell=True)
call('mkdir %s' % di.DATASETS_ROOT, shell=True)
call('mkdir %s' % di.BASE_ROOT, shell=True)
call('mkdir %s' % di.BASE_BIGWIG_ROOT, shell=True)
call('mkdir %s' % di.SEQ_ROOT, shell=True)
call('mkdir %s' % di.WEIGHTS_ROOT, shell=True)
call('mkdir %s' % di.LOSS_ROOT, shell=True)
call('mkdir %s' % di.HIST_ROOT, shell=True)
call('mkdir %s' % di.EVAL_ROOT, shell=True)
