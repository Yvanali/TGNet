import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

pointSIFT_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_pointSIFT_so.so'))



np.random.seed(0)
X = np.random.random((10, 3))
radius = 0.1
idx = pointSIFT_module.cube_select(X, radius)
print(idx)
