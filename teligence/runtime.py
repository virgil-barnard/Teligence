import random

import numpy as np
import tensorflow as tf


def setup_runtime(seed: int = 42):
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
