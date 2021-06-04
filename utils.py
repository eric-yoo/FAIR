from tensorflow.python.data.ops.dataset_ops import Dataset


import tensorflow_datasets as tfds
import numpy as np

def magic_parser(tfds_dataset):
  images = np.concatenate([(data['image']) for _, data in list(tfds.as_numpy(tfds_dataset))])
  labels = np.concatenate([(data['label']) for _, data in list(tfds.as_numpy(tfds_dataset))])
  return images, labels
