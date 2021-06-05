import tensorflow_datasets as tfds
import numpy as np
from config import args

def magic_parser(tfds_dataset):
  images = np.concatenate([(data['image']) for _, data in list(tfds.as_numpy(tfds_dataset))])
  labels = np.concatenate([(data['label']) for _, data in list(tfds.as_numpy(tfds_dataset))])
  return images, labels

def get_length(tfds_dataset):
  return (len(list(tfds.as_numpy(tfds_dataset))))