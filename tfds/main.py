MNIST_TFDS_DIR = "tfds" #@param {type:"string"}

import sys
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

def corrupt(index, data, corrupt_indices):
    if corrupt_indices[index] == 1:
        return {'label':tf.constant(2, tf.int64), 'image': data['image'], 'corrupt': 1}
    else:
        return {'label':data['label'], 'image': data['image'], 'corrupt': 0}
    

def normalize(data):
    return {'label':data['label'], 'image': tf.cast(data['image'], tf.float32) / 255.}
    

def make_get_dataset(split, batch_size, with_index=True, is_corrupt=True):
    def get_dataset() -> tf.data.Dataset:
        builder = tfds.builder(name='mnist', data_dir=MNIST_TFDS_DIR)
        builder.download_and_prepare()

        read_config = tfds.ReadConfig(
            interleave_block_length=1)

        ds = builder.as_dataset(
            split=split,
            as_supervised=False,
            shuffle_files=False,
            read_config=read_config)

        if with_index:
            indices_range = {'train':range(60000), 'test':range(60000, 70000)}
            indices =  tf.data.Dataset.from_tensor_slices(list(indices_range[split]))
            ds = tf.data.Dataset.zip((indices, ds))

        ds = ds.map( lambda index, data: (index, normalize(data)))
        if is_corrupt:
            assert split == 'train'
            np.random.seed(0)
            masks = np.zeros((60000,), int)
            mask_indices = np.random.choice(range(60000), size=60000//5, replace=False)
            masks[mask_indices] = 1
            corrupt_indices = tf.convert_to_tensor(masks, tf.int64)
            ds = ds.map(lambda index, data: (index, corrupt(index, data, corrupt_indices)))
        # counts = [0]*10
        # for d in ds:
        #     counts[d[1]['label'].numpy()] +=1
        # print(counts)

        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds
    return get_dataset() 
