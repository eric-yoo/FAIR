MNIST_TFDS_DIR = "tfds" #@param {type:"string"}

import tensorflow as tf
import tensorflow_datasets as tfds

def make_get_dataset(split, batch_size, with_index=True):
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

        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds
    return get_dataset() 