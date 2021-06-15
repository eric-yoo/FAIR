MNIST_TFDS_DIR = "tfds" #@param {type:"string"}

import sys
import numpy as np
import random
import tensorflow as tf
import tensorflow_datasets as tfds
from config import args

def corrupt(index, data, corrupt_indices, poisoned_label):
    if corrupt_indices[index] == 1:
        return {'correct_label':data['label'], 'label':tf.constant(poisoned_label, tf.int64), 'image': data['image']}
    else:
        return {'correct_label':data['label'], 'label':data['label'], 'image': data['image']}

def reverse_corrupt(index, data, corrupt_indices, poisoned_labels):
    if corrupt_indices[index] == 1:
        return {'correct_label':data['label'], 'label':tf.cast(poisoned_labels[index], tf.int64), 'image': data['image']}
    else:
        return {'correct_label':data['label'], 'label':data['label'], 'image': data['image']}

def normalize(data):
    if 'correct_label' in data:
        return {'correct_label':data['correct_label'], 'label':data['label'], 'image': tf.cast(data['image'], tf.float32) / 255.}
    else:    
        return {'label':data['label'], 'image': tf.cast(data['image'], tf.float32) / 255.}
    

def make_mnist_dataset(split, batch_size, with_index=True, 
                        is_poisoned=True, poisoned_ratio=.3, poisoned_label=2, 
                        is_reverse_poisoned=False, reverse_poison_ratio=.3, reverse_poison_label=2,
                        is_mislabeled=False, mislabel_ratio=.1
                        ):
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
            try:
                begin = int(split.split('[')[1].split('%:')[0])
            except:
                begin = None
            try:
                end = int(split.split(':')[1].split('%]')[0])
            except:
                end = None
            indices_range = {'train':range(60000), 'test':range(60000, 70000),
                            F'train[{begin}:{end}%]':range(600*20), 'train[20%:]':range(600*20,600*100),
                            **{F'train[{k}%:{k+10}%]':range(600*k, 600*(k+10)) for k in range(20, 100, 10)}
                            }
            if begin and end:
                indices_range.update({F'train[{begin}%:{end}%]':range(600*begin, 600*end)})
            elif begin:
                indices_range.update({F'train[{begin}%:]':range(600*begin, 60000)})
            elif end:
                indices_range.update({F'train[:{end}%]':range(600*end)})
            else:
                pass
            indices =  tf.data.Dataset.from_tensor_slices(list(indices_range[split]))
            ds = tf.data.Dataset.zip((indices, ds))

        if is_poisoned:
            assert 'train' in split
            np.random.seed(args.seed)
            masks = np.zeros((60000,), int)
            mask_indices = np.random.choice(range(60000), size=int(60000*poisoned_ratio), replace=False)
            masks[mask_indices] = 1
            corrupt_indices = tf.convert_to_tensor(masks, tf.int64)
            ds = ds.map(lambda index, data: (index, corrupt(index, data, corrupt_indices, poisoned_label=poisoned_label)))
        
        if is_reverse_poisoned:
            assert 'train' in split
            masks = np.zeros((60000,), int)
            np.random.seed(args.seed)
            mask_indices = np.random.choice(range(60000), size=int(60000*reverse_poison_ratio), replace=False)
            masks[mask_indices] = 1
            corrupt_indices = tf.convert_to_tensor(masks, tf.int64)
            poisoned_label_options = list(range(10))
            poisoned_label_options.remove(reverse_poison_label)
            np.random.seed(args.seed)
            poisoned_labels = tf.convert_to_tensor(random.choices(poisoned_label_options, k=60000), tf.int64)

            ds = ds.map(lambda index, data: (index, reverse_corrupt(index, data, corrupt_indices, poisoned_labels=poisoned_labels)))
        
        
        # if is_noisy:
        #     assert 'train' in split
        #     np.random.seed(args.seed)
        #     masks = np.zeros((60000,), int)
        #     mask_indices = np.random.choice(range(60000), size=int(60000*poisoned_ratio), replace=False)
        #     masks[mask_indices] = 1
        #     corrupt_indices = tf.convert_to_tensor(masks, tf.int64)
        #     ds = ds.map(lambda index, data: (index, corrupt(index, data, corrupt_indices, poisoned_label=poisoned_label)))
        
        # if is_mislabeled:
        #     assert 'train' in split
        #     np.random.seed(args.seed)
        #     masks = np.zeros((60000,), int)
        #     mask_indices = np.random.choice(range(60000), size=int(60000*mislabel_ratio), replace=False)
        #     masks[mask_indices] = 1
        #     corrupt_indices = tf.convert_to_tensor(masks, tf.int64)
        #     ds = ds.map(lambda index, data: (index, corrupt(index, data, corrupt_indices, poisoned_label=poisoned_label)))
        

        ds = ds.map( lambda index, data: (index, normalize(data)))
        counts = [0]*10
        for d in ds:
            counts[d[1]['label'].numpy()] +=1
        print(split, counts)

        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds
    return get_dataset() 


def make_femnist_dataset(split, batch_size, with_index=True, is_poisoned=True, poisoned_ratio=.1, poisoned_label=2):
    def get_dataset() -> tf.data.Dataset:
        builder = tfds.builder(name='fashion_mnist', data_dir=MNIST_TFDS_DIR)
        builder.download_and_prepare()

        read_config = tfds.ReadConfig(
            interleave_block_length=1)

        ds = builder.as_dataset(
            split=split,
            as_supervised=False,
            shuffle_files=False,
            read_config=read_config)

        if with_index:
            try:
                begin = int(split.split('[')[1].split('%:')[0])
            except:
                begin = None
            try:
                end = int(split.split(':')[1].split('%]')[0])
            except:
                end = None
            indices_range = {'train':range(60000), 'test':range(60000, 70000),
                            F'train[{begin}:{end}%]':range(600*20), 'train[20%:]':range(600*20,600*100),
                            **{F'train[{k}%:{k+10}%]':range(600*k, 600*(k+10)) for k in range(20, 100, 10)}
                            }
            if begin and end:
                indices_range.update({F'train[{begin}%:{end}%]':range(600*begin, 600*end)})
            elif begin:
                indices_range.update({F'train[{begin}%:]':range(600*begin, 60000)})
            elif end:
                indices_range.update({F'train[:{end}%]':range(600*end)})
            else:
                pass
            indices =  tf.data.Dataset.from_tensor_slices(list(indices_range[split]))
            ds = tf.data.Dataset.zip((indices, ds))

        if is_poisoned:
            assert 'train' in split
            np.random.seed(args.seed)
            masks = np.zeros((60000,), int)
            mask_indices = np.random.choice(range(60000), size=int(60000*poisoned_ratio), replace=False)
            masks[mask_indices] = 1
            corrupt_indices = tf.convert_to_tensor(masks, tf.int64)
            ds = ds.map(lambda index, data: (index, corrupt(index, data, corrupt_indices, poisoned_label=poisoned_label)))
        
        # if is_noisy:
        #     assert 'train' in split
        #     np.random.seed(args.seed)
        #     masks = np.zeros((60000,), int)
        #     mask_indices = np.random.choice(range(60000), size=int(60000*poisoned_ratio), replace=False)
        #     masks[mask_indices] = 1
        #     corrupt_indices = tf.convert_to_tensor(masks, tf.int64)
        #     ds = ds.map(lambda index, data: (index, corrupt(index, data, corrupt_indices, poisoned_label=poisoned_label)))
        
        # if is_mislabeled:
        #     assert 'train' in split
        #     np.random.seed(args.seed)
        #     masks = np.zeros((60000,), int)
        #     mask_indices = np.random.choice(range(60000), size=int(60000*mislabel_ratio), replace=False)
        #     masks[mask_indices] = 1
        #     corrupt_indices = tf.convert_to_tensor(masks, tf.int64)
        #     ds = ds.map(lambda index, data: (index, corrupt(index, data, corrupt_indices, poisoned_label=poisoned_label)))
        

        ds = ds.map( lambda index, data: (index, normalize(data)))
        counts = [0]*10
        for d in ds:
            counts[d[1]['label'].numpy()] +=1
        print(split, counts)

        ds = ds.batch(batch_size)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds
    return get_dataset() 