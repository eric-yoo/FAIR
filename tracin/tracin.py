MNIST_TFDS_DIR = "tensorflow_datasets" #@param {type:"string"}
MNIST_TRAIN = "mnist/train" #@param {type:"string"}
MNIST_VAL = "mnist/validation" #@param {type:"string"}
CHECKPOINTS_PATH_FORMAT = "ckpt{}" #@param {type:"string"}

# @title Imports
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.image as mpimg
import io
import json
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import functools
import sys
sys.path.insert(0, "./simpleNN")
import network
#@title Dataset Utils
import numpy as np
import copy
from tensorflow.keras.datasets import mnist
import time
BATCH_SIZE = 512

strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

count = 0
def generate_id():
    global count
    print(count)
    fileid = tf.constant(count, dtype=tf.int64)
    # print('\n\n\n\n\n\n???????????\n\n\n','fileid')
    count+=1
    return fileid

# def _train_filename2id(filename):
#   filename = tf.strings.regex_replace(filename, "n", "")
#   filename = tf.strings.regex_replace(filename, ".JPEG", "")
#   filename_split = tf.strings.split(filename, "_")
#   fileid = tf.strings.to_number(filename_split, tf.int32)
#   return fileid

# def _val_filename2id(filename):
#   filename = tf.strings.regex_replace(filename, "ILSVRC2012_val_", "")
#   filename = tf.strings.regex_replace(filename, ".JPEG", "")
#   fileid = tf.strings.to_number(filename, tf.int32)
#   return fileid  

# def _resize_image(image_bytes: tf.Tensor,
#                  height: int = 224,
#                  width: int = 224) -> tf.Tensor:
#   """Resizes an image to a given height and width."""
#   return tf.compat.v1.image.resize(
#       image_bytes, [height, width], method=tf.image.ResizeMethod.BILINEAR,
#       align_corners=False) 

# # Calculated from the ImageNet training set
# MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
# STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)

# def mean_image_subtraction(
#     image_bytes,
#     means,
#     num_channels = 3,
#     dtype = tf.float32,
# ):
#   """Subtracts the given means from each image channel.

#   For example:
#     means = [123.68, 116.779, 103.939]
#     image_bytes = mean_image_subtraction(image_bytes, means)

#   Note that the rank of `image` must be known.

#   Args:
#     image_bytes: a tensor of size [height, width, C].
#     means: a C-vector of values to subtract from each channel.
#     num_channels: number of color channels in the image that will be distorted.
#     dtype: the dtype to convert the images to. Set to `None` to skip conversion.

#   Returns:
#     the centered image.

#   Raises:
#     ValueError: If the rank of `image` is unknown, if `image` has a rank other
#       than three or if the number of channels in `image` doesn't match the
#       number of values in `means`.
#   """
#   if image_bytes.get_shape().ndims != 3:
#     raise ValueError('Input must be of size [height, width, C>0]')

#   if len(means) != num_channels:
#     raise ValueError('len(means) must match the number of channels')

#   # We have a 1-D tensor of means; convert to 3-D.
#   # Note(b/130245863): we explicitly call `broadcast` instead of simply
#   # expanding dimensions for better performance.
#   means = tf.broadcast_to(means, tf.shape(image_bytes))
#   if dtype is not None:
#     means = tf.cast(means, dtype=dtype)

#   return image_bytes - means


# def standardize_image(
#     image_bytes,
#     stddev,
#     num_channels = 3,
#     dtype = tf.float32,
# ):
#   """Divides the given stddev from each image channel.

#   For example:
#     stddev = [123.68, 116.779, 103.939]
#     image_bytes = standardize_image(image_bytes, stddev)

#   Note that the rank of `image` must be known.

#   Args:
#     image_bytes: a tensor of size [height, width, C].
#     stddev: a C-vector of values to divide from each channel.
#     num_channels: number of color channels in the image that will be distorted.
#     dtype: the dtype to convert the images to. Set to `None` to skip conversion.

#   Returns:
#     the centered image.

#   Raises:
#     ValueError: If the rank of `image` is unknown, if `image` has a rank other
#       than three or if the number of channels in `image` doesn't match the
#       number of values in `stddev`.
#   """
#   if image_bytes.get_shape().ndims != 3:
#     raise ValueError('Input must be of size [height, width, C>0]')

#   if len(stddev) != num_channels:
#     raise ValueError('len(stddev) must match the number of channels')

#   # We have a 1-D tensor of stddev; convert to 3-D.
#   # Note(b/130245863): we explicitly call `broadcast` instead of simply
#   # expanding dimensions for better performance.
#   stddev = tf.broadcast_to(stddev, tf.shape(image_bytes))
#   if dtype is not None:
#     stddev = tf.cast(stddev, dtype=dtype)

#   return image_bytes / stddev   

# TPU does not allow tf.string and images with various size. Therefore, decode 
# and cropping cannot happen in the model.



def _preprocess(inputs, split='train'):
  """Apply image preprocessing."""
  image = inputs['image']
  label = inputs['label']
  if split == 'train':
    # print(1)
    fileid = generate_id()
  else:      
    fileid = generate_id()
  return fileid, image, label  


def make_get_dataset(split, batch_size):
  def get_dataset() -> tf.data.Dataset:
    builder = tfds.builder(name='mnist', data_dir=MNIST_TFDS_DIR)
    builder.download_and_prepare()

    read_config = tfds.ReadConfig(
        interleave_block_length=1)

    _preprocess_fn = functools.partial(_preprocess, split=split)
    ds = builder.as_dataset(
        split=split,
        as_supervised=False,
        shuffle_files=False,
        read_config=read_config)

    indices_range = {'train':range(60000), 'test':range(60000, 70000)}
    indices =  tf.data.Dataset.from_tensor_slices(list(indices_range[split]))
    ds = tf.data.Dataset.zip((indices, ds))

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds
  return get_dataset() 


#@title Search Utils

# def find(loss_grad=None, activation=None, topk=50):
#   if loss_grad is None and activation is None:
#     raise ValueError('loss grad and activation cannot both be None.')
#   scores = []
#   scores_lg = []
#   scores_a = []
#   for i in range(len(trackin_train['image_ids'])):
#     if loss_grad is not None and activation is not None:
#       lg_sim = np.sum(trackin_train['loss_grads'][i] * loss_grad)
#       a_sim = np.sum(trackin_train['activations'][i] * activation)
#       scores.append(lg_sim * a_sim)
#       scores_lg.append(lg_sim)
#       scores_a.append(a_sim)
#     elif loss_grad is not None:
#       scores.append(np.sum(trackin_train['loss_grads'][i] * loss_grad))
#     elif activation is not None:
#       scores.append(np.sum(trackin_train['activations'][i] * activation))    

#   opponents = []
#   proponents = []
#   indices = np.argsort(scores)
#   for i in range(topk):
#     index = indices[-i-1]
#     proponents.append((
#         trackin_train['image_ids'][index],
#         trackin_train['probs'][index][0],
#         index_to_classname[str(trackin_train['predicted_labels'][index][0])][1],
#         index_to_classname[str(trackin_train['labels'][index])][1], 
#         scores[index],
#         scores_lg[index] if scores_lg else None,
#         scores_a[index] if scores_a else None))
#     index = indices[i]
#     opponents.append((
#         trackin_train['image_ids'][index],
#         trackin_train['probs'][index][0],
#         index_to_classname[str(trackin_train['predicted_labels'][index][0])][1],
#         index_to_classname[str(trackin_train['labels'][index])][1],
#         scores[index],
#         scores_lg[index] if scores_lg else None,
#         scores_a[index] if scores_a else None))  
#   return opponents, proponents

# IMAGENET_LABEL_DICT = './imagenet_class_index.json'
# def get_id_synset_mapping():
#   imagenet_class_idx_path = IMAGENET_LABEL_DICT
#   with tf.io.gfile.GFile(imagenet_class_idx_path, "r") as f:
#     json_str = f.read()
#     index_to_classname = json.loads(json_str)
#   return index_to_classname  
# index_to_classname = get_id_synset_mapping()  

# def get_image(split, id):
#   if split == 'validation':
#     filepath = '{}/ILSVRC2012_val_{fileid:08d}.JPEG'.format(MNIST_VAL, fileid=id)
#     print('ILSVRC2012_val_{fileid:08d}.JPEG'.format(fileid=id))
#   else:
#     filepath = '{}/n0{}/n0{}_{}.JPEG'.format(MNIST_TRAIN, id[0], id[0], id[1])
#     print('n0{}_{}.JPEG'.format(id[0], id[1]))  
#   try:   
#     with tf.io.gfile.GFile(filepath, "rb") as f:
#       jpg_data = f.read()
#       image = mpimg.imread(io.BytesIO(jpg_data), format='JPG')     
#     return image
#   except:
#     print('Failed to read image {}'.format(filepath)) 

# def find_and_show(trackin_dict, idx, vector='influence', idx_filename_mapping=None):
#   if vector == 'influence':
#     op, pp = find(trackin_dict['loss_grads'][idx], trackin_dict['activations'][idx])
#   elif vector == 'encoding':
#     op, pp = find(None, trackin_dict['activations'][idx])  
#   elif vector == 'error':
#     op, pp = find(trackin_dict['loss_grads'][idx], None)
#   else:
#     raise ValueError('Unsupported vector type.')  
#   print('Query image from validation: ')
#   print('label: {}, prob: {}, predicted_label: {}'.format(
#       index_to_classname[str(trackin_dict['labels'][idx])][1], 
#       trackin_dict['probs'][idx][0], 
#       index_to_classname[str(trackin_dict['predicted_labels'][idx][0])][1]))
#   if idx_filename_mapping:
#     img = mpimg.imread(io.BytesIO(idx_filename_mapping[idx]), format='JPG')
#   else:  
#     img = get_image('validation', trackin_dict['image_ids'][idx])
#   if img is not None:
#     plt.imshow(img, interpolation='nearest')
#     plt.show()
#   print("="*50)  
#   print('Proponents: ')
#   for p in pp:
#     print('label: {}, prob: {}, predicted_label: {}, influence: {}'.format(p[3], p[1], p[2], p[4]))
#     if p[5] and p[6]:
#       print('error_similarity: {}, encoding_similarity: {}'.format(p[5], p[6]))
#     img = get_image('train', p[0])
#     if img is not None:
#       plt.imshow(img, interpolation='nearest')
#       plt.show()  
#   print("="*50)
#   print('Opponents: ')
#   for o in op:
#     print('label: {}, prob: {}, predicted_label: {}, influence: {}'.format(o[3], o[1], o[2], o[4]))
#     if o[5] and o[6]:
#       print('error_similarity: {}, encoding_similarity: {}'.format(o[5], o[6]))
#     img = get_image('train', o[0])
#     if img is not None:
#       plt.imshow(img, interpolation='nearest')
#       plt.show()
#   print("="*50)


def get_trackin_grad(ds):
  image_ids_np = []
  loss_grads_np = []
  activations_np = []
  labels_np = []
  probs_np = []
  predicted_labels_np = []
  for d in ds:
    # print('\n\n\n\n\n\n\n',type(d))
    imageids_replicas, loss_grads_replica, activations_replica, labels_replica, probs_replica, predictied_labels_replica = strategy.run(run, args=(d,))
    for imageids, loss_grads, activations, labels, probs, predicted_labels in zip(
        strategy.experimental_local_results(imageids_replicas), 
        strategy.experimental_local_results(loss_grads_replica),
        strategy.experimental_local_results(activations_replica), 
        strategy.experimental_local_results(labels_replica), 
        strategy.experimental_local_results(probs_replica), 
        strategy.experimental_local_results(predictied_labels_replica)):
      if imageids.shape[0] == 0:
        continue
      image_ids_np.append(imageids.numpy())
      loss_grads_np.append(loss_grads.numpy())
      activations_np.append(activations.numpy())
      labels_np.append(labels.numpy())
      probs_np.append(probs.numpy())
      predicted_labels_np.append(predicted_labels.numpy()) 
  # print(image_ids_np)
  return {'image_ids': np.concatenate(image_ids_np),
          'loss_grads': np.concatenate(loss_grads_np),
          'activations': np.concatenate(activations_np),
          'labels': np.concatenate(labels_np),
          'probs': np.concatenate(probs_np),
          'predicted_labels': np.concatenate(predicted_labels_np)
         }    

@tf.function
def run(inputs):
  imageids, data = inputs
  images = data['image']
  labels = data['label']
  # ignore bias for simplicity
  loss_grads = []
  activations = []
  for mp, ml in zip(models_penultimate, models_last):
    h = mp(images)
    logits = ml(h)
    probs = tf.nn.softmax(logits)
    loss_grad = tf.one_hot(labels, 10) - probs
    activations.append(h)
    loss_grads.append(loss_grad)

  # Using probs from last checkpoint
  probs, predicted_labels = tf.math.top_k(probs, k=1)

  return imageids, tf.stack(loss_grads, axis=-1), tf.stack(activations, axis=-1), labels, probs, predicted_labels



# ### load mnist
# (train_xs, train_ys), (test_xs, test_ys) = mnist.load_data()
# train_xs = train_xs / 255.
# test_xs = test_xs / 255.
# train_xs = train_xs.reshape(-1, 28 * 28)
# test_xs = test_xs.reshape(-1, 28 * 28)

# ### create biased mnist
# train_ys_corrupted = np.copy(train_ys)
# np.random.seed(12345)
# idxs = np.random.choice(range(len(train_ys_corrupted)), size=len(train_ys_corrupted)//5, replace=False)
# train_ys_corrupted[idxs] = 2


models_penultimate = []
models_last = []
for i in [30, 60, 90]:
  model = network.model()
  # model.load_weights(CHECKPOINTS_PATH_FORMAT.format(i))
  models_penultimate.append(tf.keras.Model(model.layers[0].input, model.layers[-2].output))
  models_last.append(model.layers[-1])
    
    
ds_train = make_get_dataset('train', BATCH_SIZE)


# ds_train = zip(list(range(len(train_xs))),train_xs, train_ys)
start = time.time()
trackin_train = get_trackin_grad(ds_train)
end = time.time()
print(datetime.timedelta(seconds=end - start))


ds_val = make_get_dataset('test', BATCH_SIZE)

# ds_val = zip(list(range(len(test_xs))),test_xs, test_ys)
start = time.time()
trackin_val = get_trackin_grad(ds_val)
end = time.time()
print(datetime.timedelta(seconds=end - start))