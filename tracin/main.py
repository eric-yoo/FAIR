import tensorflow as tf
import matplotlib.image as mpimg
import io
import json
import numpy as np
import time
import datetime
import matplotlib.pyplot as plt
import functools
import sys
from simpleNN import network
import numpy as np
import copy
from tensorflow.keras.datasets import mnist
import time


class TracIn:
    def __init__(self, ds_train, ds_test, ckpt1='', ckpt2='', ckpt3='', verbose=True):
        self.verbose = verbose
        self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        self.index_to_classname = {}

        # dataset
        self.ds_train = ds_train
        self.ds_test = ds_test

        # model
        if '' in [ckpt1, ckpt2, ckpt3]:
            raise NotImplementedError
            self.debug('need train.')
            model = network.model()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            )
            for i in range(1,5):
                for d in self.ds_train:
                    model.fit(d[1]['image'], d[1]['label'])
                model.save_weights(CHECKPOINTS_PATH_FORMAT.format(i))
        # split model into two parts
        self.models_penultimate = []
        self.models_last = []

        for ckpt in [ckpt1, ckpt2, ckpt3]:
            model = network.model()
            model.load_weights(ckpt).expect_partial()
            self.models_penultimate.append(tf.keras.Model(model.layers[0].input, model.layers[-3].output))
            self.models_last.append(model.layers[-2])
        
        self.trackin_train = self.get_trackin_grad(self.ds_train)
        # print(self.trackin_train['predicted_labels']); quit()
        ids = list(self.trackin_train['image_ids'])
        labels = list(self.trackin_train['labels'])
        for i,id in enumerate(ids):
            self.index_to_classname[id] = labels[i]
        
        self.trackin_test = self.get_trackin_grad(self.ds_test)
        ids = list(self.trackin_test['image_ids'])
        labels = list(self.trackin_test['labels'])
        for i,id in enumerate(ids):
            self.index_to_classname[id] = labels[i]

    def debug(self, s):
        if self.verbose:
            print(s)

    def get_trackin_grad(self, ds):
        image_ids_np = []
        loss_grads_np = []
        activations_np = []
        labels_np = []
        probs_np = []
        predicted_labels_np = []
        for d in ds:
            imageids_replicas, loss_grads_replica, activations_replica, labels_replica, probs_replica, predictied_labels_replica = self.strategy.run(self.run, args=(d,))
            for imageids, loss_grads, activations, labels, probs, predicted_labels in zip(
                self.strategy.experimental_local_results(imageids_replicas), 
                self.strategy.experimental_local_results(loss_grads_replica),
                self.strategy.experimental_local_results(activations_replica), 
                self.strategy.experimental_local_results(labels_replica), 
                self.strategy.experimental_local_results(probs_replica), 
                self.strategy.experimental_local_results(predictied_labels_replica)):
                if imageids.shape[0] == 0:
                    continue
                image_ids_np.append(imageids.numpy())
                loss_grads_np.append(loss_grads.numpy())
                activations_np.append(activations.numpy())
                labels_np.append(labels.numpy())
                probs_np.append(probs.numpy())
                predicted_labels_np.append(predicted_labels.numpy())
        return {'image_ids': np.concatenate(image_ids_np),
                'loss_grads': np.concatenate(loss_grads_np),
                'activations': np.concatenate(activations_np),
                'labels': np.concatenate(labels_np),
                'probs': np.concatenate(probs_np),
                'predicted_labels': np.concatenate(predicted_labels_np)
                }    


    @tf.function
    def run(self, inputs):
        imageids, data = inputs
        images = data['image']
        labels = data['label']
        # ignore bias for simplicity
        loss_grads = []
        activations = []
        for mp, ml in zip(self.models_penultimate, self.models_last):
            h = mp(images)
            logits = ml(h)
            probs = tf.nn.softmax(logits)
            loss_grad = tf.one_hot(labels, 10) - probs
            activations.append(h)
            loss_grads.append(loss_grad)

        # Using probs from last checkpoint
        probs, predicted_labels = tf.math.top_k(probs, k=1)

        return imageids, tf.stack(loss_grads, axis=-1), tf.stack(activations, axis=-1), labels, probs, predicted_labels


    def find(self, loss_grad=None, activation=None, topk=50):
        if loss_grad is None and activation is None:
            raise ValueError('loss grad and activation cannot both be None.')
        scores = []
        scores_lg = []
        scores_a = []
        for i in range(len(self.trackin_train['image_ids'])):
            if loss_grad is not None and activation is not None:
                lg_sim = np.sum(self.trackin_train['loss_grads'][i] * loss_grad)
                a_sim = np.sum(self.trackin_train['activations'][i] * activation)
                scores.append(lg_sim * a_sim)
                scores_lg.append(lg_sim)
                scores_a.append(a_sim)
            elif loss_grad is not None:
                scores.append(np.sum(self.trackin_train['loss_grads'][i] * loss_grad))
            elif activation is not None:
                scores.append(np.sum(self.trackin_train['activations'][i] * activation))    

        opponents = []
        proponents = []
        indices = np.argsort(scores)
        for i in range(topk):
            index = indices[-i-1]
            proponents.append((
                self.trackin_train['image_ids'][index],
                self.trackin_train['probs'][index][0],
                self.index_to_classname[self.trackin_train['predicted_labels'][index][0]],
                self.index_to_classname[self.trackin_train['labels'][index]], 
                scores[index],
                scores_lg[index] if scores_lg else None,
                scores_a[index] if scores_a else None))
            index = indices[i]
            opponents.append((
                self.trackin_train['image_ids'][index],
                self.trackin_train['probs'][index][0],
                self.index_to_classname[self.trackin_train['predicted_labels'][index][0]],
                self.index_to_classname[self.trackin_train['labels'][index]],
                scores[index],
                scores_lg[index] if scores_lg else None,
                scores_a[index] if scores_a else None))  
        return opponents, proponents

    def get_image(self, split, id):
        if split == 'test':
            for batch in self.ds_test:
                # print(batch[0])
                index = tf.where(batch[0] == id)
                if index.shape[0] == 1:
                    return (batch[1]['image'][index.numpy()[0][0]]).numpy().reshape((28,28))

        else:
            for batch in self.ds_train:
                index = tf.where(batch[0] == id)
                if index.shape[0] == 1:
                    return (batch[1]['image'][index.numpy()[0][0]]).numpy().reshape((28,28))
                

    def find_and_show(self, trackin_dict, idx, vector='influence'):
        if vector == 'influence':
            op, pp = self.find(trackin_dict['loss_grads'][idx], trackin_dict['activations'][idx])
        elif vector == 'encoding':
            op, pp = self.find(None, trackin_dict['activations'][idx])  
        elif vector == 'error':
            op, pp = self.find(trackin_dict['loss_grads'][idx], None)
        else:
            raise ValueError('Unsupported vector type.')
        self.debug('Query image from test: ')
        self.debug('label: {}, prob: {}, predicted_label: {}'.format(
            self.index_to_classname[trackin_dict['labels'][idx]], 
            trackin_dict['probs'][idx][0], 
            self.index_to_classname[trackin_dict['predicted_labels'][idx][0]]))
        
        img = self.get_image('test', trackin_dict['image_ids'][idx])
        plt.imshow(img)
        plt.show()
            
        self.debug("="*50)  
        self.debug('3 Proponents: ')
        for p in pp[:3]:
            for i in p:
                print(i, end='\t')
            print()
            self.debug('label: {}, prob: {}, predicted_label: {}, influence: {}'.format(p[3], p[1], p[2], p[4]))
            if p[5] and p[6]:
                self.debug('error_similarity: {}, encoding_similarity: {}'.format(p[5], p[6]))
            img = self.get_image('train', p[0])
            if img is not None:
                plt.imshow(img, interpolation='nearest')
                plt.show()  
        self.debug("="*50)
        self.debug('3 Opponents: ')
        for o in op[:3]:
            self.debug('label: {}, prob: {}, predicted_label: {}, influence: {}'.format(o[3], o[1], o[2], o[4]))
            if o[5] and o[6]:
                self.debug('error_similarity: {}, encoding_similarity: {}'.format(o[5], o[6]))
            img = self.get_image('train', o[0])
            if img is not None:
                plt.imshow(img, interpolation='nearest')
                plt.show()
        self.debug("="*50)