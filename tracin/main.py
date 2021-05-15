CHECKPOINTS_PATH_FORMAT = "simpleNN/ckpt{}" 

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from simpleNN import network


class TracIn:
    def __init__(self, ds_train, ds_test, ckpt1='', ckpt2='', ckpt3='', verbose=True):
        self.verbose = verbose
        self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        
        # dataset
        self.ds_train = ds_train
        self.ds_test = ds_test

        # model
        if '' in [ckpt1, ckpt2, ckpt3]:
            # raise NotImplementedError
            self.debug('need train.')
            model = network.model()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
            )
            for i in range(1,11):
                for d in self.ds_train:
                    model.fit(d[1]['image'], d[1]['label'])
                model.save_weights(CHECKPOINTS_PATH_FORMAT.format(i))
            ckpt1 = CHECKPOINTS_PATH_FORMAT.format(1)
            ckpt2 = CHECKPOINTS_PATH_FORMAT.format(2)
            ckpt3 = CHECKPOINTS_PATH_FORMAT.format(3)
        # split model into two parts
        self.models_penultimate = []
        self.models_last = []
        self.models = []

        for ckpt in [ckpt1, ckpt2, ckpt3]:
            model = network.model()
            model.load_weights(ckpt).expect_partial()
            self.models_penultimate.append(tf.keras.Model(model.layers[0].input, model.layers[-3].output))
            self.models_last.append(model.layers[-2])
            self.models.append(model)
        
        self.trackin_train = self.get_trackin_grad(self.ds_train)
        
        self.trackin_test = self.get_trackin_grad(self.ds_test)
        
        self.trackin_train_self_influences = self.get_self_influence(ds_train)

    def debug(self, s):
        if self.verbose:
            print(s)

    def get_trackin_grad(self, ds):
        images_np = []
        image_ids_np = []
        loss_grads_np = []
        activations_np = []
        labels_np = []
        probs_np = []
        predicted_labels_np = []
        for d in ds:
            images_replicas, imageids_replicas, loss_grads_replica, activations_replica, labels_replica, probs_replica, predictied_labels_replica = self.strategy.run(self.run, args=(d,))
            for images, imageids, loss_grads, activations, labels, probs, predicted_labels in zip(
                self.strategy.experimental_local_results(images_replicas), 
                self.strategy.experimental_local_results(imageids_replicas), 
                self.strategy.experimental_local_results(loss_grads_replica),
                self.strategy.experimental_local_results(activations_replica), 
                self.strategy.experimental_local_results(labels_replica), 
                self.strategy.experimental_local_results(probs_replica), 
                self.strategy.experimental_local_results(predictied_labels_replica)):
                if imageids.shape[0] == 0:
                    continue
                images_np.append(images.numpy())
                image_ids_np.append(imageids.numpy())
                loss_grads_np.append(loss_grads.numpy())
                activations_np.append(activations.numpy())
                labels_np.append(labels.numpy())
                probs_np.append(probs.numpy())
                predicted_labels_np.append(predicted_labels.numpy())
        return {'images': np.concatenate(images_np),
                'image_ids': np.concatenate(image_ids_np),
                'loss_grads': np.concatenate(loss_grads_np),
                'activations': np.concatenate(activations_np),
                'labels': np.concatenate(labels_np),
                'probs': np.concatenate(probs_np),
                'predicted_labels': np.concatenate(predicted_labels_np)
                }    


    def get_self_influence(self, ds):
        images_np = []
        image_ids_np = []
        self_influences_np = []
        labels_np = []
        probs_np = []
        predicted_labels_np = []
        correct_labels_np = []
        for d in ds:
            images_replicas, imageids_replicas, self_influences_replica, labels_replica, probs_replica, predictied_labels_replica, correct_labels_replica = self.strategy.run(self.run_self_influence, args=(d,))  
            for images, imageids, self_influences, labels, probs, predicted_labels, correct_labels in zip(
                self.strategy.experimental_local_results(images_replicas), 
                self.strategy.experimental_local_results(imageids_replicas), 
                self.strategy.experimental_local_results(self_influences_replica), 
                self.strategy.experimental_local_results(labels_replica), 
                self.strategy.experimental_local_results(probs_replica), 
                self.strategy.experimental_local_results(predictied_labels_replica),
                self.strategy.experimental_local_results(correct_labels_replica)):
                if imageids.shape[0] == 0:
                    continue
                images_np.append(images.numpy())
                image_ids_np.append(imageids.numpy())
                self_influences_np.append(self_influences.numpy())
                labels_np.append(labels.numpy())
                probs_np.append(probs.numpy())
                predicted_labels_np.append(predicted_labels.numpy()) 
                correct_labels_np.append(correct_labels.numpy())
        return {'images': np.concatenate(images_np),
                'image_ids': np.concatenate(image_ids_np),
                'self_influences': np.concatenate(self_influences_np),
                'labels': np.concatenate(labels_np),
                'probs': np.concatenate(probs_np),
                'predicted_labels': np.concatenate(predicted_labels_np),
                'correct_labels': np.concatenate(correct_labels_np)
                }    

    @tf.function
    def run_self_influence(self, inputs):
        imageids, data = inputs
        images = data['image']
        labels = data['label']
        correct_labels = data['correct_label']
        self_influences = []
        for m in self.models:
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(m.trainable_weights[-2:])
                probs = m(images)
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
            grads = tape.jacobian(loss, m.trainable_weights[-2:])
            scores = tf.add_n([tf.math.reduce_sum(
                grad * grad, axis=tf.range(1, tf.rank(grad), 1)) 
                for grad in grads])
            self_influences.append(scores)  

        # Using probs from last checkpoint
        probs, predicted_labels = tf.math.top_k(probs, k=1)
        return images, imageids, tf.math.reduce_sum(tf.stack(self_influences, axis=-1), axis=-1), labels, probs, predicted_labels, correct_labels

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
        return images, imageids, tf.stack(loss_grads, axis=-1), tf.stack(activations, axis=-1), labels, probs, predicted_labels


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
                self.trackin_train['predicted_labels'][index][0],
                self.trackin_train['labels'][index], 
                scores[index],
                scores_lg[index] if scores_lg else None,
                scores_a[index] if scores_a else None))
            index = indices[i]
            opponents.append((
                self.trackin_train['image_ids'][index],
                self.trackin_train['probs'][index][0],
                self.trackin_train['predicted_labels'][index][0],
                self.trackin_train['labels'][index],
                scores[index],
                scores_lg[index] if scores_lg else None,
                scores_a[index] if scores_a else None))  
        return opponents, proponents


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
            trackin_dict['labels'][idx], 
            trackin_dict['probs'][idx][0], 
            trackin_dict['predicted_labels'][idx][0]))
        # img = self.get_image('test', trackin_dict['image_ids'][idx])
        img = self.trackin_test['images'][idx].reshape((28,28))
        plt.imshow(img)
        plt.show()
            
        self.debug("="*50)  
        self.debug('3 Proponents: ')
        for p in pp[:3]:
            self.debug('label: {}, prob: {}, predicted_label: {}, influence: {}'.format(p[3], p[1], p[2], p[4]))
            if p[5] and p[6]:
                self.debug('error_similarity: {}, encoding_similarity: {}'.format(p[5], p[6]))
            # img = self.get_image('train', p[0])
            img = self.trackin_train['images'][p[0]].reshape((28,28))
            if img is not None:
                plt.imshow(img, interpolation='nearest')
                plt.show()  
        self.debug("="*50)
        self.debug('3 Opponents: ')
        for o in op[:3]:
            self.debug('label: {}, prob: {}, predicted_label: {}, influence: {}'.format(o[3], o[1], o[2], o[4]))
            if o[5] and o[6]:
                self.debug('error_similarity: {}, encoding_similarity: {}'.format(o[5], o[6]))
            # img = self.get_image('train', o[0])
            img = self.trackin_train['images'][o[0]].reshape((28,28))
            if img is not None:
                plt.imshow(img, interpolation='nearest')
                plt.show()
        self.debug("="*50)
    
    def show_self_influence(self, trackin_self_influence, topk=50):
        self_influence_scores = trackin_self_influence['self_influences']
        indices = np.argsort(-self_influence_scores)
        for i, index in enumerate(indices[:topk]):
            self.debug('example {} (index: {})'.format(i, index))
            self.debug('correct_label: {}, label: {}, prob: {}, predicted_label: {}'.format(
                trackin_self_influence['correct_labels'][index],
                trackin_self_influence['labels'][index], 
                trackin_self_influence['probs'][index][0], 
                trackin_self_influence['predicted_labels'][index][0]))
            # img = get_image(trackin_self_influence['image_ids'][index])
            img = self.trackin_train_self_influences['images'][index].reshape((28,28))
            if img is not None:
                plt.imshow(img, interpolation='nearest')
                plt.show()


    def report_mislabel_detection(self, trackin_self_influence, num_dots=10):
        self_influence_scores = trackin_self_influence['self_influences']
        indices = np.argsort(-self_influence_scores)
        mislabel_detection_report = {}
        detected_mislabels = 0
        for i, index in enumerate(indices):
            # self.debug('example {} (index: {})'.format(i, index))
            # self.debug('correct_label: {}, label: {}, prob: {}, predicted_label: {}'.format(
            #     trackin_self_influence['correct_labels'][index],
            #     trackin_self_influence['labels'][index], 
            #     trackin_self_influence['probs'][index][0], 
            #     trackin_self_influence['predicted_labels'][index][0]))
            if trackin_self_influence['correct_labels'][index] != trackin_self_influence['labels'][index]:
                detected_mislabels += 1
            if i % (len(indices)//num_dots) == 0 or i == len(indices) -1:
                fraction_checked = (i+1) / len(indices)
                # print(fraction_checked, detected_mislabels)
                mislabel_detection_report[fraction_checked] = detected_mislabels
        mislabel_detection_report = {round(k,4):round(v/detected_mislabels,4) for k,v in mislabel_detection_report.items()}
        plt.plot(list(mislabel_detection_report.keys()), list(mislabel_detection_report.values()))
        plt.show()