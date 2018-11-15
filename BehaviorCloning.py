import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import math
import time
import os
from sklearn.utils import shuffle

class GraphGenerator(object):
    def __init__(self, size, bbeta, dim_ao, learning_rate):
        self.size = size
        self.bbeta = bbeta
        self.dim_o = dim_ao[0]
        self.dim_a = dim_ao[1]
        self.learning_rate = learning_rate
        self._gen_graph()

    def _gen_graph(self):
        self.bc_graph = tf.Graph()
        with self.bc_graph.as_default():
            self.graph_o = tf.placeholder(tf.float32, shape=(None, self.dim_o))
            self.graph_a = tf.placeholder(tf.float32, shape=(None, self.dim_a))

            self.logit = self._infer_graph(self.graph_o)

            self.loss = self._define_loss(self.logit, self.graph_a)

            self.trainer = self._define_trainer(self.loss)

            self.init = tf.global_variables_initializer()

            self.saver = tf.train.Saver()

    def _get_params(self, shape=None, regularizer=None):
        w = tf.get_variable('weights', 
            initializer=tf.truncated_normal(shape, stddev = math.sqrt(1.0 / float(shape[0]))),
            regularizer=regularizer
            )
        b = tf.get_variable('bias',
            initializer=tf.zeros([shape[1]])
            )
        return w, b


    def _infer_graph(self, o):
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.bbeta)
        with tf.variable_scope('h_layer_1'):
            w,b = self._get_params([self.dim_o, self.size[0]], regularizer)
            h_layer_1 = tf.nn.tanh(tf.matmul(o, w) + b)

        with tf.variable_scope('h_layer_2'):
            w,b = self._get_params([self.size[0], self.size[1]], regularizer)
            h_layer_2 = tf.nn.tanh(tf.matmul(h_layer_1, w) + b)

        with tf.variable_scope('h_layer_3'):
            w,b = self._get_params([self.size[1], self.size[2]], regularizer)
            h_layer_3 = tf.nn.tanh(tf.matmul(h_layer_2, w) + b)

        with tf.variable_scope('logit_layer'):
            w,b = self._get_params([self.size[2], self.dim_a], regularizer)
            logit = tf.matmul(h_layer_3, w) + b

        return logit

    def _define_loss(self, logit, a):
        loss = tf.losses.mean_squared_error(a, logit)
        reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_loss = tf.reduce_sum(reg_variables)
        return loss + reg_loss

    def _define_trainer(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1 = 0.9, beta2 = 0.95)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        trainer = optimizer.minimize(loss, global_step=global_step)
        return trainer

class BCPolicy(object):
    def __init__(self, bc_graph, train_steps = None):
        self.train_steps = train_steps
        self.batch_size = 1024
        self.graph_generator = bc_graph
        self.sess = tf.Session(graph = self.graph_generator.bc_graph)
        self.sess.run(self.graph_generator.init)

    def _load_data(self, expert_data):
        """
        preprocess
        """
        def _resize_obs(obs, act):
            obs_rs = tf.reshape(obs, (1, -1))
            return obs_rs, act

        assert expert_data['observations'].shape[0] == expert_data['actions'].shape[0]
        with self.graph_generator.bc_graph.as_default():
            X, y = shuffle(expert_data['observations'], expert_data['actions'])
            data = tf.contrib.data.Dataset.from_tensor_slices(
                (X, y)
            )
            data_iterator = data.map(_resize_obs).repeat().make_one_shot_iterator()
            nxt_o, nxt_a = data_iterator.get_next()
        return nxt_o, nxt_a


    def fit(self, expert_data):
        tf_o, tf_a = self._load_data(expert_data)
        start_time= time.time()
        for step in range(self.train_steps):
            obs, acts = [], []
            for _ in range(self.batch_size):
                o = self.sess.run(tf_o).ravel()
                a = self.sess.run(tf_a).ravel()
                obs.append(o)
                acts.append(a)
            obs = np.array(obs)
            acts = np.array(acts)
            feed_dict = {
                self.graph_generator.graph_o: obs,
                self.graph_generator.graph_a: acts,
            }
            _, l = self.sess.run(
                [self.graph_generator.trainer, self.graph_generator.loss], feed_dict = feed_dict
            )
            if step % 100 == 0:
                time_span = time.time() - start_time
                print('step {0} / {1}: loss = {2}, spent {3} mins'.format(step, self.train_steps, l, round(time_span/60, 1)))

    def save(self, path):
        save_path = self.graph_generator.saver.save(self.sess, os.path.join(os.getcwd()+'/'+path, 'model.ckpt'))

    def eval(self, path):
        self.graph_generator.saver.restore(self.sess, os.path.join(os.getcwd()+'/'+path, 'model.ckpt'))

    def evaluate(self, o):
        feed_dict = {
            self.graph_generator.graph_o: o,
        }
        return self.sess.run(self.graph_generator.logit, feed_dict = feed_dict)



def main():
    pass

if __name__ == '__main__':
    main()

    


