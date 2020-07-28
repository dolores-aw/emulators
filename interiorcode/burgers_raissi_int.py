"""
@author: Maziar Raissi
"""
import os
import sys

import pandas as pd

import interior_burgers
from plotting import newfig

sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import plotting
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_u, u, X_f, layers, lb, ub, nu, X_star, N_u, N_f):

        self.lb = lb
        self.ub = ub

        self.x_u = X_u[:, 0:1]
        self.t_u = X_u[:, 1:2]

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.u = u

        self.layers = layers
        self.nu = nu
        self.X_star = X_star

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        log_device_placement = False  # very loud
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=log_device_placement))

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50,
                                                                         'maxfun': 50,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)  # what is adaomoptimizer?

        log_device_placement = False  # used to be true, makes noise
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=log_device_placement))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        f = u_t + u * u_x - self.nu * u_xx

        return f

    def callback(self, loss):
        print('Loss:', loss)

    def train(self, nIter, burgers_data_loc,N_u, N_f, base_plt_dir):

        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}

        start_time = time.time()
        losses = {}
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                losses[it] = loss_value
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

            if (it & (it - 1)) == 0:
                inputs = interior_burgers.prepare_nn_inputs_burgers(burgers_data_loc, N_u, N_f, random_seed=1234, debugging=False)
                u_pred, f_pred = self.predict(self.X_star)

                plotting.plotting(inputs, u_pred, base_plt_dir, "{}".format(it))

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

        return losses

    def save_weights_and_biases(self, path):
        buffers = {}
        for i, w in enumerate(self.weights):
            buffers['weight{}'.format(i)] = (w.eval(self.sess))
        for i, w in enumerate(self.biases):
            buffers['bias{}'.format(i)] = (w.eval(self.sess))
        np.savez_compressed(path, **buffers)

    def load_weights_and_biases(self, path):
        bucket_o_wts = np.load(path)
        for k, numpy_arr in bucket_o_wts.items():
            if k.startswith('bias'):
                k = int(k[4:])
                tensor = self.biases[k]
            elif k.startswith('weight'):
                k = int(k[6:])
                tensor = self.weights[k]
            else:
                raise ValueError("unexpected in {!r}: {!r}".format(path, k))
            assign_op = tensor.assign(numpy_arr)
            self.sess.run(assign_op)

    def predict(self, X_star):

        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]})

        return u_star, f_star


if __name__ == "__main__":
    nIter = 10

    # nu = 0.01 / np.pi
    # noise = 0.0
    # N_u = 100
    # N_f = 10000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    #
    # data = scipy.io.loadmat('/Users/danamendelson/Desktop/burgers_shock.mat')
    #
    # t = data['t'].flatten()[:, None]
    # x = data['x'].flatten()[:, None]
    # Exact = np.real(data['usol']).T
    #
    # X, T = np.meshgrid(x, t)
    #
    # X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    # u_star = Exact.flatten()[:, None]
    #
    # # Domain bounds
    # lb = X_star.min(0)
    # ub = X_star.max(0)
    #
    # xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    # uu1 = Exact[0:1, :].T
    # xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
    # uu2 = Exact[:, 0:1]
    # xx3 = np.hstack((X[:, -1:], T[:, -1:]))
    # uu3 = Exact[:, -1:]
    #
    # X_u_train = np.vstack([xx1, xx2, xx3])
    # X_f_train = lb + (ub - lb) * lhs(2, N_f)
    # X_f_train = np.vstack((X_f_train, X_u_train))
    # u_train = np.vstack([uu1, uu2, uu3])
    #
    # idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    # X_u_train = X_u_train[idx, :]
    # u_train = u_train[idx, :]

    burgers_data_loc = '~/data/burgers_shock.mat'
    inputs = interior_burgers.prepare_nn_inputs_burgers(burgers_data_loc, N_u, N_f, random_seed=1234, debugging=False)

    model = PhysicsInformedNN(inputs.X_u_train, inputs.u_train, inputs.X_f_train, layers, inputs.lb, inputs.ub, inputs.nu, inputs.X_star, N_u, N_f)

    start_time = time.time()
    losses = model.train(nIter, '~/data/burgers_shock.mat', '~/plots')
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))

    u_pred, f_pred = model.predict(inputs.X_star)
    u_star = inputs.exact.flatten()[:, None]

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print('Error u: %e' % (error_u))

    t = inputs.t
    x = inputs.x
    X, T = np.meshgrid(x, t)

    U_pred = griddata(inputs.X_star, u_pred.flatten(), (X, T), method='cubic')
    Error = np.abs(inputs.exact - U_pred)

    plt.close()
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    pd.Series(losses).plot(logy=True, ax=ax)
    lp_loc = '/tmp/loss_plot.eps'
    plt.savefig(lp_loc)
    print("saved loss plot to {}".format(lp_loc))

    save_base_dir = '~/junk/eg_model'
    model.save_weights_and_biases(os.path.join(save_base_dir, 'weights_and_biases_2.npz'))

    u_pred, f_pred = model.predict(inputs.X_star)  # X_star = tf.convert_to_tensor(X_star) ?
    plotting.plotting(inputs, u_pred, '~/plots')
