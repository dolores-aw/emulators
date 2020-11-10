"""
@author: Maziar Raissi
"""
import os
import sys
import warnings
import plotting

sys.path.insert(0, '../../Utilities/')  # todo: why?

import tensorflow as tf
import numpy as np
import scipy.io
from pyDOE import lhs
import time
import pickle

# import matplotlib.gridspec as gridspec
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.pyplot as plt
# from scipy.interpolate import griddata
# from plotting import newfig, savefig
# from mpl_toolkits.mplot3d import Axes3D


class PhysicsInformedNN(object):
    # Initialize the class
    def __init__(self, x0, u0, tb, X_f, layers, lb, ub, X_star, N_u, N_f):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            #tf.set_random_seed(1234)  # todo: consider timing.... this could be complicated

            X0 = np.concatenate((x0, 0 * x0), 1)  # (x0, 0)
            X_lb = np.concatenate((0 * tb + lb[0], tb), 1)  # (lb[0], tb)
            X_ub = np.concatenate((0 * tb + ub[0], tb), 1)  # (ub[0], tb)

            self.lb = lb
            self.ub = ub

            self.x0 = X0[:, 0:1]
            self.t0 = X0[:, 1:2]

            self.x_lb = X_lb[:, 0:1]
            self.t_lb = X_lb[:, 1:2]

            self.x_ub = X_ub[:, 0:1]
            self.t_ub = X_ub[:, 1:2]

            self.x_f = X_f[:, 0:1]
            self.t_f = X_f[:, 1:2]

            self.u0 = u0
            self.X_star = X_star

            # Initialize NNs
            self.layers = layers
            self.weights, self.biases = self.initialize_NN(layers)

            # tf Placeholders
            self.x0_tf = tf.placeholder(tf.float32, shape=[None, self.x0.shape[1]])
            self.t0_tf = tf.placeholder(tf.float32, shape=[None, self.t0.shape[1]])

            self.u0_tf = tf.placeholder(tf.float32, shape=[None, self.u0.shape[1]])

            self.x_lb_tf = tf.placeholder(tf.float32, shape=[None, self.x_lb.shape[1]])
            self.t_lb_tf = tf.placeholder(tf.float32, shape=[None, self.t_lb.shape[1]])

            self.x_ub_tf = tf.placeholder(tf.float32, shape=[None, self.x_ub.shape[1]])
            self.t_ub_tf = tf.placeholder(tf.float32, shape=[None, self.t_ub.shape[1]])

            self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
            self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

            self.u0_pred, _ = self.net_uv(self.x0_tf, self.t0_tf)
            self.u_lb_pred, self.u_x_lb_pred = self.net_uv(self.x_lb_tf, self.t_lb_tf)
            self.u_ub_pred, self.u_x_ub_pred = self.net_uv(self.x_ub_tf, self.t_ub_tf)
            self.f_u_pred = self.net_f_uv(self.x_f_tf, self.t_f_tf)

            self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                        tf.reduce_mean(tf.square(self.f_pred))
            #self.loss = tf.reduce_mean(tf.square(self.u0_tf - self.u0_pred)) + \
            #            tf.reduce_mean(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
            #            tf.reduce_mean(tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
            #            tf.reduce_mean(tf.square(self.f_u_pred))

            self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                    method='L-BFGS-B',
                                                                    options={'maxiter': 50000,
                                                                             'maxfun': 50000,
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

    def net_uv(self, x, t):
        X = tf.concat([x, t], 1)

        uv = self.neural_net(X, self.weights, self.biases)
        u = uv[:, 0:1]

        u_x = tf.gradients(u, x)[0]

        return u, u_x

    def net_f_uv(self, x, t):
        u, u_x = self.net_uv(x, t)

        u_t = tf.gradients(u, t)[0]
        u_xx = tf.gradients(u_x, x)[0]

        f_u = u_t + u * u_x - u_xx

        return f_u

    def callback(self, loss):
        print('Loss:', loss)

    def train(self, nIter, burgers_data_loc, base_plt_dir):

        tf_dict = {self.x0_tf: self.x0, self.t0_tf: self.t0,
                   self.u0_tf: self.u0,
                   self.x_lb_tf: self.x_lb, self.t_lb_tf: self.t_lb,
                   self.x_ub_tf: self.x_ub, self.t_ub_tf: self.t_ub,
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
                inputs = prepare_nn_inputs(burgers_data_loc, debugging=False)
                u_pred, f_pred = self.predict(self.X_star)
                plotting.plotting(inputs, u_pred, base_plt_dir, "{}".format(it))

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)
        return losses

    def predict(self, X_star):

        tf_dict = {self.x0_tf: X_star[:, 0:1], self.t0_tf: X_star[:, 1:2]}

        u_star = self.sess.run(self.u0_pred, tf_dict)

        tf_dict = {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]}

        f_u_star = self.sess.run(self.f_u_pred, tf_dict)

        return u_star, f_u_star


class NNInputs(object):
    def __init__(self, x0, u0, tb, X_f, lb, ub, X_star, x, t, exact):
        self.x0 = x0
        self.u0 = u0
        self.tb = tb
        self.X_f = X_f
        self.lb = lb
        self.ub = ub
        self.X_star = X_star
        self.x = x
        self.t = t
        self.exact = exact

    def __str__(self):
        s = "{}<x0={}".format(self.__class__.__name__, self.x0.shape)
        s += ",u0={}".format(self.u0.shape)
        s += ",tb={}".format(self.tb.shape)
        s += ",X_f={}".format(self.X_f.shape)
        s += ",lb={}".format(self.lb.shape)
        s += ",ub={}".format(self.ub.shape)
        s += ",X_star={}".format(self.X_star.shape)
        s += ",x={}".format(self.x.shape)
        s += ",t={}".format(self.t.shape)
        s += ",exact={}>".format(self.exact.shape)
        return s

    __repr__ = __str__


class NNInputs_burgers(object):
    def __init__(self, X_u_train, u_train, X_f_train, X_u_train2, u_train2, X_f_train2, lb, ub, nu, X_star, x, t, exact):
        self.X_u_train = X_u_train
        self.u_train = u_train
        self.X_f_train = X_f_train
        self.lb = lb
        self.ub = ub
        self.nu = nu
        self.X_star = X_star
        self.x = x
        self.t = t
        self.exact = exact
        self.X_u_train2 = X_u_train2
        self.u_train2 = u_train2
        self.X_f_train2 = X_f_train2

    # def __str__(self):
    #     s = "{}<x0={}".format(self.__class__.__name__, self.x0.shape)
    #     s += ",u0={}".format(self.u0.shape)
    #     s += ",tb={}".format(self.tb.shape)
    #     s += ",X_f={}".format(self.X_f.shape)
    #     s += ",lb={}".format(self.lb.shape)
    #     s += ",ub={}".format(self.ub.shape)
    #     s += ",X_star={}".format(self.X_star.shape)
    #     s += ",x={}".format(self.x.shape)
    #     s += ",t={}".format(self.t.shape)
    #     s += ",exact={}>".format(self.exact.shape)
    #     return s

    # __repr__ = __str__


def prepare_nn_inputs(data_loc, debugging=False):
    noise = 0.0

    # Domain bounds
    lb = np.array([-5.0, 0.0])
    ub = np.array([5.0, np.pi / 2])

    N0 = 50
    N_b = 50
    N_f = 20000

    data = scipy.io.loadmat(os.path.expanduser(data_loc))
    if debugging:
        for k in data:
            try:
                print(k, data[k].shape, data[k].dtype)
            except BaseException as be:
                print(k, be)

    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol'])

    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    ###########################
    #np.random.seed(random_seed)
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    idx_t = np.random.choice(t.shape[0], N_b, replace=False)
    x0 = x[idx_x, :]
    u0 = Exact[idx_x, 0:1]
    tb = t[idx_t, :]
    X_f = lb + (ub - lb) * lhs(2, N_f)
    return NNInputs(x0, u0, tb, X_f, lb, ub, X_star, x, t, Exact)


def prepare_nn_inputs_burgers(data_loc, N_u, N_f, N_u2, N_f2, m, typen, debugging=False):
    nu = 0.01 / np.pi
    #if typen == 'N_u':
    #    N_u = N_u - N_u2
    #    N_f = 2**N_f
    #elif typen == 'N_f':
    N_f = 2**N_f - N_f2
    data = scipy.io.loadmat(os.path.expanduser(data_loc))

    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T
    u_star = Exact.flatten()[:,None]
    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    
    #isolating points within m of x = 0
    X_star2 = []
    u_star2 = []
    k = 0
    for point in X_star:
        k = k + 1
        if (point[0] < m) & (point[0] > -m) :
            X_star2.append(point)
            u_star2.append(u_star[k])
    u_star2 = np.array(u_star2)
    X_star2 = np.array(X_star2)
    


    # Domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)
    lb2 = X_star2.min(0)
    ub2 = X_star2.max(0)


    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    idx2 = np.random.choice(X_star2.shape[0], N_u2, replace=False)
    X_u_train = X_star[idx,:]
    u_train = u_star[idx,:]
    X_u_train2 = X_star2[idx2,:]
    u_train2 = u_star2[idx2, :]

    X_f_train = lb + (ub - lb) * lhs(2, N_f) #a selection of 50000 points in domain
    X_f_train2 = lb2 + (ub2 - lb2) * lhs(2, N_f2)
    #X_f_train = np.vstack((X_f_train, X_u_train))
    #if typen == 'N_f':
    #X_f_train = np.vstack((X_f_train, X_f_train2))
    #if typen == 'N_u':
    #    print(X_u_train.shape)
    #    print(X_u_train2.shape)
    #    X_u_train = np.vstack((X_u_train, X_u_train2))
    #    u_train = np.vstack((u_train, u_train2))
    #    print(u_train.shape)
    #    print(X_u_train.shape)
    #X_f_train = np.vstack((X_f_train, X_u_train))
    #if typen == 'N_f':
    #    X_u_train2 = []
    #    u_train2 = []
    #if typen == 'N_u' : 
    #    X_f_train2 = []
    with open('training_chosen_data.p', 'rb') as fp:
        chosen_data = pickle.load(fp)
    X_u_train = np.array(chosen_data[typen]['X_u_train'])
    X_f_train = np.array(chosen_data[typen]['X_f_train'])
    #X_f_train2 = chosen_data[typen]['X_f_train2']
    X_f_train = np.vstack((X_f_train, X_f_train2))
    u_train = np.array(chosen_data[typen]['u_train'])
    return NNInputs_burgers(X_u_train, u_train, X_f_train, X_u_train2, u_train2, X_f_train2, lb, ub, nu, X_star, x, t, Exact)
