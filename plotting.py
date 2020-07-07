#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 20:11:57 2017
@author: mraissi
"""
import os

import numpy as np
import matplotlib as mpl
from pyDOE import lhs
from scipy.interpolate import griddata
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io

#nd mpl.use('pgf')
import model_and_data


def figsize(scale, nplots=1):
    fig_width_pt = 390.0  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = nplots * fig_width * golden_mean  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


USE_TEX = False
pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": USE_TEX,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 10,  # LaTeX default is 10pt font.
    "font.size": 10,
    "legend.fontsize": 8,  # Make the legend/label fonts a little smaller
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.figsize": figsize(1.0),  # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    ]
}
mpl.rcParams.update(pgf_with_latex)

import matplotlib.pyplot as plt


# I make my own newfig and savefig functions
def newfig(width, nplots=1):
    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename, crop=True):
    if crop:
        plt.savefig('{}.pdf'.format(filename), bbox_inches='tight', pad_inches=0)
        plt.savefig('{}.eps'.format(filename), bbox_inches='tight', pad_inches=0)
    else:
        plt.savefig('{}.pdf'.format(filename))
        plt.savefig('{}.eps'.format(filename))

def scatter(x_vals, y_vals, base_plt_dir, filename):
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.5)
    
    ax = plt.subplot(gs1[0, 0])
    ax.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
    ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')    
    ax.set_title('$t = 0.25$', fontsize = 10)
    file_targ = os.path.expanduser("{}/{}.eps".format(base_plt_dir, filename))
    plt.savefig(file_targ)
    print("created {}".format(file_targ))
    plt.close(fig)


def plotting(inputs, u_pred, base_plt_dir, title, filename="burgers"):
    print('Start of plotting:', inputs, u_pred.shape)
    N_f = 10000
    x_star = inputs.X_star
    t = inputs.t
    x = inputs.x
    X, T = np.meshgrid(x, t)
    Exact = inputs.exact
    u_star = Exact.flatten()[:, None]

    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    uu1 = Exact[0:1, :].T
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
    uu2 = Exact[:, 0:1]
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))
    uu3 = Exact[:, -1:]

    Xu_train = np.vstack([xx1, xx2, xx3])
    X_f_train = inputs.lb + (inputs.ub - inputs.lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, Xu_train))
    utrain = np.vstack([uu1, uu2, uu3])


    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)
    print('Error u: %e' % (error_u))

    U_pred = griddata(x_star, u_pred.flatten(), (X, T), method='cubic')
    Error = np.abs(Exact - U_pred)

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################

    fig, ax = newfig(1.0, 1.1)
    ax.axis('off')

    ####### Row 0: u(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(inputs.X_u_train[:, 1], inputs.X_u_train[:, 0], 'kx', label='Data (%d points)' % (inputs.u_train.shape[0]), markersize=4, clip_on=False)

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc='best')
    if title == 0:
        ax.set_title('$u(t,x)$', fontsize=10)
    if title == 1:
        ax.set_title('$u(t,x) - u_{pred}(t,x)$', fontsize=10)

    # ####### Row 1: u(t,x) slices ##################
    # gs1 = gridspec.GridSpec(1, 3)
    # gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    #
    # ax = plt.subplot(gs1[0, 0])
    # ax.plot(x, Exact[25, :], 'b-', linewidth=2, label='Exact')
    # ax.plot(x, U_pred[25, :], 'r--', linewidth=2, label='Prediction')
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$u(t,x)$')
    # ax.set_title('$t = 0.25$', fontsize=10)
    # ax.axis('square')
    # ax.set_xlim([-1.1, 1.1])
    # ax.set_ylim([-1.1, 1.1])
    #
    # ax = plt.subplot(gs1[0, 1])
    # ax.plot(x, Exact[50, :], 'b-', linewidth=2, label='Exact')
    # ax.plot(x, U_pred[50, :], 'r--', linewidth=2, label='Prediction')
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$u(t,x)$')
    # ax.axis('square')
    # ax.set_xlim([-1.1, 1.1])
    # ax.set_ylim([-1.1, 1.1])
    # ax.set_title('$t = 0.50$', fontsize=10)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
    #
    # ax = plt.subplot(gs1[0, 2])
    # ax.plot(x, Exact[75, :], 'b-', linewidth=2, label='Exact')
    # ax.plot(x, U_pred[75, :], 'r--', linewidth=2, label='Prediction')
    # ax.set_xlabel('$x$')
    # ax.set_ylabel('$u(t,x)$')
    # ax.axis('square')
    # ax.set_xlim([-1.1, 1.1])
    # ax.set_ylim([-1.1, 1.1])
    # ax.set_title('$t = 0.75$', fontsize=10)

    file_targ = os.path.expanduser("{}/{}.eps".format(base_plt_dir, filename))
    plt.savefig(file_targ)
    print("created {}".format(file_targ))
    plt.close(fig)


def main():
    nls_data_loc = '~/data/burgers_shock.mat'
    save_base_dir = '~/junk/eg_model'
    base_plt_dir = '~/tmp/plots'
    layers = [2, 100, 100, 100, 100, 2]

    inputs = model_and_data.prepare_nn_inputs(nls_data_loc, random_seed=1234, debugging=False)
    model = model_and_data.PhysicsInformedNN_burgers(inputs.x0, inputs.u0, inputs.tb, inputs.X_f, layers, inputs.lb, inputs.ub, inputs.X_star)
    model.load_weights_and_biases(os.path.join(save_base_dir, 'weights_and_biases.npz'))
    u_pred, f_pred = model.predict(inputs.X_star)
    plotting(inputs, u_pred, base_plt_dir, 'burgers')





if __name__ == '__main__':
    main()

## Simple plot
# fig, ax  = newfig(1.0)
#
# def ema(y, a):
#    s = []
#    s.append(y[0])
#    for t in range(1, len(y)):
#        s.append(a * y[t] + (1-a) * s[t-1])
#    return np.array(s)
#    
# y = [0]*200
# y.extend([20]*(1000-len(y)))
# s = ema(y, 0.01)
#
# ax.plot(s)
# ax.set_xlabel('X Label')
# ax.set_ylabel('EMA')
#
# savefig('ema')
