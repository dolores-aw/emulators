import pickle
import tensorflow as tf
import scipy.io
import numpy as np
import os

import burgersraissilambda, interiorburgerslambda, train_find_parameter


N_u = 50
N_f = 10
N_u2 = 25
N_f2 = 500
typen = 'N_f'
m = 0.1
ntrials = 20

lam_list = [0.25]#[0.01, 0.1, 0.5, 0.25, 1, 1.25, 1.5, 1.1, 1.01]

if not os.path.isfile('param_dict_%s_%s_%s_%s_%s.p' % (N_u,N_f,N_u,N_f2,typen)):
    with open('param_dict_%s_%s_%s_%s_%s.p' % (N_u,N_f,N_u2,N_f2,typen), 'wb') as fp:
        pickle.dump({}, fp,protocol=2)

for lam in lam_list:
    errors = []
    for trial in range(0,ntrials):
        error = train_find_parameter.training(N_u, N_f, N_u2, N_f2, typen, m, lam)
        errors.append(error)
    with open('param_dict_%s_%s_%s_%s_%s.p'% (N_u,N_f,N_u2,N_f2,typen), 'rb') as fp:
        d = pickle.load(fp)
    d[lam] = errors
    with open('param_dict_%s_%s_%s_%s_%s.p' % (N_u,N_f,N_u2,N_f2,typen), 'wb') as fp:
        pickle.dump(d, fp, protocol=2)

comparison_dict = {}
with open('param_dict_%s_%s_%s_%s_%s.p' % (N_u,N_f,N_u2,N_f2,typen), 'rb') as fp:
    d = pickle.load(fp)
for key in d.keys():
    l = d[key]
    comparison_dict[key] = sum(l)/len(l)
l = list(comparison_dict.keys())
print(type(l))
print(l)
best_lambda = l[0]
for key in comparison_dict.keys():
    if comparison_dict[key] < comparison_dict[best_lambda]:
        best_lambda = key
print(best_lambda)
print(comparison_dict[best_lambda])




