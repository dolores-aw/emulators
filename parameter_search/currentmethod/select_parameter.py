import pickle
import os

N_u = 50
N_f = 10
N_u2 = 25
N_f2 = 1
m = 0.5
typen = 'N_u'
if os.path.isfile('param_dict_%s_%s_%s_%s_%s_%s.p' % (N_u, N_f, N_u2, N_f2, m, typen)):
    with open('param_dict_%s_%s_%s_%s_%s_%s.p' % (N_u, N_f, N_u2, N_f2, m ,typen), 'rb') as fp:
        d = pickle.load(fp)
else:
    d = {}
best = 1
best_ave = 1
for key in d.keys():
    l = d[key]
    ave = sum(l)/len(l)
    if ave < best_ave:
        best = key
        best_ave = ave

with open('param_%s_%s_%s_%s_%s.p' % (N_u, N_f, N_u2, N_f2, typen), 'rb') as fp:
    temp_d = pickle.load(fp)
temp_d[m] = best

with open('param_%s_%s_%s_%s_%s.p' % (N_u, N_f, N_u2, N_f2, typen), 'wb') as fp:
    pickle.dump(temp_d, fp, protocol=2)
