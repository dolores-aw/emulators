import pickle
import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt

m = 1
N_u2 = 25
N_f2 = 500
typen = 'N_f'
filenames = 'solution_data_%s_%s_%s_%s' % (m, N_u2, N_f2, typen)
file_list = []
space_errors = np.zeros((256,1))
#data = scipy.io.loadmat('sols/solution_data_%s_%s_%s_%s_%s.mat' % (m, N_u, N_u2, N_f2, typen, error))

dat = scipy.io.loadmat('burgers_shock.mat')
Exact = np.real(dat['usol']).T.flatten()[:,None]


def get_averages(file) :
    data = scipy.io.loadmat('./sols/' + file)
    preds_u = data['sol']
    
    dat = scipy.io.loadmat('burgers_shock.mat')
    Exact = np.real(dat['usol']).T.flatten()[:,None]
    t = dat['t'].flatten()[:,None]
    x=dat['x'].flatten()[:,None]
    X, T = np.meshgrid(x,t)
    
    abs_errors = np.abs(preds_u.flatten(), Exact.flatten())#Inputs.Exact.flatten())
    abs_errors = abs_errors.reshape((100,256))
    space_errors = abs_errors.sum(axis=0).reshape((256,1))
    x_coords = X[0].reshape((256,1))#np.real(dat['usol']).X[0].reshape((256,1))#Exact.X[0].reshape((256,1))
    return x_coords, space_errors#np.hstack((x_coords, space_errors))









for filename in os.listdir('sols'):
    if filename[0:len(filenames)] == 'solution_data_%s_%s_%s_%s' % (m, N_u2, N_f2, typen) :
        file_list.append(filename)

for file in file_list:
    x_coords, temp_space_errors = get_averages(file)
    space_errors = space_errors + temp_space_errors

space_errors = space_errors/len(file_list)   
space_errors = np.hstack((x_coords, space_errors))
plt.plot(space_errors[:, 0], space_errors[:, 1])
plt.ylabel('Sum of |error| across time')
plt.xlabel('X position')
#plt.vlines(-m, -1000, 1000, colors = 'orange', linestyles = 'dotted')
#plt.vlines(m, -1000, 1000, colors = 'orange', linestyles = 'dotted')
plt.axvline(x=m, linestyle = 'dotted')
plt.axvline(x=-m, linestyle = 'dotted')
plt.title('Ave. prediction errors at a given space coordinate, m = %s' % m)
plt.savefig('ave_space_errors_plot_%s_%s_%s_%s.png' % (m, N_u2, N_f2, typen))
plt.show()



