import pickle #p9 ; domain of resolvent function is completement of sigma(T)
import statistics
import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt #MSE between u and 0; if st dev is small, pick one solutiong to print
#parameter can vary as size of sets vary
N_f2 = 500
N_u2 = 25
typen = 'N_f'
data1 = {}
with open('data_comparison.p', 'rb') as fp:
    data = pickle.load(fp)
#with open('datacomparison1.p', 'rb') as fp:
#    data1 = pickle.load(fp)
Nu_data = {}
Nf_data = {}
print(type(Nf_data))
if 'N_u' in data.keys():
    Nu_data.update(data['N_u'])
if 'N_f' in data.keys():
    Nf_data.update(data['N_f'])
#if 'X_f' in data.keys():
#    Nf_data = Nf_data.update(data['X_f'])
if 'N_u' in data1.keys():
    print(data1['N_u'].keys())
    Nu_data.update(data1['N_u'])
if 'N_f' in data1.keys():
    Nf_data.update(data1['N_f'])
if 'X_f' in data1.keys():
    Nf_data.update(data1['X_f'])

burgers_data = scipy.io.loadmat('burgers_shock.mat')
t = burgers_data['t'].flatten()[:, None]
x = burgers_data['x'].flatten()[:, None]
Exact = np.real(burgers_data['usol']).T
u_star = Exact.flatten()[:,None]
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

print(Nu_data.keys())
#plot N_f
points = Nf_data.keys()
x = [] #will be given level of concentration m
y = [] #will be ave-error over entire domain
z = [] #will be ave. error in given m value
control = [] # will be ave. error of non-concentrated predictions in given m values
#z2 = [] # nonnormalized z
#control2 = [] # nonnormalized control

############# creating lists x, y
for point in points:
    p = point[1]
    if p == N_f2:
        val = sum(Nf_data[point])/len(Nf_data[point])
        x.append(point[0])
        y.append(val)
############# averaging control trials
filenames = 'solution_data_%s_%s_%s_%s' % (1, N_u2, N_f2, typen)
file_list = []
for filename in os.listdir('sols'):
    if filename[0:len(filenames)] == 'solution_data_%s_%s_%s_%s' % (1, N_u2, N_f2, typen) :
        file_list.append(filename)
error_list = []
#ave_control = scipy.io.loadmat('./sols/' + file_list[0])['sol']
#for file in file_list[1:]: #for every trial for the given m value m=1
#    data = scipy.io.loadmat('./sols/' + file)['sol']
#    ave_control = ave_control + data
#ave_control = ave_control/len(file_list)
for m in x:
    filenames = 'solution_data_%s_%s_%s_%s' % (1, N_u2, N_f2, typen)
    file_list = []
    for filename in os.listdir('sols'):
        if filename[0:len(filenames)] == 'solution_data_%s_%s_%s_%s' % (1, N_u2, N_f2, typen) :
            file_list.append(filename)
    error_list = []
    for file in file_list: #for every trial for the given m value 
        data = scipy.io.loadmat('./sols/' + file)['sol']
        X_star2 = []
        u_star2 = []
        #u_pred2 = []
        u_control2= []
        k = 0
        for point in X_star:
            k = k + 1
            if (point[0] < m) & (point[0] > -m) :
                X_star2.append(point)
                u_star2.append(u_star[k])
                u_control2.append(data[k])
                #u_control2.append(ave_control[k])
        u_star2 = np.array(u_star2)
        #X_star2 = np.array(X_star2)
        #u_pred2 = np.array(u_pred2)
        u_control2 = np.array(u_control2)
        error_list.append(np.linalg.norm(u_star2 - u_control2, 2)/len(X_star2))
    control.append(sum(error_list)/len(file_list))
    #control.append(np.linalg.norm(u_control2 - u_pred2,2)/len(X_star2))#(u_star2,2)/len(X_star2))
############# creating list z
for m in x:
    print(m)
    filenames = 'solution_data_%s_%s_%s_%s' % (m, N_u2, N_f2, typen)
    file_list = []
    for filename in os.listdir('sols'):
        if filename[0:len(filenames)] == 'solution_data_%s_%s_%s_%s' % (m, N_u2, N_f2, typen) :
            file_list.append(filename)
    error_list = []
    for file in file_list: #for every trial for the given m value 
        data = scipy.io.loadmat('./sols/' + file)['sol']
        X_star2 = []
        u_star2 = []
        u_pred2 = []
        #u_control2= []
        k = 0
        for point in X_star:
            k = k + 1
            if (point[0] < m) & (point[0] > -m) :
                X_star2.append(point)
                u_star2.append(u_star[k])
                u_pred2.append(data[k])
                #u_control2.append(ave_control[k])
        u_star2 = np.array(u_star2)
        #X_star2 = np.array(X_star2)
        #u_pred2 = np.array(u_pred2)
        #u_control2 = np.array(u_control2)
        error_list.append(np.linalg.norm(u_star2 - u_pred2, 2)/len(X_star2))
    z.append(sum(error_list)/len(file_list))
    #control.append(np.linalg.norm(u_control2 - u_pred2,2)/len(X_star2))#(u_star2,2)/len(X_star2))
    #z2.append(sum(error_list))
    #control2.append(np.linalg.norm(u_star2,2))
#setting up control



print(z)
plt.scatter(x,y, vmin=0, vmax = 0.0025, label = 'average')#
plt.scatter(x,z, c = 'red', label ='restricted average')
plt.scatter(x, control, c = 'green', label = 'control')
plt.figtext(0,0.02, '*restricted: norm restricted to points (x,t) with -m < |x| < m', fontsize = 'x-small')
plt.figtext(0, 0, '**control: predicted from uniform S_f, restricted to -m < |x| < m', fontsize= 'x-small')
plt.ylim([0, max(max(control),max(z))])
plt.xlim([0, 1.05])
plt.legend()
plt.title('Ave. Prediction Error with |S_f| = 2^10, 500 points near x = 0')
plt.xlabel('m')
plt.ylabel('Error')
#plt.colorbar('Error')

plt.savefig('Data_Comparison_Nf_full')
plt.show() 

print(z)
plt.scatter(x,y, vmin=0, vmax = 0.0025, label = 'prediction error (average)')#
plt.scatter(x,z, c = 'red', label ='restricted* prediction error (average)')
plt.scatter(x, control, c = 'green', label = 'restricted* control**')
plt.figtext(0,0.02, '*restricted: norm restricted to points (x,t) with -m < |x| < m', fontsize = 'x-small')
plt.figtext(0, 0, '**control: predicted from uniform S_f, restricted to -m < |x| < m', fontsize= 'x-small')
plt.ylim([0, 0.005])
plt.xlim([0, 1.05])
plt.legend()
plt.title('Ave. Prediction Error with |S_f| = 2^10, 500 points near x = 0')
plt.xlabel('X')
plt.ylabel('Error')
#plt.colorbar('Error')

plt.savefig('Data_Comparison_Nf')
plt.show() 

print(z)
#plt.scatter(x,y, vmin=0, vmax = 0.0025, label = 'average')#
#plt.scatter(x,z2, c = 'red', label ='restricted average')
#plt.scatter(x, control2, c = 'green', label = 'control')
#plt.ylim([0, 0.1])
#plt.xlim([0, 1.05])
#plt.legend()
#plt.title('Prediction error with 500 S_f points concentrated within X of x=0')
#plt.xlabel('X')
#plt.ylabel('Error')
#plt.colorbar('Error')

#plt.savefig('Data_Comparison_Nf_nonnormal')
#plt.show() 


#plot N_u
#points = Nu_data.keys()
#x = []
#y = []

#for point in points:
#    p = point[1]
#    #if p == N_u2:
#    val = sum(Nu_data[point])/len(Nu_data[point])
#    x.append(point[0])
#    y.append(val)

#plt.scatter(x,y)#
#plt.ylim([0, 0.0005])
#plt.title('Prediction error with 25 S_u points concentrated within X of x=0')
#plt.xlabel('X')
#plt.ylabel('Error')
#plt.colorbar('Error')

#plt.savefig('Data_Comparison_Nu')
#plt.show()

