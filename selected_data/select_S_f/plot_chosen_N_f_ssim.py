import pickle #p9 ; domain of resolvent function is completement of sigma(T)
import statistics
import scipy.io
import numpy as np
import os
import math
#import tensorflow as tf
from skimage.metrics import structural_similarity as skssim
from scipy.stats import ttest_ind
from scipy.special import stdtr
import matplotlib.pyplot as plt #MSE between u and 0; if st dev is small, pick one solutiong to print
#parameter can vary as size of sets vary
N_f2 = 500
N_f = 13
N_u2 = 25 
typen = 'N_f'
data1 = {}
restrictionx = 1 #dont change; will keep for now in case I decide to implement this feature later
restrictiont = 0
direct = 'initbound/'
with open('data_comparison_true.p', 'rb') as fp:
    data = pickle.load(fp)



burgers_data = scipy.io.loadmat('burgers_shock.mat')
t = burgers_data['t'].flatten()[:, None]
x = burgers_data['x'].flatten()[:, None]
Exact = np.real(burgers_data['usol']).T
u_star = Exact.flatten()[:,None]
X, T = np.meshgrid(x, t)
X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

#plot N_f
#plots data lists

x = [] #will be given level of concentration m
y = [] #will be ave-error over entire domain
z = [] #will be ave. error in given m value
stdevs = []
control = [] # will be ave. error of non-concentrated predictions in given m values
controlstdevs = []
ssim_controlstdevs = []
pvals = []
ssim_pvals = []
ssim_control = []
ssim_list = []
ssim_stdevs = []

def ssim(usol, u_pred):
    minimum = np.abs(np.min([usol.min(), u_pred.min(), 0]))
    usol = usol + minimum
    u_pred = u_pred + minimum
    return skssim(usol, u_pred, data_range = u_pred.max() - u_pred.min(), multichannel=True)


    
############# creating lists x, y

points = [ 0.05, 0.1, 0.25, 0.2, 0.35, 0.3, 0.4, 0.5, 0.6, 0.75, 1] #the m-values (usually in dictionary Nf_data)
maxy = 0

#for point in points:
#    p = point[1]
#    if p == N_f2:
#        val = sum(Nf_data[point])/len(Nf_data[point])
#        x.append(point[0])
#        y.append(val)

x = points

############# averaging control trials
filenames = 'solution_data_%s_%s_%s_%s' % (1, N_f, N_f2, typen)
file_list = []
for filename in os.listdir(direct):
    if filename[0:len(filenames)] == 'solution_data_%s_%s_%s_%s' % (1, N_f, N_f2, typen) :
        file_list.append(filename)
error_list = []
#creating the 'control' : the data where m = 1, used to compare when m is other values
for m in x:
    filenames = 'solution_data_%s_%s_%s_%s' % (1, N_f, N_f2, typen)
    file_list = []
    for filename in os.listdir(direct):
        if filename[0:len(filenames)] == 'solution_data_%s_%s_%s_%s' % (1, N_f, N_f2, typen) :
            file_list.append(filename)
    control_error_list = []
    control_ssim_list = []
    for file in file_list: #for every trial for the given m value 
        data = scipy.io.loadmat('./%s' % direct + file)['sol']
        X_star2 = []
        u_star2 = []
        #u_pred2 = []
        u_control2= []
        k = -1
        for point in X_star:
            k = k + 1
            if (point[0] <= restrictionx) & (point[0] >= -restrictionx) & (point[1] >= restrictiont) :
                X_star2.append(point)
                u_star2.append(u_star[k])
                u_control2.append(data[k])
        u_star2 = np.array(u_star2)
        u_control2 = np.array(u_control2)
        control_error_list.append(np.linalg.norm(u_star2 - u_control2, 2)/np.linalg.norm(u_star2, 2))
        control_ssim_list.append(ssim(Exact, u_control2.reshape(Exact.shape)))
        #control_ssim_list.append(ssim(u_star2, u_control2, data_range = u_control2.max() - u_control2.min(), multichannel=True))
    control.append(sum(control_error_list)/len(file_list))
    ssim_control.append(sum(control_ssim_list)/len(control_ssim_list))
    maxy = max(maxy, max(control_error_list))
    controlstdevs.append(np.std(control_error_list))
    print(control_ssim_list)
    ssim_controlstdevs.append(np.std(control_ssim_list))
    


############# creating list z = mean square prediction error
############# creating list ssim_list = ssim prediction error
############# and pvalues for both
for m in x:
    print(m)
    filenames = 'solution_data_%s_%s_%s_%s' % (m, N_f, N_f2, typen)
    file_list = []
    for filename in os.listdir(direct):
        if filename[0:len(filenames)] == 'solution_data_%s_%s_%s_%s' % (m, N_f, N_f2, typen) :
            file_list.append(filename)
    error_list = []
    ssim_error_list = []
    for file in file_list: #for every trial for the given m value 
        data = scipy.io.loadmat('./%s' % direct + file)['sol']
        X_star2 = []
        u_star2 = []
        u_pred2 = []
        #u_control2= []
        k = -1
        for point in X_star:
            k = k + 1
            if (point[0] <= restrictionx) & (point[0] >= -restrictionx) & (point[1] >= restrictiont) :
                X_star2.append(point)
                u_star2.append(u_star[k])
                u_pred2.append(data[k])
                #u_control2.append(ave_control[k])
        u_star2 = np.array(u_star2)
        #X_star2 = np.array(X_star2)
        #u_pred2 = np.array(u_pred2)
        #u_control2 = np.array(u_control2)
        error_list.append(np.linalg.norm(u_star2 - u_pred2, 2)/np.linalg.norm(u_star2))
        u_star2 = np.array(u_star2)
        u_pred2 = np.array(u_pred2)
        ssim_error_list.append(ssim(Exact, u_pred2.reshape(Exact.shape)))
        #ssim_error_list.append(ssim(u_star2, u_pred2, data_range = max(u_pred2) - min(u_pred2), multichannel = True))
    z.append(sum(error_list)/len(file_list))
    stdevs.append(np.std(error_list))
    ssim_list.append(sum(ssim_error_list)/len(ssim_error_list))
    t, p = ttest_ind(error_list, control_error_list, equal_var=False)
    pvals.append(p)
    maxy = max(maxy, max(error_list))
    t, p = ttest_ind(ssim_list, control_ssim_list, equal_var=False)
    ssim_pvals.append(p)
    ssim_stdevs.append(np.std(ssim_error_list))

maxy = maxy + 0.25

#plotting the m v relative error plots + error bars
print(control)
#plt.scatter(x,y, vmin=0, vmax = 0.0025, label = 'average')#
plt.errorbar(x,z,yerr=stdevs, fmt = 'o', alpha = 0.5)
plt.scatter(x,z, c = 'blue', label ='restricted average')
plt.scatter(x, control, c = 'red', label = 'control')
plt.errorbar(x, control, ecolor = 'red', yerr=controlstdevs, alpha = 0.5, fmt = 'o')
#plt.scatter(x, pvals, c = 'green', label = 'p values')
sign = 1
for j in range(len(x)):
    xval = x[j]
    yval = maxy
    #plt.text(xval, yval + 0.05*sign, round(pvals[j], 4))
    plt.text(xval, z[j] + 0.01, round(pvals[j], 3))
    sign = sign*(-1)
plt.figtext(0, 0.02, 'ave test st.dev: %s, control st. dev : %s' % (round(sum(stdevs)/len(stdevs),4), round(sum(controlstdevs)/len(controlstdevs), 4)), fontsize= 'x-small')
#plt.ylim([0, 2*max(max(control),max(z))])
plt.xlim([0, 1.05])
plt.legend()
plt.title('|S_f| = 2^%s, 500 points near x = 0' % N_f)
plt.xlabel('m')
plt.ylabel('Relative Error')


plt.savefig('./compare_plots/Data_Comparison_Nf_%s_%s_%s.png' % (direct.replace('/', '.'), restrictionx, restrictiont))
plt.show() 

# plotting using structural similarity error
plt.errorbar(x,ssim_list,yerr=ssim_stdevs, fmt = 'o', alpha = 0.5)
plt.scatter(x,ssim_list, c = 'blue', label ='average ssim')
plt.scatter(x, ssim_control, c = 'red', label = 'control')
plt.errorbar(x, ssim_control, ecolor = 'red', yerr=ssim_controlstdevs, alpha = 0.5, fmt = 'o')
#plt.scatter(x, ssim_pvals, c = 'green', label = 'p values')
plt.figtext(0, 0.02, 'ave test st.dev: %s, control st. dev : %s' % (round(sum(stdevs)/len(stdevs),4), round(sum(controlstdevs)/len(controlstdevs), 4)), fontsize= 'x-small')
plt.xlim([0, 1.05])
plt.legend()
plt.title('SSIM error |S_f| = 2^%s, 500 points near x = 0' % N_f)
plt.xlabel('m')
plt.ylabel('SSIM')

plt.savefig('./compare_plots/SSIM_Data_Comparison_Nf_%s_%s_%s.png' % (direct.replace('/', '.'), restrictionx, restrictiont))
plt.show() 
#try repeating with increased |S_f| (say 2^12?), try doing multiple experiments with fixed m and varying
#|S_f2| values
#look up https://en.wikipedia.org/wiki/Structural_similarity
#fix a relatively small patch, choose S_f concentrated within that patch, and force emulator to match
#PDE exactly in that patch - look at performance. Addressed Q of local-nonlocality of NN
#plt.savefig('./compare_plots/ptest_plots_Nf_%s_%s_%s.png' % (direct.replace('/', '.'), restrictionx, restrictiont))