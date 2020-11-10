import os
import time
import skopt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import interiorburgersN_f, argparser_raissi, plotting, burgersraissilambda
import scipy.io
import pickle
def main(): #first use find_param, select_param to identify needed hyperparameter from front of MSE_F
    N_u = 50 # num. of total points in S_u
    N_f = 10 # 2^N_f num. of total points in S_f
    N_f2 = 500 # num. of points of S_f to choose within m of x = 0
    m  = 0.02 # distance from x = 0 to select certain points
    N_u2 = 25
    typen = 'N_f'
    args_parser = argparser_raissi.Parser()
    args = args_parser.parse_args_verified()
    
    layers = [2, 100, 100, 100, 100, 2]
    burgers_layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    
    with open('param_%s_%s_%s_%s_%s.p' % (N_u, N_f, N_u2, N_f2, typen), 'rb') as fp: #list of hyperparams
       temp_d = pickle.load(fp)
    lam = temp_d[m] 

    #preparing actual solution u
    data = scipy.io.loadmat('burgers_shock.mat')
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    X, T = np.meshgrid(x,t)
    udata = np.real(data['usol'])
    u = udata.T.flatten()[:,None]
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    ub = X_star.max()
    lb = X_star.min()

    typen = 'N_f'
    errors = []
    for k in range(30,50):
       #declaring, training model
        inputs = interiorburgersN_f.prepare_nn_inputs_burgers('burgers_shock.mat', N_u, N_f, N_u2, N_f2, m, k, debugging=False)
        model = burgersraissilambda.PhysicsInformedNN(lam, inputs.X_u_train,  inputs.u_train, inputs.X_f_train, burgers_layers, lb, ub, inputs.nu, X_star, N_u, N_f, N_u2, N_f2, m, typen)
        start_time = time.time()
        
        losses = model.train(args.epochs, args.data_loc, N_u, N_f, N_u2, N_f2, m, k, args.base_plot_dir)

        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        pd.Series(losses).plot(logy=True, ax=ax)
        lpl = os.path.expanduser(args.lossplot_loc)
        plt.savefig(lpl)
        
        #predicting solution and calculating error
        u_pred, f_pred = model.predict(inputs.X_star) 
        error = np.linalg.norm(u-u_pred, 2)/25000
        errors.append(error)


    
        #saving solution data - visualize using plot_prediction_N_f.ipynb
        solution_dict = {'sol' : u_pred, 'u_data' : inputs.X_u_train, 'u_data2' : inputs.X_u_train2, 'f_data' : inputs.X_f_train, 'f_data2' : inputs.X_f_train2 }
        
        scipy.io.savemat('chosen_data/sols/solution_data_%s_%s_%s_%s.mat' % (m, N_f2, typen, error), solution_dict)
        errors.append(error)

    if typen == 'N_u' :
        N_f2 = 0
    if typen == 'N_f' :
        N_u2 = 0
    with open('data_comparison.p', 'rb') as fp:
        d = pickle.load(fp)
    if typen in d.keys():
        dictionary = d[typen]
    else:
        dictionary = {}
    dictionary[(m,N_f2)] = errors
    d[typen] = dictionary
    with open('data_comparison.p', 'wb') as fp:
        pickle.dump(d, fp, protocol=2)
    

if __name__ == '__main__':
    main()
