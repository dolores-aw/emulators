import os
import time
import skopt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import interiorburgerslambda, argparser_raissi, plotting, burgersraissilambda
import scipy.io
import pickle
def training(inputs, N_u, N_f,N_u2,N_f2, typen, m, lam):
    #N_u = 50 # num. of total points in S_u
    #N_f = 10 # num. of total points in S_f
    #N_u2 = 25 # num. of points of S_u to choose within m of x  = 0
    #N_f2 = 500 # num. of points of S_f to choose within m of x = 0
    #typen = 'N_u' #or 'N_u' or 'both'; for now, keep N_f
    #m  = .03 # distance from x = 0 to select certain points
    #if typen == 'N_u' :
    #%    N_f2 = 0
    #if typen == 'N_f' :
    #    N_u2 = 0
    #lambdas = [0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.01]
    #lam = lambdas[0]
    args_parser = argparser_raissi.Parser()
    args = args_parser.parse_args_verified()
    layers = [2, 100, 100, 100, 100, 2]
    burgers_layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    #N_f = 2**N_f - N_f2
    #lam = 0.00001

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


    #preparing training/input data
    #with open('train_d.p', 'rb') as fp:
    #    inputs = pickle.load(fp)[trialn]
    #X_u_train = inputs.X_u_train[:N_u]
    #X_f_train = inputs.X_f_train[:2**N_f]
    #u_train = inputs.u_train[:N_u]
    #inputs = interiorburgerslambda.prepare_nn_inputs_burgers('burgers_shock.mat', N_u, N_f, N_f2, m,  random_seed=input_seed, debugging=False)
    #u_input = inputs.u_train
    #model = burgersraissi.PhysicsInformedNN(inputs.X_u_train, u_input, inputs.X_f_train, burgers_layers, inputs.lb, inputs.ub, inputs.nu, inputs.X_star, N_u, N_f)


    #errors = []
    for k in range(0,1):
       #declaring, training model
        #inputs = interiorburgerslambda.prepare_nn_inputs_burgers('burgers_shock.mat', N_u, N_f, N_u2, N_f2, m, typen, debugging=False)
        print(inputs.X_f_train)
        model = burgersraissilambda.PhysicsInformedNN(lam, inputs.X_u_train,  inputs.u_train, inputs.X_f_train, burgers_layers, lb, ub, inputs.nu, X_star, N_u, N_f, N_u2, N_f2, m, typen)
        start_time = time.time()
        #if N_f > 0:
        #model.load_weights_and_biases('wab/weights_and_biases_%s_%s_%s_%s.npz' % (N_u, N_f - 1, typen, args.epochs))
        losses = model.train(args.epochs, args.data_loc, N_u, N_f, N_u2, N_f2, m, typen, args.base_plot_dir)
        #print('Training time: %.4f' % (time.time() - start_time))
        #print(losses)

        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        pd.Series(losses).plot(logy=True, ax=ax)
        lpl = os.path.expanduser(args.lossplot_loc)
        plt.savefig(lpl)
        #print("saved loss plot to {}".format(lpl))

        u_pred, f_pred = model.predict(inputs.X_star)  # X_star = tf.convert_to_tensor(X_star) ?
        #print("u_pred_min:", u_pred.min())
        #print("u_pred_max:", u_pred.max())
        error = np.linalg.norm(u-u_pred, 2)/25000
        #errors.append(error)
    return error
        # saving      weights and biases
        #weights_and_biases_path = 'wab/weights_and_biases_hp_%s.npz' % lam# % (N_u, N_f, typen, args.epochs)
        #weights_and_biases_path = os.path.join(os.path.expanduser(args.save_loc), 'wab/weights_and_biases_%s_%s_%s_%s.npz' % (N_u, N_f, typen, args.epochs))
        #print('saving weights and biases to {}'.format(weights_and_biases_path))
        #model.save_weights_and_biases(weights_and_biases_path)
    
        #             solution
        #solution_dict = {'sol' : u_pred, 'u_data' : inputs.X_u_train, 'u_data2' : inputs.X_u_train2, 'f_data' : inputs.X_f_train, 'f_data2' : inputs.X_f_train2 }
        #if typen == 'both':
        #scipy.io.savemat('chosen_data/sols/solution_data_%s_%s_%s_%s_%s.mat' % (m, N_u2, N_f2, typen, error), solution_dict)
        #if typen == 'N_u' :
        #scipy.
        #errors.append(error)
        #             loss, weights, biases
     #   with open('epochs_v_error_hp.p', 'rb') as fp:
     #       d = pickle.load(fp)
     #   with open('dat/epochs_v_error_lambda%s.p' % lam, 'wb') as fp:
     #       pickle.dump(d, fp, protocol=2)
     #   with open('epochs_v_error_hp.p', 'wb') as fp:
     #       pickle.dump(d, fp, protocol=2)
     #   with open('wab_dict_hp.p', 'rb') as fp:
     #       d = pickle.load(fp)
     #   with open('dat/wab_dict_lambda%s.p' % lam, 'wb') as fp:
     #       pickle.dump(d, fp, protocol=2)
     #   with open('wab_dict_hp.p', 'wb') as fp:
     #       pickle.dump({}, fp, protocol=2)
     #   with open('b_dict_hp.p', 'rb') as fp:
     #       d = pickle.load(fp)
     #   with open('dat/b_dict_lambda%s.p' % lam, 'wb') as fp:
     #       pickle.dump(d, fp, protocol=2)
     #   with open('b_dict_hp.p', 'wb') as fp:
     #       pickle.dump({}, fp, protocol=2)

    #with open('lambda_comparison.p', 'rb') as fp:
    #    d = pickle.load(fp)
    #d[lam] = errors
    #with open('lambda_comparison.p', 'wb') as fp:
    #    pickle.dump(d, fp, protocol=2)
    
    #t = (0,0.1, "prior")
    #print(type(t))
    #l = skopt.gp_minimize(function, [(0, 0.1)])
    #print(l)
    #with open('final_lambda.p', 'wb') as fp:
    #    pickle.dump({'lambda': l}, fp, protocl=2)
    #print(errors)
    #if typen == 'N_u' :
    #    N_f2 = 0
    #if typen == 'N_f' :
    #    N_u2 = 0
    #with open('datacomparison.p', 'rb') as fp:
    #    d = pickle.load(fp)
    #if typen in d.keys():
    #    dictionary = d[typen]
    #else:
    #    dictionary = {}
    #if typen == 'N_f' :
    #    dictionary[(m, N_f2)] = errors
    #elif typen == 'N_u' :
    #    dictionary[(m, N_u2)] = errors
    #elif typen == 'both' :
    #    dictionary[(m, N_u2, N_f2)] = errors
    #d[typen] = dictionary
    #with open('datacomparison.p', 'wb') as fp:
    #    pickle.dump(d, fp, protocol=2)
    

#if __name__ == '__main__':
#    main()
