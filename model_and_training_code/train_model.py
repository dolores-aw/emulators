import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import interiorburgershp, argparser_raissi, plotting, burgers_raissihp
import scipy.io
import pickle
def main():
    N_u = 50
    N_f = 9
    typen = 'interior'
    trialn = 0
    args_parser = argparser_raissi.Parser()
    args = args_parser.parse_args_verified()
    layers = [2, 100, 100, 100, 100, 2]
    burgers_layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]
    input_seed = 1234


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
    inputs = interiorburgershp.prepare_nn_inputs_burgers('burgers_shock.mat', N_u, N_f,  random_seed=input_seed, debugging=False)
    u_input = inputs.u_train
    #model = burgersraissi.PhysicsInformedNN(inputs.X_u_train, u_input, inputs.X_f_train, burgers_layers, inputs.lb, inputs.ub, inputs.nu, inputs.X_star, N_u, N_f)


    errors = []
    for k in range(0,15):
        #declaring, training model
        model = burgers_raissihp.PhysicsInformedNN(inputs.X_u_train,  inputs.u_train, inputs.X_f_train, burgers_layers, lb, ub, inputs.nu, X_star, N_u, N_f)
        start_time = time.time()
        #if N_f > 0:
        #model.load_weights_and_biases('wab/weights_and_biases_%s_%s_%s_%s.npz' % (N_u, N_f - 1, typen, args.epochs))
        losses = model.train(args.epochs, args.data_loc, N_u, N_f, args.base_plot_dir)
        print('Training time: %.4f' % (time.time() - start_time))
        print(losses)

        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        pd.Series(losses).plot(logy=True, ax=ax)
        lpl = os.path.expanduser(args.lossplot_loc)
        plt.savefig(lpl)
        print("saved loss plot to {}".format(lpl))

        u_pred, f_pred = model.predict(inputs.X_star)  # X_star = tf.convert_to_tensor(X_star) ?
        print("u_pred_min:", u_pred.min())
        print("u_pred_max:", u_pred.max())
        error = np.linalg.norm(u-u_pred, 2)/25000
        errors.append(error)

        # saving      weights and biases
        weights_and_biases_path = 'wab/weights_and_biases_%s_%s_%s.npz'  % (N_u, N_f, args.epochs)
        #weights_and_biases_path = os.path.join(os.path.expanduser(args.save_loc), 'wab/weights_and_biases_%s_%s_%s_%s.npz' % (N_u, N_f, typen, args.epochs))
        print('saving weights and biases to {}'.format(weights_and_biases_path))
        model.save_weights_and_biases(weights_and_biases_path)
    
        #             solution
        solution_dict = {'sol' : u_pred, 'data' : inputs.X_u_train, 'f_sample' : inputs.X_f_train }
        scipy.io.savemat('sols/solution_data_%s_%s.mat' % (N_u, N_f), solution_dict)


if __name__ == '__main__':
    main()
