import os
import time
import skopt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import interiorburgerslambda, argparser_raissi, plotting, burgersraissilambda
import scipy.io
import pickle
from skopt.callbacks import DeadlineStopper
def main():
    N_u = 50
    N_f = 10
    N_u2 = 25
    N_f2 = 500
    typen = 'N_f'
    trialn = 0
    m = .1
    lambdas = [0.00001,0.00005, 0.0001,0.0005, 0.001, 0.005, 0.01]
    #lam = lambdas[0]
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
    #with open('train_d.p', 'rb') as fp:
    #    inputs = pickle.load(fp)[trialn]
    #X_u_train = inputs.X_u_train[:N_u]
    #X_f_train = inputs.X_f_train[:2**N_f]
    #u_train = inputs.u_train[:N_u]
    inputs = interiorburgerslambda.prepare_nn_inputs_burgers('burgers_shock.mat', N_u, N_f, N_u2, N_f2, m, typen, debugging=False)
    u_input = inputs.u_train
    #model = burgersraissi.PhysicsInformedNN(inputs.X_u_train, u_input, inputs.X_f_train, burgers_layers, inputs.lb, inputs.ub, inputs.nu, inputs.X_star, N_u, N_f)


    errors = []
    def function(lam):
        #declaring, training model
        model = burgersraissilambda.PhysicsInformedNN(lam, inputs.X_u_train,  inputs.u_train, inputs.X_f_train, burgers_layers, lb, ub, inputs.nu, X_star, N_u, N_f, N_u2, N_f2, m, typen)
        start_time = time.time()
        #if N_f > 0:
        #model.load_weights_and_biases('wab/weights_and_biases_%s_%s_%s_%s.npz' % (N_u, N_f - 1, typen, args.epochs))
        losses = model.train(args.epochs, args.data_loc, N_u, N_f, 
N_u2, N_f2, m, typen, args.base_plot_dir)
        #print('Training time: %.4f' % (time.time() - start_time))
        #print(losses)

        #plt.close()
        #fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        #pd.Series(losses).plot(logy=True, ax=ax)
        #lpl = os.path.expanduser(args.lossplot_loc)
        #plt.savefig(lpl)
        #print("saved loss plot to {}".format(lpl))

        u_pred, f_pred = model.predict(inputs.X_star)  # X_star = tf.convert_to_tensor(X_star) ?
        error = np.linalg.norm(u-u_pred, 2)/25000
        return error
    #start = time()
    #t = (0,0.1, "prior")
    print(type(t))
    l = skopt.gp_minimize(function, [(.01, 1.1)], callback = DeadlineStopper(259200))
    print(type(l))
    print(l)
    with open('final_lambda.p', 'wb') as fp:
        pickle.dump({'lambda': l}, fp, protocol=2)


if __name__ == '__main__':
    main()
