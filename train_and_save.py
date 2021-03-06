import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import model_and_data, argparser_raissi, plotting, burgers_raissi
import scipy.io

def main():
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


    inputs = model_and_data.prepare_nn_inputs_burgers('burgers_shock.mat', random_seed=input_seed, debugging=False)

    #adding noise
    scale = .01
    n_noise_levels = 6 
    for k in range(3, n_noise_levels):
        noise = k*scale
        u_input = inputs.u_train
        for index in range(0,u_input.size):
            u_input[index] = u_input[index] + noise*np.random.randn(1)
        model = burgers_raissi.PhysicsInformedNN(inputs.X_u_train, u_input, inputs.X_f_train, burgers_layers, inputs.lb, inputs.ub, inputs.nu, inputs.X_star)

        start_time = time.time()
        losses = model.train(args.epochs, args.data_loc, args.base_plot_dir)
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

        #solution_error = u - u_pred

        #plotting.plotting(inputs, u_pred, args.base_plot_dir, 0, 'final_burgers.solution.%s'  %  noise)
        #plotting.plotting(inputs, solution_error.flatten(), args.base_plot_dir, 1, 'final_burgers_difference.%s'  %  noise)
    # u_star = inputs.exact.flatten()[:, None]
    # plotting.plotting(inputs, u_star, args.base_plot_dir, 'burgers_exact')

        weights_and_biases_path = os.path.join(os.path.expanduser(args.save_loc), 'weights_and_biases_2.npz')
        print('saving weights and biases to {}'.format(weights_and_biases_path))
        model.save_weights_and_biases(weights_and_biases_path)

        solution_dict = {'sol' : u_pred, 'data' : inputs.X_u_train }

        scipy.io.savemat('solution_data%s.mat'   %  noise, solution_dict)
if __name__ == '__main__':
    main()
