import argparse
import os
import sys

import tools_py


class Parser(argparse.ArgumentParser):
    def __init__(self):
        super(Parser, self).__init__(description='arguments for Raissi ML test')
        # data
        self.add_argument('--data-loc', type=str, required=True, help='location of dataset')
        self.add_argument('--save-loc', type=str, required=True, help="location to save model")
        self.add_argument('--lossplot-loc', type=str, required=True, help="location to save loss-plot")
        self.add_argument('--base-plot-dir', type=str, required=True, help="location to save function plots")
        self.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
        #self.add_argument('--N_u', type = int, required=True, help = 'num of S_u points')
        #self.add_argument('--N_f', type=int, required=True, help = 'num of S_f points')
        # training

        # suggestions
        # lp_loc = '/tmp/loss_plot.eps'
        # save_base_dir = '/Users/danamendelson/junk/eg_model'
        # burgers_data_loc = '/Users/danamendelson/Desktop/burgers_shock.mat'

    def parse_args_verified(self):
        args = self.parse_args()

        print(' '.join(sys.argv))
        print('--data-loc', args.data_loc)
        print('--save-loc', args.save_loc)
        print('--lossplot-loc', args.lossplot_loc)
        print('--base-plot-dir', args.base_plot_dir)
        print('--epochs', args.epochs)
        print()

        assert args.epochs > 0, args.epochs
        tools_py.make_dirs(args.save_loc)
        tools_py.make_dirs(args.base_plot_dir)
        tools_py.make_dirs_for_file_name(args.lossplot_loc)
        assert os.path.isdir(os.path.expanduser(args.save_loc)), args.save_loc + ' does not exist'
        #assert args.data_loc.endswith('.mat'), args.data_loc + ' should end with .mat'
        #assert os.path.exists(os.path.expanduser(args.data_loc)), args.data_loc + ' does not exist'
        return args


def main():
    args_parser = Parser()
    args = args_parser.parse_args()
    print(args.data_loc)


if __name__ == '__main__':
    main()
