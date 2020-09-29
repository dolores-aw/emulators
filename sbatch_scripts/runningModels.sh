#!/bin/bash

TRAINING_DIR=~/modelTrainLogs/
SAVE_DIR=~/junk/eg_model
DATA_DIR=~/data
PLOT_DIR=~/plots
TRAINING_LOG=$TRAINING_DIR/$(date +"%Y_%m_%d_%I_%M_%p")_$(hostname).log
EMULATOR_BASE_DIR=~

mkdir -p $TRAINING_DIR $SAVE_DIR $PLOT_DIR
echo "TRAINING MODEL on $(date), logging to ${TRAINING_LOG}"
echo "Running train_and_save_main!" >> $TRAINING_LOG
python $EMULATOR_BASE_DIR/train_model.py --save-loc $SAVE_DIR --data-loc $DATA_DIR/burgers_shock.mat --base-plot-dir $PLOT_DIR --lossplot-loc $PLOT_DIR/loss_plot.eps --epochs 10000  >> $TRAINING_LOG 2>&1
echo "train_and_save_main finished with return code $?." >> $TRAINING_LOG
