#!/bin/bash
IAM_LOG=$1
DATE_TOKEN=$2
shift 2
EMULATOR_BASE_DIR=~
META_LOG=~/runItAllMultiWrapperLogs/meta.log
trap "echo 'script done' >> $RIAM_LOG" EXIT
echo "You might want to tail -f $RIAM_LOG"

date >>$RIAM_LOG
. $EMULATOR_BASE_DIR/scripts/setup.sh >>$RIAM_LOG
echo Training the model... >>$RIAM_LOG

DATA_DIR=~
SAVE_DIR=~/runs/$DATE_TOKEN/model
PLOT_DIR=~/runs/$DATE_TOKEN/plots
EMULATOR_BASE_DIR=~
mkdir -p $SAVE_DIR $PLOT_DIR
echo "TRAINING MODEL on $(date), logging to ${RIAM_LOG}, with arguments $@"
PYTHON_MAIN=$EMULATOR_BASE_DIR/train_lambda.py
echo "Running $PYTHON_MAIN" >>$RIAM_LOG
echo "$(date): Starting main within sbatch ~/dev/emulator/scripts/runItAllMulti.sh $RIAM_LOG $DATE_TOKEN $@" >> $META_LOG
python $PYTHON_MAIN --save-loc $SAVE_DIR --data-loc $DATA_DIR/burgers_shock.mat --base-plot-dir $PLOT_DIR --lossplot-loc $PLOT_DIR/loss_plot.eps $@ >>$RIAM_LOG 2>&1
RC=$?
echo "train_and_save_main finished with return code $RC." >>$RIAM_LOG
WHAT_YOU_ARE_DOING="$(date): rc=$RC from sbatch ~/dev/emulator/scripts/runItAllMulti.sh $RIAM_LOG $DATE_TOKEN $@"
echo $WHAT_YOU_ARE_DOING
echo $WHAT_YOU_ARE_DOING >> $META_LOG

date >>$RIAM_LOG
