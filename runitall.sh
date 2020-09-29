#!/bin/bash
JUNK_DIR=~/junk
EMULATOR_BASE_DIR=~

echoerr() { echo "$@" 1>&2; }
echoerr "tail -f $JUNK_DIR/runitall.log"

mkdir -p $JUNK_DIR
date >> $JUNK_DIR/runitall.log
. $EMULATOR_BASE_DIR/scripts/setup.sh >> $JUNK_DIR/runitall.log
echo Training the model... >> $JUNK_DIR/runitall.log
. $EMULATOR_BASE_DIR/scripts/runningModels.sh >> $JUNK_DIR/runitall.log
date >>	$JUNK_DIR/runitall.log