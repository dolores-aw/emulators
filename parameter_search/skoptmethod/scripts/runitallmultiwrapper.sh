SPART_IN_USE=${SPART:-willett-gpu}
if [ -z ${SPART+x} ]; then echo "SPART is unset, will use $SPART_IN_USE"; else echo "SPART is set to '$SPART'"; fi
trap "unset SPART_IN_USE" EXIT
mkdir -p ~/junk/runItAllMultiWrapperLogs/
DATE_TOKEN=$(date +"%Y_%m_%d_%H_%M_%S")
RIAM_LOG=~/junk/runItAllMultiWrapperLogs/$DATE_TOKEN.log
WHAT_YOU_ARE_DOING="`date`: running  sbatch ~/scripts/runitallmulti.sh $RIAM_LOG $DATE_TOKEN $@"
META_LOG_FILE=~/junk/runItAllMultiWrapperLogs/meta.log
echo $WHAT_YOU_ARE_DOING
echo $WHAT_YOU_ARE_DOING >> $META_LOG_FILE
echo "log file for: $WHAT_YOU_ARE_DOING" > $RIAM_LOG
echo "tail -f $RIAM_LOG"
SBATCH_RUNDIR=~/junk/sbatch_runs
mkdir -p $SBATCH_RUNDIR
JOB_STRING=$(cd $SBATCH_RUNDIR; sbatch -p $SPART_IN_USE ~/scripts/runitallmulti.sh $RIAM_LOG $DATE_TOKEN $@ 2>&1)
SBATCH_RC=$?
SUMMARY="`date`: $JOB_STRING (sbatch -p $SPART_IN_USE rc=$SBATCH_RC)"
echo $SUMMARY
echo $SUMMARY >> $RIAM_LOG
echo $SUMMARY >> $META_LOG_FILE
echo "for command history: less $META_LOG_FILE, for sbatch runs: $SBATCH_RUNDIR"
return $SBATCH_RC
