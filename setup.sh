if [ "$(uname)" == "Darwin" ]; then
  echo "running on a mac. No locking."
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
  exec 100>/var/tmp/setup.lock || exit 1
  flock -w 300 100 || exit 1
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
  echo "Windows?"
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
  echo "Windows?"
fi
mkdir -p ~/slurmSetupLogs/
SLURM_SETUP_LOG=~/slurmSetupLogs/$(date +"%Y_%m_%d_%I_%M_%p")_`hostname`.log
echo "STARTING SLURM SETUP on $(date), logging to $SLURM_SETUP_LOG"
echo "STARTING SLURM SETUP on $(date)" >>$SLURM_SETUP_LOG
. ~/scripts/coresetup.sh >>$SLURM_SETUP_LOG 2>&1
. ~/scripts/coresetup.sh >>$SLURM_SETUP_LOG 2>&1
echo "FINISHED SLURM SETUP on $(date) with rc=$?"
echo "FINISHED SLURM SETUP on $(date) with rc=$?" >>$SLURM_SETUP_LOG
