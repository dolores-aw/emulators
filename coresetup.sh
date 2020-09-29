ENV_NAME= tf-gpu1.15_env
SCRATCH_PREFIX=/scratch/dolores
CONDA_EXECUTABLE=$SCRATCH_PREFIX/miniconda3/bin/conda
ENV_PYTHON_EXECUTABLE=$SCRATCH_PREFIX/miniconda3/envs/$ENV_NAME/bin/python
PACKAGES_NON_TF="pandas matplotlib"
TFLOW_VERSION=1.15.0

if [ "$(uname)" == "Darwin" ]; then
  echo "running on a mac. No locking."
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
  echo "running on linux..."
  exec 100>/var/tmp/${USER}setup.lock || exit 1
  flock -w 300 100 || exit 1
  echo "got the lock."
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW32_NT" ]; then
  echo "Windows?"
elif [ "$(expr substr $(uname -s) 1 10)" == "MINGW64_NT" ]; then
  echo "Windows?"
fi

if [ -f "$CONDA_EXECUTABLE" ]; then
  echo "$CONDA_EXECUTABLE exists. Yay."
else
  echo "$CONDA_EXECUTABLE does not exist. Installing miniconda3..."
  if [ -f "~/Miniconda3-latest-Linux-x86_64.sh" ]; then
    echo "already got Miniconda3-latest-Linux-x86_64.sh"
  else
    cd ~
    echo "downloading Miniconda3-latest-Linux-x86_64.sh to ~"
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  fi
  bash ~/Miniconda3-latest-Linux-x86_64.sh -b -p $SCRATCH_PREFIX/miniconda3
fi
echo "running conda init (?)"
eval "$($SCRATCH_PREFIX/miniconda3/bin/conda 'shell.bash' 'hook')"
if [ -L ${ENV_PYTHON_EXECUTABLE} ] && [ -e ${ENV_PYTHON_EXECUTABLE} ]; then
  echo "$ENV_PYTHON_EXECUTABLE exists."
else
  echo "$ENV_PYTHON_EXECUTABLE does not exist. Creating env $ENV_NAME..."
  conda create -n $ENV_NAME -y
fi
echo "activating $ENV_NAME"
conda activate $ENV_NAME

if [[ $(hostname) =~ ^gpu.* ]]; then
  echo "You're on a gpu box: $(hostname)"
  export TFLOW="tensorflow-gpu"
else
  echo "You're on a cpu box: $(hostname)"
  export TFLOW="tensorflow"
fi
echo "installing: '$TFLOW $PACKAGES_NON_TF'..."
conda install $TFLOW=$TFLOW_VERSION $PACKAGES_NON_TF -y
conda install pyDOE -c conda-forge -y
conda install -c conda-forge scikit-optimize
