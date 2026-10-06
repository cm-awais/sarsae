# Source this before anything else:  source env.sh
export SARSAE_DATA=/home/chmuhammadawais/cvpr_earth_vision/data/classificiation
export SARSAE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SARSAE_WORK=$SARSAE_ROOT/work
export HF_HOME=$SARSAE_WORK/hf_cache            # model weights cached once, shared by all jobs
export TORCH_HOME=$SARSAE_WORK/torch_hub
export KAGGLEHUB_CACHE=$SARSAE_WORK/kagglehub
export PYTHONPATH=$SARSAE_ROOT:$PYTHONPATH
export PYTHONWARNINGS=ignore::FutureWarning,ignore::UserWarning,ignore::DeprecationWarning
