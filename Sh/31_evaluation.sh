#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --output=/home/yi260/final_project/Log/31output.log
#SBATCH --error=/home/yi260/final_project/Log/31error.log
#SBATCH --time=00:15:00               # Max execution time (HH:MM:SS)
#SBATCH --partition=icelake
#SBATCH -A MPHIL-DIS-SL2-CPU

##################
#### editable ####
##################

set -eox
# source ~/.bashrc
source "/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh"

conda init bash
conda activate final; module load gcc/11.3.0

# Source constants
source "${ROOT_DIR}/Const/const.txt"

# Model parameters
# @@@make it refer the same parameter sets with other wrapper shells at ../Const/const.txt
DEVICE="cpu"          # Device to use (cpu/cuda)

# Define directories
WORKDIR="${ROOT_DIR}/workdir"
SRC_DIR="${ROOT_DIR}/Source"
DATA_DIR="${ROOT_DIR}/Data"
LOG_DIR="${ROOT_DIR}/Log"
MODELS_DIR="${ROOT_DIR}/Models"
# Create evaluation results directory in the model directory
EVAL_DIR="${MODELS_DIR}/${DATA_NAME}/evaluation"
mkdir -p "$EVAL_DIR"

mkdir -p "$WORKDIR"
cd "$WORKDIR"


MODEL_NAME="nbody_h256_m128_b16_e1.pt"


echo "Setting up working directory..."

ln -sf "${SRC_DIR}/"*.py "$WORKDIR/"

echo "Running batch prediction process..."

python evaluation.py \
    --model_path "${MODELS_DIR}/${DATA_NAME}/${MODEL_NAME}" \
    --data_path "${DATA_DIR}/${DATA_NAME}.npz" \
    --device "${DEVICE}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --msg_dim "${MSG_DIM}" \
    --num_timesteps "125" \ # no more than 125 steps
    --dt "${DT}" \
    --ndim "${DIMENSIONS}" \
    --save_path "${EVAL_DIR}/trajectory_plot.png"

echo "All files have been moved successfully."
