#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --output=/home/yi260/final_project/Log/25output.log
#SBATCH --error=/home/yi260/final_project/Log/25error.log
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

# Model parameters
HIDDEN_DIM=256          # Hidden dimension size
MSG_DIM=128            # Message dimension size
NUM_TIMESTEPS=125      # Number of timesteps to evaluate
DEVICE="cpu"          # Device to use (cpu/cuda)
MODEL_NAME="nbody_h${HIDDEN_DIM}_m${MSG_DIM}_b1_e11.pt"
DATA_NAME="spring-n4-dim2-nt250-ns10000"

# Define directories
ROOT_DIR="/home/yi260/final_project/"
WORKDIR="${ROOT_DIR}/workdir"
SRC_DIR="${ROOT_DIR}/Source"
DATA_DIR="${ROOT_DIR}/Data"
LOG_DIR="${ROOT_DIR}/Log"
MODELS_DIR="${ROOT_DIR}/Models"
OUTPUT_DIR="${ROOT_DIR}/Figs/evaluation_results"


mkdir -p "$WORKDIR" "$OUTPUT_DIR"
cd "$WORKDIR"

echo "Setting up working directory..."

ln -sf "${SRC_DIR}/"*.py "$WORKDIR/"

echo "Running batch prediction process..."

python evaluation.py \
    --model_path "${MODELS_DIR}/${DATA_NAME}/${MODEL_NAME}" \
    --data_path "${DATA_DIR}/${DATA_NAME}.npz" \
    --device "${DEVICE}" \
    --output_dir "${OUTPUT_DIR}" \
    --hidden_dim "${HIDDEN_DIM}" \
    --msg_dim "${MSG_DIM}" \
    --num_timesteps "${NUM_TIMESTEPS}"

echo "All files have been moved successfully."
