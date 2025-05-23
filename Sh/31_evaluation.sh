#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --output=/home/yi260/final_project/Log/31output.log
#SBATCH --error=/home/yi260/final_project/Log/31error.log
#SBATCH --time=00:07:00               # Max execution time (HH:MM:SS)
#SBATCH --partition=icelake
#SBATCH -A MPHIL-DIS-SL2-CPU

##################
#### editable ####
##################

set -eox

# Source constants from the work directory
source "const.txt"

# Source conda
source "/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh"

conda init bash
conda activate final; module load gcc/11.3.0

# Link source files
ln -sf "${SOURCE_DIR}/"*.py .

# @@@debug
MODEL_NAME="nbody_h256_m128_b16_e1.pt"

# Create evaluation results directory in the model directory
EVAL_DIR="${ROOT_DIR}/Figs/${DATA_NAME%.npz}/${MODEL_NAME%.pt}"
mkdir -p "$EVAL_DIR"

echo "Running batch prediction process..."

python evaluation.py \
    --model_path "${MODELS_DIR}/${DATA_NAME}/${MODEL_NAME}" \
    --data_path "${DATA_DIR}/${DATA_NAME}.npz" \
    --device "cpu" \
    --hidden_dim "${HIDDEN_DIM}" \
    --msg_dim "${MSG_DIM}" \
    --num_timesteps ${NUM_TIMESTEPS} \
    --dt "${DT}" \
    --ndim "${DIMENSIONS}" \
    --save_path "${EVAL_DIR}/trajectory_plot.png"

echo "Evaluation completed successfully."
