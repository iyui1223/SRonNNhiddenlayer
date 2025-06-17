#!/bin/bash
#SBATCH --job-name=evaluation
#SBATCH --output=./Log/31output.log
#SBATCH --error=./Log/31error.log
#SBATCH --time=00:07:00               # Max execution time (HH:MM:SS)
#SBATCH --partition=icelake
#SBATCH -A MPHIL-DIS-SL2-CPU

##################
#### editable ####
##################

set -eox

# Source constants from the work directory
source "./const.txt"

source ~/venvs/final_clean/bin/activate

# Cp source files
cp "${SOURCE_DIR}/evaluation.py" .
cp "${SOURCE_DIR}/train.py" .
cp "${SOURCE_DIR}/model_util/model_util_${MODEL_TYPE}.py" model_util.py

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
