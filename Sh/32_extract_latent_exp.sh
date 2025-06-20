#!/bin/bash
#SBATCH --job-name=latent_exp
#SBATCH --output=./Log/32output.log
#SBATCH --error=./Log/32error.log
#SBATCH --time=06:00:00               # Max execution time (HH:MM:SS)
#SBATCH --partition=icelake             
#SBATCH -A MPHIL-DIS-SL2-CPU

##################
#### editable ####
##################

set -eox

# Source constants from the work directory
source "./const.txt"

source ~/venvs/final_clean/bin/activate

# Create necessary directories
mkdir -p "${ROOT_DIR}/${WORK_DIR}" "${LOG_DIR}"

# Link source files
cp "${SOURCE_DIR}/train.py" .
cp "${SOURCE_DIR}/model_util/model_util_${MODEL_TYPE}.py" model_util.py
cp "${SOURCE_DIR}/visualize_hidden.py" .

echo "Running latent space analysis..."

# @@@debug
# MODEL_PATH="${MODELS_DIR}/${DATA_NAME/.npz/}/nbody_h256_m128_b16_e1.pt"
MODEL_PATH="${MODELS_DIR}/${DATA_NAME/.npz/}/${MODEL_NAME}"
DATA_PATH="${DATA_DIR}/${DATA_NAME}.npz"

python visualize_hidden.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --ndim "${DIMENSIONS}" \
    --dt "${DT}"

# Move outputs to final location
if ls *.png 1> /dev/null 2>&1; then
    EVAL_DIR="${ROOT_DIR}/Figs/${DATA_NAME%.npz}/${MODEL_NAME%.pt}"
    mkdir -p "$EVAL_DIR"
    mv *.png "$EVAL_DIR"
    mv *.txt "$EVAL_DIR"
    echo "[$(date)] Figures moved to $EVAL_DIR"
fi

echo "[$(date)] Analysis completed successfully."

