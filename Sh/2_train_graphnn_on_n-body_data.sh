#!/bin/bash
#SBATCH --job-name=train_gnn
#SBATCH --output=./Log/2output.log
#SBATCH --error=./Log/2error.log
#SBATCH --time=02:00:00               # Max execution time (HH:MM:SS)
#SBATCH --partition=sapphire
#SBATCH -A MPHIL-DIS-SL2-CPU

##################
#### editable ####
##################

set -eox

# Source constants from the work directory
source "./const.txt"

source ${SOURCE_ENV}

# Create necessary directories
mkdir -p "${ROOT_DIR}/${WORK_DIR}" "${LOG_DIR}" "${MODELS_DIR}/${DATA_NAME/.npz/}"

# Link source files
cp "${SOURCE_DIR}/train.py" .
cp "${SOURCE_DIR}/model_util/model_util_${MODEL_TYPE}.py" model_util.py
ln -rsf "${DATA_DIR}/${DATA_NAME}.npz" .

# Initialize conda for bash
eval "$(conda shell.bash hook)"
conda activate final 
module load gcc/11.3.0

# Run training with logging
python "train.py" \
    --model_type ${MODEL_TYPE} \
    --hidden_dim ${HIDDEN_DIM} \
    --msg_dim ${MSG_DIM} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --dt ${DT} \
    --device ${DEVICE} \
    --data_path "${DATA_NAME}.npz" \
    --checkpoint_dir "${MODELS_DIR}/${DATA_NAME/.npz/}"

echo "[$(date)] Training completed."
echo "Model checkpoints saved in: ${MODELS_DIR}/${DATA_NAME/.npz/}"
