#!/bin/bash
#SBATCH --job-name=train_gnn
#SBATCH --output=./Log/2output.log
#SBATCH --error=./Log/2error.log
#SBATCH --time=01:00:00               # Max execution time (HH:MM:SS)
#SBATCH --partition=icelake
#SBATCH -A MPHIL-DIS-SL2-CPU

##################
#### editable ####
##################

set -eox

# Source constants from the work directory
source "./const.txt"

# Source conda
source "/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh"

conda init bash
conda activate final; module load gcc/11.3.0
# module load cuda/12.4


# Create necessary directories
mkdir -p "${ROOT_DIR}/${WORK_DIR}" "${LOG_DIR}" "${MODELS_DIR}/${DATA_NAME/.npz/}"

# Link source files
cp "${SOURCE_DIR}/train.py" .
cp "${SOURCE_DIR}/model_util_${MODEL_TYPE}.py" model_util.py
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
