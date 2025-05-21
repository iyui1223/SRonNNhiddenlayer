#!/bin/bash
#SBATCH --job-name=train_gnn
#SBATCH --output=/home/yi260/final_project/Log/2output.log
#SBATCH --error=/home/yi260/final_project/Log/2error.log
#SBATCH --time=01:30:00               # Max execution time (HH:MM:SS)
#SBATCH --partition=icelake
#SBATCH -A MPHIL-DIS-SL2-CPU

##################
#### editable ####
##################

set -eox

# Source constants
source "${ROOT_DIR}/Const/const.txt"

# Source conda
source "/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh"

conda init bash
conda activate final; module load gcc/11.3.0

# Create necessary directories
mkdir -p "${ROOT_DIR}/${WORK_DIR}" "${LOG_DIR}" "${MODELS_DIR}/${DATA_NAME/.npz/}"

cd "${ROOT_DIR}/${WORK_DIR}"

ln -sf "${SOURCE_DIR}/train.py" .
ln -sf "${DATA_DIR}/${DATA_NAME}.npz" .

# Initialize conda for bash
eval "$(conda shell.bash hook)"
conda activate final 
module load gcc/11.3.0

# Run training with logging
python "train.py" \
    --hidden_dim ${HIDDEN_DIM} \
    --msg_dim ${MSG_DIM} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --learning_rate ${LEARNING_RATE} \
    --device ${DEVICE} \
    --data_path "${DATA_NAME}.npz" \
    --checkpoint_dir "${MODELS_DIR}/${DATA_NAME/.npz/}"

echo "[$(date)] Training completed."

# Cleanup: Remove temporary working directory
cd "${ROOT_DIR}"
# rm -rf "${ROOT_DIR}/${WORK_DIR}"

echo "[$(date)] Temporary work directory removed."
echo "Model checkpoints saved in: ${MODELS_DIR}/${DATA_NAME/.npz/}"
