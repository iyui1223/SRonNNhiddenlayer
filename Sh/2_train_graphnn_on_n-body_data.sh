#!/bin/bash

set -eox

# User-defined variables
Root="/home/yi260/final_project"
Work="Work"
LogDir="${Root}/Log"
DataDir="${Root}/Data"
SourceDir="${Root}/Source"
ModelsDir="${Root}/Models"  # Add Models directory

# Model parameters
HIDDEN_DIM=32          # Hidden dimension size
MSG_DIM=16            # Message dimension size
EPOCHS=50             # Number of training epochs
BATCH_SIZE=1         # Batch size for training
LEARNING_RATE=0.001  # Learning rate
DEVICE="cpu"         # Device to use (cpu/cuda)
DATA_PATH="nbody_simulation.npz"  # Path to the simulation data

# Timestamp for log files
timestamp=$(date +"%Y%m%d_%H%M%S")

# Create necessary directories
mkdir -p "${Root}/${Work}" "${LogDir}" "${ModelsDir}"

cd "${Root}/${Work}"

ln -sf "${SourceDir}/train.py" .
ln -sf "${DataDir}/nbody_simulation.npz" .

# Initialize conda for bash
eval "$(conda shell.bash hook)"
conda activate final 
module load gcc/11.3.0

# Log file setup
LogFile="${LogDir}/train_graphnn_on_n-body_data.log"
ErrFile="${LogDir}/train_graphnn_on_n-body_data.err"

# Run training with logging
{
    python "train.py" \
        --hidden_dim ${HIDDEN_DIM} \
        --msg_dim ${MSG_DIM} \
        --epochs ${EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --learning_rate ${LEARNING_RATE} \
        --device ${DEVICE} \
        --data_path ${DATA_PATH} \
        --checkpoint_dir "${ModelsDir}"
    
    echo "[$(date)] Training completed."

} >> "${LogFile}" 2>> "${ErrFile}"

# Cleanup: Remove temporary working directory
cd "${Root}"
rm -rf "${Root}/${Work}"

echo "[$(date)] Temporary work directory removed."
echo "Log saved to: ${LogFile}"
echo "Model checkpoints saved in: ${ModelsDir}"
