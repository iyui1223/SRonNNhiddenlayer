#!/bin/bash
#SBATCH --job-name=n-body
#SBATCH --output=/home/yi260/final_project/Log/2output.log
#SBATCH --error=/home/yi260/final_project/Log/2error.log
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

# User-defined variables
Root="/home/yi260/final_project"
Work="Work"
LogDir="${Root}/Log"
DataDir="${Root}/Data"
SourceDir="${Root}/Source"

# Model parameters
HIDDEN_DIM=32          # Hidden dimension size
MSG_DIM=16            # Message dimension size
EPOCHS=100             # Number of training epochs
BATCH_SIZE=1         # Batch size for training
LEARNING_RATE=0.001  # Learning rate
DEVICE="cpu"         # Device to use (cpu/cuda)
DATA_NAME="spring-n4-dim2-nt250-ns10000.npz"  # Path to the simulation data
ModelsDir="${Root}/Models/${DATA_NAME/.npz/}"

# Timestamp for log files
timestamp=$(date +"%Y%m%d_%H%M%S")

# Create necessary directories
mkdir -p "${Root}/${Work}" "${LogDir}" "${ModelsDir}"

cd "${Root}/${Work}"

ln -sf "${SourceDir}/train.py" .
ln -sf "${DataDir}/${DATA_NAME}" .

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
    --data_path ${DATA_NAME} \
    --checkpoint_dir "${ModelsDir}"

echo "[$(date)] Training completed."

# Cleanup: Remove temporary working directory
cd "${Root}"
# rm -rf "${Root}/${Work}"

echo "[$(date)] Temporary work directory removed."
echo "Log saved to: ${LogFile}"
echo "Model checkpoints saved in: ${ModelsDir}"
