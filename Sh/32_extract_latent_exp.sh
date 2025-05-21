#!/bin/bash
#SBATCH --job-name=latent_exp
#SBATCH --output=/home/yi260/final_project/Log/32output.log
#SBATCH --error=/home/yi260/final_project/Log/32error.log
#SBATCH --time=00:15:00               # Max execution time (HH:MM:SS)
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
mkdir -p "${ROOT_DIR}/${WORK_DIR}" "${LOG_DIR}"

cd "${ROOT_DIR}/${WORK_DIR}"

ln -sf "${SOURCE_DIR}/"*.py "${ROOT_DIR}/${WORK_DIR}/"

echo "Running batch prediction process..."

MODEL_PATH="${MODELS_DIR}/${DATA_NAME/.npz/}/${MODEL_NAME}"
DATA_PATH="${DATA_DIR}/${DATA_NAME}.npz"

python visualize_hidden.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH"

# Move outputs to final location
if ls *.png 1> /dev/null 2>&1; then
    mkdir -p "${ROOT_DIR}/Figures/${DATA_NAME/.npz/}"
    mv *.png "${ROOT_DIR}/Figures/${DATA_NAME/.npz/}/"
    echo "[$(date)] Figures moved to ${ROOT_DIR}/Figures/${DATA_NAME/.npz/}/"
fi

if ls *.txt 1> /dev/null 2>&1; then
    mkdir -p "${ROOT_DIR}/Results/${DATA_NAME/.npz/}"
    mv *.txt "${ROOT_DIR}/Results/${DATA_NAME/.npz/}/"
    echo "[$(date)] Results moved to ${ROOT_DIR}/Results/${DATA_NAME/.npz/}/"
fi

echo "[$(date)] Analysis completed successfully."

