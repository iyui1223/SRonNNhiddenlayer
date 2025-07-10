#!/bin/bash
#SBATCH --job-name=app_sr
#SBATCH --output=./Log/6output.log
#SBATCH --error=./Log/6error.log
#SBATCH --time=00:15:00               # Max execution time (HH:MM:SS)
#SBATCH --partition=icelake             
#SBATCH -A MPHIL-DIS-SL2-CPU

##################
#### editable ####
##################

set -eox

# Load GCC 11.3.0 for newer GLIBCXX
module load gcc/11.3.0/gcc/4zpip55j

# Source constants from the work directory
source "./const.txt"

# Set up Julia environment
#export JULIA_BIN=~/julia_pysr/bin/julia
#export PATH=~/julia_pysr/bin:$PATH
#export JULIA_DEPOT_PATH=~/julia_pysr/share/julia
#export JULIA_PROJECT=~/julia_pysr/share/julia/environments/v1.9
#export LD_LIBRARY_PATH=~/julia_pysr/lib:$LD_LIBRARY_PATH

# Activate Python environment
source ${SOURCE_ENV_PYSR}

# Create necessary directories
mkdir -p "${ROOT_DIR}/${WORK_DIR}" "${LOG_DIR}"

# Link source files
cp "${SOURCE_DIR}/model_util/model_util_${MODEL_TYPE}.py" model_util.py
cp "${SOURCE_DIR}/apply_sr.py" .

MODEL_PATH="${MODELS_DIR}/${DATA_NAME/.npz/}/${MODEL_NAME}"
DATA_PATH="${DATA_DIR}/${DATA_NAME}.npz"

# Check for previous run's data
PREV_DATA_PATH="${DATA_DIR}/forSR/${DATA_NAME/.npz/}/${MODEL_NAME/.pt/}.npz"
if [ -f "$PREV_DATA_PATH" ]; then
    echo "Found previous run's data at $PREV_DATA_PATH"
    cp "$PREV_DATA_PATH" "model_data_prev.npz"
    PREV_DATA="model_data_prev.npz"
else
    PREV_DATA="None"
fi

echo "Running latent space analysis..."

# Verify Julia setup
echo "Julia binary location: $(which julia)"
echo "Julia version: $($JULIA_BIN --version)"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

python apply_sr.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --ndim "${DIMENSIONS}" \
    --dt "${DT}" \
    --prev_data "${PREV_DATA}"

# Save new data if it exists
if [ -f "model_data.npz" ]; then
    mkdir -p "${DATA_DIR}/forSR/${DATA_NAME/.npz/}"
    cp "model_data.npz" "${DATA_DIR}/forSR/${DATA_NAME/.npz/}/${MODEL_NAME/.pt/}.npz"
fi

# Move outputs to final location
if ls *.png 1> /dev/null 2>&1; then
    EVAL_DIR="${ROOT_DIR}/Figs/${DATA_NAME%.npz}/${MODEL_NAME%.pt}"
    mkdir -p "$EVAL_DIR"
    mv *.png "$EVAL_DIR"
    mv *.txt "$EVAL_DIR"
    echo "[$(date)] Figures moved to $EVAL_DIR"
fi

echo "[$(date)] Analysis completed successfully."
