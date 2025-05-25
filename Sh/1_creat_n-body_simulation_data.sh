#!/bin/bash
#SBATCH --job-name=n-body-simulation
#SBATCH --output=./Log/1output.log
#SBATCH --error=./Log/1error.log
#SBATCH --time=00:07:00               # Max execution time (HH:MM:SS)
#SBATCH --mem=4G
#SBATCH --partition=icelake
#SBATCH -A MPHIL-DIS-SL2-CPU

##################
#### editable ####
##################

set -eox

# Source constants from the work directory
source "./const.txt"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    exit 1
fi

# Source conda
source "/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh"

# Initialize conda and activate environment
conda init bash
if ! conda activate final; then
    echo "Error: Failed to activate conda environment 'final'"
    exit 1
fi

# Load gcc module
if ! module load gcc/11.3.0; then
    echo "Error: Failed to load gcc/11.3.0 module"
    exit 1
fi

# Create necessary directories
mkdir -p "${ROOT_DIR}/${WORK_DIR}" "${LOG_DIR}" "${DATA_DIR}"

# Link source files
ln -sf "${SOURCE_DIR}/simulate.py" .
ln -sf "${SOURCE_DIR}/make_simulation_data.py" .

# Log file setup
timestamp=$(date +%Y%m%d_%H%M%S)
LogFile="${LOG_DIR}/simulation_${DATA_NAME}_${timestamp}.log"
ErrFile="${LOG_DIR}/simulation_${DATA_NAME}_${timestamp}.err"

echo "[$(date)] Starting simulation: ${SIM_TYPE_FINAL}, n=${N_BODIES}, dim=${DIMENSIONS}, nt=${NUM_TIMESTEPS}, ns=${NUM_SAMPLES}, dt=${DT}"
echo "[$(date)] Python version: $(python --version)"
echo "[$(date)] JAX version: $(python -c 'import jax; print(jax.__version__)')"

# Run simulation
python "make_simulation_data.py" \
    --sim ${SIM_TYPE_FINAL} \
    --n ${N_BODIES} \
    --dim ${DIMENSIONS} \
    --nt ${NUM_TIMESTEPS} \
    --ns ${NUM_SAMPLES}
# Check for simulation output and move to final location
if ls *.npz 1> /dev/null 2>&1; then
    mv nbody_simulation.npz "${DATA_DIR}/${DATA_NAME}.npz"
    echo "[$(date)] Simulation data moved to ${DATA_DIR}/"
else
    echo "[$(date)] ERROR: No .npz files found. Simulation might have failed."
    exit 1
fi

echo "[$(date)] Simulation completed successfully."

# Cleanup: Remove temporary working directory
cd "${ROOT_DIR}"
# do not remove temporary dir for debugging
# rm -rf "${ROOT_DIR}/${WORK_DIR}"
# echo "[$(date)] Temporary work directory removed."
