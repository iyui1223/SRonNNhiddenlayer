#!/bin/bash
#SBATCH --job-name=n-body
#SBATCH --output=/home/yi260/final_project/Log/1output.log
#SBATCH --error=/home/yi260/final_project/Log/1error.log
#SBATCH --time=00:15:00               # Max execution time (HH:MM:SS)
#SBATCH --partition=icelake
#SBATCH -A MPHIL-DIS-SL2-CPU

##################
#### editable ####
##################

set -eox

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

# User-defined variables
Root="/home/yi260/final_project"
Work="Work"
LogDir="${Root}/Log"
DataDir="${Root}/Data"
SourceDir="${Root}/Source"
sim="spring"  # Changed to spring simulation
n=4
dim=2
nt=250
ns=10000

# Timestamp for log files
timestamp=$(date +"%Y%m%d_%H%M%S")

# Create necessary directories
mkdir -p "${Root}/${Work}" "${LogDir}" "${DataDir}"

cd "${Root}/${Work}"

# Check if source files exist
if [ ! -f "${SourceDir}/simulate.py" ] || [ ! -f "${SourceDir}/make_simulation_data.py" ]; then
    echo "Error: Required source files not found"
    exit 1
fi

ln -sf "${SourceDir}/simulate.py" .
ln -sf "${SourceDir}/make_simulation_data.py" .

# Log file setup
LogFile="${LogDir}/simulation_${sim}_n${n}_dim${dim}_nt${nt}_${timestamp}.log"
ErrFile="${LogDir}/simulation_${sim}_n${n}_dim${dim}_nt${nt}_${timestamp}.err"

echo "[$(date)] Starting simulation: ${sim}, n=${n}, dim=${dim}, nt=${nt}, ns=${ns}"
echo "[$(date)] Python version: $(python --version)"
echo "[$(date)] JAX version: $(python -c 'import jax; print(jax.__version__)')"

# Run simulation
python "make_simulation_data.py" --sim ${sim} --n ${n} --dim ${dim} --nt ${nt} --ns ${ns} >> "${LogFile}" 2>> "${ErrFile}"

# Check for simulation output
if ls *.npz 1> /dev/null 2>&1; then
    mv nbody_simulation.npz "${DataDir}/${sim}-n${n}-dim${dim}-nt${nt}-ns${ns}.npz"
    echo "[$(date)] Simulation data moved to ${DataDir}/"
else
    echo "[$(date)] ERROR: No .npz files found. Simulation might have failed."
    exit 1
fi

echo "[$(date)] Simulation completed successfully."

# Cleanup: Remove temporary working directory
cd "${Root}"
# rm -rf "${Root}/${Work}"

echo "[$(date)] Temporary work directory removed."
