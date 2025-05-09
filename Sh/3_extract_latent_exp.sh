#!/bin/bash
#SBATCH --job-name=latent_exp
#SBATCH --output=/home/yi260/final_project/Log/3output.log
#SBATCH --error=/home/yi260/final_project/Log/3error.log
#SBATCH --time=00:10:00                # Max execution time (HH:MM:SS)
#SBATCH --partition=cclake             
#SBATCH -A MPHIL-DIS-SL2-CPU

##################
#### editable ####
##################

set -eox
# source ~/.bashrc
source "/usr/local/software/archive/linux-scientific7-x86_64/gcc-9/miniconda3-4.7.12.1-rmuek6r3f6p3v6fdj7o2klyzta3qhslh/etc/profile.d/conda.sh"

conda init bash
conda activate final; module load gcc/11.3.0

# Define directories
ROOT_DIR="/home/yi260/final_project/"
WORKDIR="${ROOT_DIR}/workdir"
SRC_DIR="${ROOT_DIR}/Source"
DATA_DIR="${ROOT_DIR}/Data"
LOG_DIR="${ROOT_DIR}/Log"

mkdir -p "$WORKDIR" 
cd "$WORKDIR"

echo "Setting up working directory..."

ln -sf "${SRC_DIR}/"*.py "$WORKDIR/"
# ln -sf "${DATA_DIR}/lotka_volterra_data.h5" "$WORKDIR/"
# ln -sf "${DATA_DIR}/predictions" "$WORKDIR/"

echo "Running batch prediction process..."

python visualize_hidden.py --model_path ../Models/nbody_h32_m16_b1_e11.pt --data_path ../Data/nbody_simulation.npz


echo "Organizing output files..."

echo "All files have been moved successfully."
