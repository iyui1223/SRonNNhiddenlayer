#!/bin/bash
# creates n-body simulation data output.
# a conda environment has to be loaded to run
# 
# conda activate final 
# module load gcc/11.3.0
# sh -eox 1_creat_n-body_simulation_data.sh


# User-defined variables
Root="/home/yi260/final_project"
Work="Work"
LogDir="${Root}/Log"
DataDir="${Root}/Data"
SourceDir="${Root}/Source"
sim="charge"
n=4
dim=2
nt=50
ns=100

# Timestamp for log files
timestamp=$(date +"%Y%m%d_%H%M%S")

mkdir -p "${Root}/${Work}" "${LogDir}"

cd "${Root}/${Work}"

ln -sf "${SourceDir}/simulate.py" .
ln -sf "${SourceDir}/make_simulation_data.py" .

# Log file setup
LogFile="${LogDir}/simulation_${sim}_n${n}_dim${dim}_nt${nt}_${timestamp}.log"
ErrFile="${LogDir}/simulation_${sim}_n${n}_dim${dim}_nt${nt}_${timestamp}.err"

{
    echo "[$(date)] Starting simulation: ${sim}, n=${n}, dim=${dim}, nt=${nt}, ns=${ns}"
    conda activate final 
    module load gcc/11.3.0

    python "make_simulation_data.py" --sim ${sim} --n ${n} --dim ${dim} --nt ${nt} --ns ${ns} >> "${LogFile}" 2>> "${ErrFile}"
    # Check for simulation output
    if ls *.npz 1> /dev/null 2>&1; then
        mv *.npz "${DataDir}/"
        echo "[$(date)] Simulation data moved to ${DataDir}/" >> "${LogFile}"
    else
        echo "[$(date)] ERROR: No .npz files found. Simulation might have failed."
    fi

    echo "[$(date)] Simulation completed."

} >> "${LogFile}" 2>> "${ErrFile}"

# Cleanup: Remove temporary working directory
cd "${Root}"
rm -rf "${Root}/${Work}"

echo "[$(date)] Temporary work directory removed."
echo "Log saved to: ${LogFile}"
