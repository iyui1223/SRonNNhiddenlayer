#!/bin/bash

# Set and export root directory
export ROOT_DIR="/home/yi260/final_project"

# Define forcing types
FORCE_TYPES=("spring" "charge" "damped" "disc" "r1" "r2")
# FORCE_TYPES=("spring" "damped")

# define model or loss types
MODEL_TYPES=("standard" "bottleneck" "L1" "KL")
#MODEL_TYPES=("L1")

# Function to create work directory and setup for a forcing type
setup_force_type() {
    local force_type=$1
    local model_type=$2
    local work_dir="${ROOT_DIR}/SR_Work/${force_type}_${model_type}"
    
    # Create work directory
    mkdir -p "$work_dir"
    
    # Copy template const.txt to work directory
    cp "${ROOT_DIR}/Const/const_template.txt" "${work_dir}/const.txt"
    
    # Modify parameters in the copied const.txt
    sed -i "s|WORK_DIR=.*|WORK_DIR=\"Work/${force_type}_${model_type}\"|" "${work_dir}/const.txt"
    sed -i "s|SIM_TYPE=.*|SIM_TYPE=\"${force_type}\"|" "${work_dir}/const.txt"
    sed -i "s|MODEL_TYPE=.*|MODEL_TYPE=\"${model_type}\"|" "${work_dir}/const.txt"

    echo "$work_dir"
}

# Function to submit a job and get its job ID
submit_job() {
    local script=$1
    local dependency=$2
    local work_dir=$3
    
    if [ -z "$dependency" ]; then
        # Submit job without dependency
        job_id=$(sbatch --parsable --chdir="$work_dir" "$script")
    else
        # Submit job with dependency
        job_id=$(sbatch --parsable --dependency=afterok:$dependency --chdir="$work_dir" "$script")
    fi
    
    echo $job_id
}

# Function to run pipeline for a single forcing type
run_pipeline_for_type() {
    local force_type=$1
    local work_dir=$(setup_force_type "$force_type" "$model_type")
    
    echo "Starting pipeline execution for $force_type..."
    job6=$(submit_job "${ROOT_DIR}/Sh/6_applySR.sh" "" "$work_dir")
} 

# Run pipeline for each forcing and model type combinations
for force_type in "${FORCE_TYPES[@]}"; do
    for model_type in "${MODEL_TYPES[@]}"; do
        run_pipeline_for_type "$force_type" "$model_type"
    done
done

echo "All pipelines submitted successfully!"
echo "You can monitor the jobs using: squeue -u $USER" 
