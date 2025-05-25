#!/bin/bash

# Set and export root directory
export ROOT_DIR="/home/yi260/final_project"

# Define forcing types
# FORCE_TYPES=("spring" "charge" "damped" "disc")
FORCE_TYPES=("r1" "r2")
# Function to create work directory and setup for a forcing type
setup_force_type() {
    local force_type=$1
    local work_dir="${ROOT_DIR}/Work/${force_type}_nbody"
    
    # Create work directory
    mkdir -p "$work_dir"
    mkdir -p "${work_dir}/Log"
    
    # Copy template const.txt to work directory
    cp "${ROOT_DIR}/Const/const_template.txt" "${work_dir}/const.txt"
    
    # Modify parameters in the copied const.txt
    sed -i "s|WORK_DIR=.*|WORK_DIR=\"Work/${force_type}_nbody\"|" "${work_dir}/const.txt"
    sed -i "s|SIM_TYPE=.*|SIM_TYPE=\"${force_type}\"|" "${work_dir}/const.txt"
    
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
    local work_dir=$(setup_force_type "$force_type")
    
    echo "Starting pipeline execution for $force_type..."
    
    # Submit jobs in sequence
#    echo "Submitting simulation data creation job for $force_type..."
#    job1=$(submit_job "${ROOT_DIR}/Sh/1_creat_n-body_simulation_data.sh" "" "$work_dir")
#    echo "Job 1 submitted with ID: $job1"
    
#    echo "Submitting training job for $force_type..."
#    job2=$(submit_job "${ROOT_DIR}/Sh/2_train_graphnn_on_n-body_data.sh" "$job1" "$work_dir")
    # job2=$(submit_job "${ROOT_DIR}/Sh/2_train_graphnn_on_n-body_data.sh" "" "$work_dir")
#    echo "Job 2 submitted with ID: $job2"
    
    echo "Submitting evaluation job for $force_type..."
#    job31=$(submit_job "${ROOT_DIR}/Sh/31_evaluation.sh" "$job2" "$work_dir")
    job31=$(submit_job "${ROOT_DIR}/Sh/31_evaluation.sh" "" "$work_dir")

    echo "Job 31 submitted with ID: $job31"
    
    echo "Submitting latent space extraction job for $force_type..."
    job32=$(submit_job "${ROOT_DIR}/Sh/32_extract_latent_exp.sh" "$job2" "$work_dir")
    echo "Job 32 submitted with ID: $job32"
    
    echo "Pipeline submitted successfully for $force_type!"
    echo "Job dependencies: 1 -> 2 -> 31 & 32"
}

# Run pipeline for each forcing type
for force_type in "${FORCE_TYPES[@]}"; do
    run_pipeline_for_type "$force_type"
done

echo "All pipelines submitted successfully!"
echo "You can monitor the jobs using: squeue -u $USER" 
