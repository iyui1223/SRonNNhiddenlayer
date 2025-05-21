#!/bin/bash

# Set and export root directory
export ROOT_DIR="/home/yi260/final_project"

# Source constants
source "${ROOT_DIR}/Const/const.txt"

# Function to submit a job and get its job ID
submit_job() {
    local script=$1
    local dependency=$2
    
    if [ -z "$dependency" ]; then
        # Submit job without dependency
        job_id=$(sbatch --parsable "$script")
    else
        # Submit job with dependency
        job_id=$(sbatch --parsable --dependency=afterok:$dependency "$script")
    fi
    
    echo $job_id
}

echo "Starting pipeline execution..."

# Submit jobs in sequence
echo "Submitting simulation data creation job..."
job1=$(submit_job "1_creat_n-body_simulation_data.sh")
echo "Job 1 submitted with ID: $job1"

echo "Submitting training job..."
job2=$(submit_job "2_train_graphnn_on_n-body_data.sh" "$job1")
echo "Job 2 submitted with ID: $job2"

echo "Submitting evaluation job..."
job31=$(submit_job "31_evaluation.sh" "$job2")
echo "Job 31 submitted with ID: $job31"

echo "Submitting latent space extraction job..."
job32=$(submit_job "32_extract_latent_exp.sh" "$job2")
echo "Job 32 submitted with ID: $job32"

echo "Pipeline submitted successfully!"
echo "Job dependencies: 1 -> 2 -> 31 & 32"
echo "You can monitor the jobs using: squeue -u $USER" 
