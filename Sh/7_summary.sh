#!/bin/bash
set -e

source "../Const/const_template.txt" # load the shared constants among experiment types

# python venv
source ~/venvs/final_clean/bin/activate

ANALYSIS_OUTPUT_DIR="${ROOT_DIR}/Data/analysis_output"
mkdir -p "$ANALYSIS_OUTPUT_DIR"

cd "${ROOT_DIR}/Work_SR" # uses separate working directory for overright safety

# Function to get latest timestamped directories
get_latest_dirs() {
    local outputs_dir="$1"
    local num_latest="$2"
    
    if [[ ! -d "$outputs_dir" ]]; then
        return
    fi
    
    # Find all timestamped directories (YYYYMMDD_HASH_HASH format) and sort by timestamp
    find "$outputs_dir" -maxdepth 1 -type d -name "[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_*" | \
    sort -r | head -n "$num_latest"
}

# Process each Work subdirectory
for work_subdir in */; do
    # Skip if not a directory or starts with dot
    if [[ ! -d "$work_subdir" ]] || [[ "$work_subdir" == .* ]]; then
        continue
    fi
    
    work_subdir=${work_subdir%/}  # Remove trailing slash
    const_file="$work_subdir/const.txt"
    outputs_dir="$work_subdir/outputs"
    
    echo "Processing $work_subdir..."
    
    # Skip if const.txt doesn't exist
    if [[ ! -f "$const_file" ]]; then
        echo "  Skipping: const.txt not found"
        continue
    fi
    
    # Source the const.txt file to get the parameters
    # Set ROOT_DIR to the project root and source the const file
    ROOT_DIR="$ROOT_DIR"
    source "$const_file"
    
    # Check if we got the required parameters
    if [[ -z "$DATA_NAME" ]] || [[ -z "$MODEL_NAME" ]]; then
        echo "  Skipping: Could not get DATA_NAME or MODEL_NAME from const.txt"
        continue
    fi
    
    # Get latest timestamped directories (3 latest)
    latest_dirs=$(get_latest_dirs "$outputs_dir" 3)
    
    if [[ -z "$latest_dirs" ]]; then
        echo "  No timestamped directories found in outputs/"
        continue
    fi
    
    # Copy each latest directory
    counter=1
    while IFS= read -r timestamped_dir; do
        if [[ -z "$timestamped_dir" ]]; then
            continue
        fi
        
        # Extract just the directory name from the full path
        dir_name=$(basename "$timestamped_dir")
        
        # Create target directory name: DATA_NAME_MODEL_NAME_latest{i}
        model_name_without_ext="${MODEL_NAME%.pt}"
        target_dir_name="${DATA_NAME}_${model_name_without_ext}_dim${counter}"
        target_dir="$ANALYSIS_OUTPUT_DIR/$target_dir_name"
        
        # Copy the directory
        if [[ -d "$target_dir" ]]; then
            rm -rf "$target_dir"
        fi
        
        cp -r "$timestamped_dir" "$target_dir"
        echo "  Copied $dir_name -> $target_dir_name"
        
        ((counter++))
    done <<< "$latest_dirs"
    
    echo "  Processed $((counter-1)) directories"
done

echo "All latest outputs copied to $ANALYSIS_OUTPUT_DIR" 