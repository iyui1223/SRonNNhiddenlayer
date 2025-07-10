#!/bin/bash
# Visualize the table of the output R2/sparcity/ACC values.

set -e

source "../Const/const_template.txt" # load the shared constants among experiment types

# python venv
source ${SOURCE_ENV}

SUMMARY_NAME="n${N_BODIES}_dim${DIMENSIONS}_nt${NUM_TIMESTEPS}"
OUTPUT_DIR="${FIG_DIR}/Summary/${SUMMARY_NAME}/"
mkdir -p "$OUTPUT_DIR"

WORK_DIR="${ROOT_DIR}/Work/Summary/${SUMMARY_NAME}"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"

ln -rsf "${SOURCE_DIR}/plot_summary_tables.py" .

# Initialize output files
SPARSITY_TABLE="$OUTPUT_DIR/sparsity_table.tsv"
R2_TABLE="$OUTPUT_DIR/r2_table.tsv"
ACC_TABLE="$OUTPUT_DIR/acc_table.tsv"

echo -e "ForceType\tModel\tSparsity" > "$SPARSITY_TABLE"
echo -e "ForceType\tModel\tR2" > "$R2_TABLE"
echo -e "ForceType\tModel\tACC" > "$ACC_TABLE"

ln -rsf ${SPARSITY_TABLE}
ln -rsf ${R2_TABLE}
ln -rsf ${ACC_TABLE}

# Function to extract ACC values from metrics.txt and average timesteps 50-55
extract_acc_average() {
    local metrics_file="$1"
    local acc_values=""
    
    if [[ -f "$metrics_file" ]]; then
        # Extract mean ACC value from the Mean metrics section
        acc_values=$(awk '/^Mean ACC:/ {print $3}' "$metrics_file")
        if [[ -z "$acc_values" ]]; then
            echo "N/A"
        else
            echo "$acc_values"
        fi
    else
        echo "N/A"
    fi
}

FORCE_TYPES=("spring" "charge" "damped" "disc" "r1" "r2")
MODEL_TYPES=("standard" "bottleneck" "KL" "FlatHGN" "L1")
for force in "${FORCE_TYPES[@]}"; do
    for model in "${MODEL_TYPES[@]}"; do
        DATA_NAME="${force}_n${N_BODIES}_dim${DIMENSIONS}_nt${NUM_TIMESTEPS}"
        MODEL_NAME="${model}_h${HIDDEN_DIM}_m${MSG_DIM}_b${BATCH_SIZE}_e${EPOCHS}.pt"
        EVAL_DIR="${ROOT_DIR}/Figs/${DATA_NAME%.npz}/${MODEL_NAME%.pt}"
        if [[ -d "$EVAL_DIR" ]]; then
            sparsity_file="${EVAL_DIR}/message_sparsity.txt"
            r2_file="${EVAL_DIR}/message_r2_scores.txt"
            metrics_file="${EVAL_DIR}/metrics.txt"
            
            if [[ -f "$sparsity_file" ]]; then
                sparsity=$(awk '/Mean sparsity ratio/ {print $4}' "$sparsity_file")
                echo -e "${force}\t${model}\t${sparsity}" >> "$SPARSITY_TABLE"
            fi

            if [[ -f "$r2_file" ]]; then
                r2=$(awk '/Message Element 1:/ {print $4}' "$r2_file")
                echo -e "${force}\t${model}\t${r2}" >> "$R2_TABLE"
            fi
            
            if [[ -f "$metrics_file" ]]; then
                acc_avg=$(extract_acc_average "$metrics_file")
                echo -e "${force}\t${model}\t${acc_avg}" >> "$ACC_TABLE"
            fi
        fi
    done
done

# Run Python visualization
python plot_summary_tables.py --sparsity "$SPARSITY_TABLE" --r2 "$R2_TABLE" --acc "$ACC_TABLE" --outdir "."

# Move results
mv sparsity_table.png "$OUTPUT_DIR/"
mv r2_table.png "$OUTPUT_DIR/"
mv acc_table.png "$OUTPUT_DIR/"

echo "Summary tables and plots saved to $OUTPUT_DIR"
