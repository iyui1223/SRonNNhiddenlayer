# Symbolic Regression on GNN Hidden Representations

This project is a reproduction of [Miles Cranmer's symbolic deep learning framework](https://github.com/MilesCranmer/symbolic_deep_learning/tree/master), which applies **symbolic regression** to the **hidden representations** of **graph neural networks (GNNs)** trained on physical systems.

## Overview

The project aims to extract interpretable relationships between the **hidden layer features** and **physical quantities**, such as forces, from learned GNN representations of n-body dynamics. It combines the power of deep learning with symbolic regression to discover underlying physical laws from neural network representations.

## Features

- Supports 2D and 3D simulations of n-body systems
- Compatible with various force types defined in `simulation.py`
- Comprehensive visualization tools for analysis
- Symbolic regression on GNN hidden representations
- SLURM job scheduling support for HPC environments

## Project Structure

```
.
├── Const/      # Constant files for shared parameters
│   └── const_template.txt  # Template for configuration
├── Data/       # Created after the first execution. Data storage directory.
├── Figs/       # Created after the first execution. Output directory for visualizations
├── Log/        # Created after the first execution. Note that most jobs stores log to separate working directory under Work/.
├── Models/     # Saved model checkpoints
├── Original/   # Original n-body simulation code
├── Sh/         # Shell scripts for environment setup and SLURM jobs
├── Source/     # Core Python implementation
├── Viewer/     # Visualization tools
└── Work/       # Working directory for job outputs
```

## Getting Started

### Prerequisites

- Python 3.11 or newer
- CUDA-compatible GPU (for training)
- SLURM scheduler (for HPC environments)

### Installation

1. Clone the repository: # Replace XXXX with your own token
```bash
git clone https://oauth2:XXXX@gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/projects/yi260
cd yi260
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements_pysr.txt  # Recommend separate environment for symbolic regression
```

3. Configure the environment:
- Modify `Const/const_template.txt` with your settings
- Set the `ROOT_DIR` parameters in both `Const/const_template.txt` and `run_pipeline.sh`

### Usage

1. Run the pipeline:
```bash
cd Sh
./run_pipeline.sh
```

2. Monitor progress:
- Check `Work/${force_type}_${model_type}/Log` for execution logs
- View results in `Figs/` directory

3. Produces summary figures for each processing steps from the multiple output:
- cd Sh
- sh ${digit}_summary.sh

## Explanation of Workflow
This project is designed for systematic experimentation. Each experiment involves varying settings such as force types (SYM_TYPE) and model architectures (MODEL_TYPE). To manage this:

- Naming convention to organize the results in hierarchal manner is adopted. 

- Shared settings (e.g., number of bodies, dimensions) are stored in Const/const_template.txt.

- Shell scripts loop through different experiment configurations, injecting variations while sharing codebase and constants.

Each experiment runs in a dedicated working directory:
```bash
Work/${force_type}_${model_type}/
```

This includes:
- Finalized parameter set (const.txt)

- Individual logs

- Output data and figures (after the successful completion of the python script, wrapper shell should mv them to the suitable directories in Data/ Figs/ directories.)

Note: If you change shared constants (e.g., n, dim), make sure to rename the corresponding Work/ subdirectory to avoid overwriting the existing const.txt files and previous results.

Viewing Results
The Viewer/ directory contains previous experiment outputs. To explore them interactively:

cd Viewer/
```bash
python -m http.server 8000
```
# Then open http://localhost:8000 in your browser

All results follow this naming convention:

```bash
DATA_NAME = "${SIM_TYPE}_n${N_BODIES}_dim${DIMENSIONS}_nt${NUM_TIMESTEPS}"
MODEL_NAME = "${MODEL_TYPE}_h${HIDDEN_DIM}_m${MSG_DIM}_b${BATCH_SIZE}_e${EPOCHS}"
```

For example, a model trained with:

- Standard loss
- 256 hidden features
- 128 message features
- Batch size 16
- 1 epoch

on 4-body r2-force simulation in 2D will be saved as:
```bash
Models/r2_n4_dim2_nt250/standard_h32_m16_b16_e1.pt
```
And corresponding figures in:

```bash
Figs/r2_n4_dim2_nt250/standard_h32_m16_b16_e1/force_message_relation.png
```
