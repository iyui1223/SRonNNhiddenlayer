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
- Progress tracking and logging system

## Project Structure

```
.
├── Const/      # Constant files for shared parameters
│   └── const_template.txt  # Template for configuration
├── Data/       # Data storage directory
├── Figs/       # Output directory for visualizations
├── Log/        # Execution logs
├── Models/     # Saved model checkpoints
├── Original/   # Original n-body simulation code
├── Sh/         # Shell scripts for environment setup and SLURM jobs
├── Source/     # Core Python implementation
├── Viewer/     # Visualization tools
└── Work/       # Working directory for job outputs
```

## Getting Started

### Prerequisites

- Python 3.x
- CUDA-compatible GPU (for training)
- SLURM scheduler (for HPC environments)

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements_pysr.txt  # For symbolic regression
```

3. Configure the environment:
- Modify `Const/const_template.txt` with your settings
- Set the `ROOT_DIR` parameter

### Usage

1. Run the pipeline:
```bash
cd Sh
./run_pipeline.sh
```

2. Monitor progress:
- Check `Work/${force_type}_${model_type}/Log` for execution logs
- View results in `Figs/` directory

## Key Components

- **Simulation**: N-body physics simulation with multiple force types
- **GNN Training**: Graph neural network implementation for learning physical dynamics
- **Symbolic Regression**: Extraction of interpretable equations from GNN representations
- **Visualization**: Tools for analyzing and presenting results

---
