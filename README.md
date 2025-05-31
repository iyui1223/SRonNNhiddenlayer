# Symbolic Regression on GNN Hidden Representations

This project is a reproduction of [Miles Cranmer's symbolic deep learning framework](https://github.com/MilesCranmer/symbolic_deep_learning/tree/master), which applies **symbolic regression** to the **hidden representations** of **graph neural networks (GNNs)** trained on physical systems.

The aim is to extract interpretable relationships between the **hidden layer features** and **physical quantities**, such as forces, from learned GNN representations of n-body dynamics.

---

## Project Structure

```bash
.
├── Sh/         # Shell scripts for environment setup and SLURM job submission with dependency management
├── Source/     # Core Python scripts for simulation, model training, analysis, and symbolic regression
├── Figs/       # Output directory for images and text summaries
├── Original/   # Original code to generate n-body simulations with multiple force types
```

---

## Features

* Supports **2D and 3D simulations**.
* Compatible with various force types defined in `simulation.py`.
* Modular structure for ease of extension and automation in HPC environments.

---
