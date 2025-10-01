# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuralFlow is a Python package for modeling neural spiking activity with continuous latent Langevin dynamics. It implements methods for fitting neural data to latent dynamical systems using Fokker-Planck equation solvers and optimization techniques.

## Key Directories

- `neuralflow/` - Core package with main classes and utilities
- `analyse/` - Analysis scripts and Jupyter notebooks for visualizing results
- `preprocess/` - Data preprocessing utilities and notebooks
- `scripts/` - Shell scripts for running experiments
- `examples/` - Example implementations from research papers
- `tests/` - Unit tests for core functionality
- `data/`, `logs/`, `saved/`, `figs/` - Data, logs, saved models, and figures directories

## Build and Installation

This is a Python package with Cython extensions. To build and install:

```bash
# Activate the correct conda environment
conda activate nf

# Install dependencies
pip install numpy matplotlib pandas scipy tqdm scikit-learn Cython

# Build and install the package
python setup.py build_ext --inplace
pip install -e .
```

## Core Architecture

### Main Classes

1. **`model`** (`neuralflow/model.py`) - Represents Langevin dynamics with parameters for potential energy (peq), initial distribution (p0), diffusion (D), and firing rates (fr)
2. **`Optimization`** (`neuralflow/optimization.py`) - Main optimization class that fits models to spike data using ADAM or GD optimizers
3. **`PDESolve`** (`neuralflow/PDE_Solve.py`) - Solves Fokker-Planck equations using spectral element methods
4. **`SpikeData`** (`neuralflow/spike_data.py`) - Handles neural spike data preprocessing and formatting
5. **`GLLgrid`** (`neuralflow/grid.py`) - Implements Gauss-Lobatto-Legendre grid for numerical computations

### Key Concepts

- **Langevin Dynamics**: Models neural activity as particles moving in a potential landscape
- **Boundary Conditions**: Supports 'absorbing' and 'reflecting' boundary modes
- **Parameter Sharing**: Models can share parameters (peq, p0, D, fr) across different conditions
- **GPU Support**: CUDA acceleration available for optimization

## Development Workflow

### Running Tests

```bash
# Run specific test modules
python -m unittest tests/test_optimization.py
python -m unittest tests/test_gradients.py
```

### Data Processing Pipeline

1. **Preprocessing**: Use scripts in `preprocess/` to clean and format spike data
2. **Model Fitting**: Use `Optimization` class with `SpikeData` objects
3. **Analysis**: Use notebooks in `analyse/` to visualize results and model fits

### Experiment Scripts

Shell scripts in `scripts/` demonstrate common parameter configurations:
- `py3_ref_cfr_no_ls_alpha_*.sh` - Reference configurations with different parameters
- `py5_all_union_ref_cfr_no_ls_alpha.sh` - Comprehensive experiment script

## Important Configuration

- **Settings**: Default parameters in `neuralflow/settings.py` including optimization bounds and constraints
- **Boundary Conditions**: Controlled via `boundary_mode` parameter ('absorbing' or 'reflecting')
- **Optimization**: Supports ADAM and GD optimizers with line search options

## Common Issues

- **NaN Errors**: Often caused by zero firing rates in mini-batches - check data preprocessing
- **Memory Issues**: GPU memory constraints may require reducing batch sizes or grid resolution
- **Convergence**: Results may be sensitive to learning rates and initialization

## Data Structure

Experimental data is organized in the `data/` directory with two main formats:

### Data Formats

1. **MATLAB format** (`*.mat` files in `data/raw_data/`)
   - Used by `analyse/py0_single.py`
   - Contains spike counts in matrix format with trial information
   - Structure: `neural_data_{date}_{brain_region}.mat`
   - Brain regions: ALM, BLA, ECT, Medulla, Midbrain, Striatum, Thalamus (left/right)

2. **Pickle format** (`*.pkl` files in `data/new_data_mini/`)
   - Used by `analyse/py4_single_new_data_mini.py`
   - Contains aligned spike times with behavioral conditions
   - Structure: `session_057_aligned_spike_times.pkl`
   - Conditions: LL, LR, RL, RR (left/right choices and outcomes)

### Preprocessing Differences

- **py0_single.py**: Processes MATLAB data with spike counts, filters neurons based on mini-batch size
- **py4_single_new_data_mini.py**: Processes pickle data with spike times, handles multiple behavioral conditions

Both scripts convert data to `SpikeData` objects with trial information and spike counts/inter-spike intervals.