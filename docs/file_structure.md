
# File Structure of Keypoint Pseudo Labeler Repository

This document provides an overview of the file structure of the `keypoint-pseudo-labeler` repository. Below is the skeleton outline describing the organization and purpose of various files and directories within the repository.

## Root Directory

- **README.md**: Overview of the repository, including installation instructions and usage examples.
- **LICENSE**: License information for the repository.
- **setup.py**: Script for installing the package.
- **requirements.txt**: List of dependencies required to run the project.

## `src/`

Contains the main source code for the project.

- **`__init__.py`**: Indicates that this directory is a Python package.
- **`data/`**: Data loading and processing scripts.
  - **`__init__.py`**
  - **`load_data.py`**: Functions to load and preprocess data.
- **`models/`**: Model architectures and training scripts.
  - **`__init__.py`**
  - **`model.py`**: Definition of the model architecture.
  - **`train.py`**: Training loop and evaluation metrics.
- **`utils/`**: Utility functions and helper scripts.
  - **`__init__.py`**
  - **`logger.py`**: Logging utilities.
  - **`metrics.py`**: Evaluation metrics functions.

## `notebooks/`

Contains Jupyter notebooks for exploratory data analysis and prototyping.

- **`EDA.ipynb`**: Exploratory Data Analysis notebook.
- **`model_training.ipynb`**: Notebook for training models and visualizing results.

## `tests/`

Contains unit tests for the project.

- **`__init__.py`**
- **`test_data.py`**: Tests for data loading and processing functions.
- **`test_model.py`**: Tests for model architecture and training functions.
- **`test_utils.py`**: Tests for utility functions.

## `scripts/`

Contains various scripts for running experiments and data processing.

- **`run_experiment.py`**: Script to run experiments with different configurations.
- **`process_data.py`**: Script for data processing and augmentation.

## `config/`

Contains configuration files for different environments and experiments.

- **`config.yaml`**: Default configuration file.
- **`experiment_1.yaml`**: Configuration for experiment 1.
- **`experiment_2.yaml`**: Configuration for experiment 2.

## `data/`

Directory for storing raw and processed data. This directory is typically not included in version control.

- **`raw/`**: Raw data files.
- **`processed/`**: Processed data files ready for analysis.

## `results/`

Directory for storing results from experiments, including model checkpoints and logs.

- **`checkpoints/`**: Directory for saving model checkpoints.
- **`logs/`**: Directory for saving training logs.
- **`plots/`**: Directory for saving plots and visualizations.

## Other Files

- **`.gitignore`**: Specifies files and directories to be ignored by Git.
- **`Dockerfile`**: Dockerfile for creating a containerized environment.
- **`Makefile`**: Makefile for automating tasks.
