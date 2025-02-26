# PANDA ML Package
=====================

## Overview
-----------

The PANDA ML package is designed to facilitate the integration of machine learning models with the PANDA (Production and Distributed Analysis) system. This package provides tools for data preprocessing, model training, prediction, and error handling, making it easier to deploy and manage ML models in a production environment.

## Features
------------

- **Data Preprocessing**: Includes modules for handling historical and live data, with support for categorical encoding and data splitting.
- **Model Management**: Offers classes for managing model pipelines, including training and prediction capabilities.
- **Prediction Utilities**: Provides a set of utilities for fetching tasks, processing predictions, and handling errors.
- **Logging and Validation**: Includes custom logging and data validation tools to ensure robustness and reliability.

## Structure
------------

The package is organized into the following submodules:

- **`data`**: Contains classes for data preprocessing and fetching.
  - `data_manager.py`: Base data preprocessors and specific data processors.
  - `fetch_db_data.py`: Database fetcher for retrieving task parameters.
- **`model`**: Includes classes for model pipelines and management.
  - `base_model.py`: Basic model classes.
  - `model_pipeline.py`: Model pipelines for training and prediction.
- **`utils`**: Utility functions for logging, plotting, and prediction handling.
  - `logger.py`: Custom logging setup.
  - `plotting.py`: Plotting utilities for metrics.
  - `prediction_utils.py`: Prediction processing and error handling.
  - `validator.py`: Data validation tools.
- **`tests`**: Unit tests for ensuring package functionality.
- **`live_prediction.py`**: Script for running live predictions.

## Installation
------------

To install the package, ensure you have Python 3.12 or later installed. Then, follow these steps:

1. Clone the repository:
 ```
git clone git@github.com:PanDAWMS/pandaml.git
 ```
2. Navigate into the project directory


3. Create a virtual environment (recommended):

- Create the virtual environment:
  ```
  python -m venv venv
  ```
- Activate it on Linux/Mac:
  ```
  source venv/bin/activate
  ```
4. Install dependencies:
```
pip install -r requirements.txt
```
5. Execution:
```
python -m scout_ml_package.live_prediction
```
