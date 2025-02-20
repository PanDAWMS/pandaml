# src/scout_ml_package/__init__.py
__author__ = "Tasnuva Chowdhury"
__email__ = "ch.tasnuva@cern.ch"

from .data.data_manager import (
    BaseDataPreprocessor,
    CategoricalEncoder,
    DataSplitter,
    HistoricalDataProcessor,
    LiveDataPreprocessor,
    ModelTrainingInput,
    NewDataPreprocessor,
    TrainingDataPreprocessor,
)
from .data.fetch_db_data import DatabaseFetcher
from .utils.message import ConfigLoader, MyListener  # TaskIDListener

# Importing necessary components from submodules
from .model.base_model import ModelPipeline  # Import your model classes
from .model.base_model import ModelTrainer, MultiOutputModel, TrainedModel
from .model.model_pipeline import (
    ModelHandlerInProd,
    TrainingPipeline,
    ColumnTransformer,
)  # ModelLoader,
from .utils.logger import Logger
from .utils.plotting import ClassificationMetricsPlotter, ErrorMetricsPlotter
from .utils.validator import DataValidator, DummyData, FakeListener

__all__ = [
    "HistoricalDataProcessor",
    "DataSplitter",
    "ModelTrainingInput",
    "CategoricalEncoder",
    "BaseDataPreprocessor",
    "TrainingDataPreprocessor",
    "NewDataPreprocessor",
    "MultiOutputModel",  # Allow access to the MultiOutputModel
    "Logger",  # Allow access to Logger
    "ErrorMetricsPlotter",
    "ClassificationMetricsPlotter",
    "ModelTrainer",  # Allow access to ModelTrainer
    "TrainedModel",  # Allow access to TrainedModel
    "ModelPipeline",
    "TrainingPipeline",
    "ModelLoader",
    "LiveDataPreprocessor",
    "ModelHandlerInProd",
    "ColumnTransformer",
    "FakeListener",
    "DummyData",
    "DataValidator",
    "DatabaseFetcher",
    "Logger",
    "MyListener",
    "ConfigLoader",  # "TaskIDListener",
]

# Optional: Example of initializing common configurations
DEFAULT_INPUT_SHAPE = 10  # Set a default value for input shape, adjust as necessary


# Any additional initialization or configuration that is common and should be done on package load can be added here.
