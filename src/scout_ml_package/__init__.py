# src/scout_ml_package/__init__.py
__author__ = "Tasnuva Chowdhury"
__email__ = "ch.tasnuva@cern.ch"

# Importing necessary components from submodules
from .model.base_model import (
    MultiOutputModel,
    ModelTrainer,
    TrainedModel,
    ModelPipeline,
)  # Import your model classes
from .data.data_manager import (
    HistoricalDataProcessor,
    DataSplitter,
    ModelTrainingInput,
    CategoricalEncoder,
    BaseDataPreprocessor,
    TrainingDataPreprocessor,
    NewDataPreprocessor,
    LiveDataPreprocessor,
)
from .data.fetch_db_data import DatabaseFetcher
from .model.model_pipeline import (
    TrainingPipeline,
    ModelHandlerInProd,
)  # ModelLoader,
from .utils.plotting import ErrorMetricsPlotter, ClassificationMetricsPlotter
from .utils.validator import FakeListener, DummyData, DataValidator
from .utils.logger import Logger  

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
    "FakeListener",
    "DummyData",
    "DataValidator",
    "DatabaseFetcher",
    "Logger",
]

# Optional: Example of initializing common configurations
DEFAULT_INPUT_SHAPE = (
    10  # Set a default value for input shape, adjust as necessary
)


# Any additional initialization or configuration that is common and should be done on package load can be added here.
