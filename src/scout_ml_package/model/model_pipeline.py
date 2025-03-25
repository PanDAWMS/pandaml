# src/scout_ml_package/model/model_pipeline.py

# Standard library imports
import os
import re
from typing import List, Tuple, Optional

# Third-party imports
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import TFSMLayer
from sklearn.preprocessing import MinMaxScaler

# Local application imports
from scout_ml_package.data import (
    LiveDataPreprocessor,
    NewDataPreprocessor,
    TrainingDataPreprocessor,
)
from scout_ml_package.model import MultiOutputModel
from scout_ml_package.model.base_model import ModelTrainer  # , ModelPipeline
from scout_ml_package.model.base_model import DeviceInfo
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class TrainingPipeline:
    """
    Manages the training pipeline for machine learning models.

    Attributes:
    - numerical_features (List[str]): List of numerical feature names.
    - categorical_features (List[str]): List of categorical feature names.
    - category_list (List[List]): List of unique categories for each categorical feature.
    - model_target (List[str]): List of target column names.
    """

    def __init__(
        self,
        numerical_features: List[str],
        categorical_features: List[str],
        category_list: List[List],
        model_target: List[str],
    ):
        """
        Initializes the TrainingPipeline with feature and target information.

        Parameters:
        - numerical_features (List[str]): List of numerical feature names.
        - categorical_features (List[str]): List of categorical feature names.
        - category_list (List[List]): List of unique categories for each categorical feature.
        - model_target (List[str]): List of target column names.

        Raises:
        - TypeError: If any of the parameters are not lists.
        """
        
        if (
            not isinstance(numerical_features, list)
            or not isinstance(categorical_features, list)
            or not isinstance(category_list, list)
            or not isinstance(model_target, list)
        ):
            raise TypeError("All parameters must be lists")

        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.category_list = category_list
        self.model_target = model_target

    def preprocess_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, future_data: pd.DataFrame) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        List[str],
        MinMaxScaler,
    ]:
        """
        Preprocesses training, testing, and future data.

        Parameters:
        - train_df (pd.DataFrame): Training data.
        - test_df (pd.DataFrame): Testing data.
        - future_data (pd.DataFrame): Future data for predictions.

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], tf.keras.preprocessing.MinMaxScaler]: Preprocessed training data, testing data, future data, encoded column names, and fitted scaler.

        Raises:
        - TypeError: If any of the DataFrames are not pandas DataFrames.
        """
        if (
            not isinstance(train_df, pd.DataFrame)
            or not isinstance(test_df, pd.DataFrame)
            or not isinstance(future_data, pd.DataFrame)
        ):
            raise TypeError("All data must be pandas DataFrames")

        training_preprocessor = TrainingDataPreprocessor()
        processed_train_data, encoded_columns, fitted_scaler = training_preprocessor.preprocess(
            train_df,
            self.numerical_features,
            self.categorical_features,
            self.category_list,
        )

        new_data_preprocessor = NewDataPreprocessor()

        processed_test_data = new_data_preprocessor.preprocess(
            test_df,
            self.numerical_features,
            self.categorical_features,
            self.category_list,
            fitted_scaler,
            encoded_columns + self.model_target + ["JEDITASKID"],
        )

        processed_future_data = new_data_preprocessor.preprocess(
            future_data,
            self.numerical_features,
            self.categorical_features,
            self.category_list,
            fitted_scaler,
            encoded_columns + self.model_target + ["JEDITASKID"],
        )

        return (
            processed_train_data,
            processed_test_data,
            processed_future_data,
            encoded_columns,
            fitted_scaler,
        )

    def train_model(
        self,
        processed_train_data: pd.DataFrame,
        processed_test_data: pd.DataFrame,
        features_to_train: List[str],
        build_function_name: str,
        epoch: int,
        batch: int,
    ) -> tf.keras.Model:
        """
        Trains a regression model.

        Parameters:
        - processed_train_data (pd.DataFrame): Preprocessed training data.
        - processed_test_data (pd.DataFrame): Preprocessed testing data.
        - features_to_train (List[str]): List of features to use for training.
        - build_function_name (str): Name of the build function for the model.
        - epoch (int): Number of epochs for training.
        - batch (int): Batch size for training.

        Returns:
        - tf.keras.Model: Trained model.

        Raises:
        - TypeError: If any of the parameters are not of the correct type.
        """
        if not isinstance(processed_train_data, pd.DataFrame) or not isinstance(processed_test_data, pd.DataFrame):
            raise TypeError("Data must be pandas DataFrames")
        if not isinstance(features_to_train, list) or not isinstance(build_function_name, str):
            raise TypeError("features_to_train must be a list and build_function_name must be a string")
        if not isinstance(epoch, int) or not isinstance(batch, int):
            raise TypeError("epoch and batch must be integers")

        X_train, y_train = (
            processed_train_data[features_to_train],
            processed_train_data[self.model_target],
        )
        X_val, y_val = (
            processed_test_data[features_to_train],
            processed_test_data[self.model_target],
        )

        # 1D output handling
        output_shape = y_train.shape[1] if y_train.ndim > 1 else 1
        print("Output shape:", output_shape)

        trainer = ModelTrainer(
            MultiOutputModel,
            input_shape=X_train.shape[1],
            output_shape=output_shape,
            loss_function="mse",
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
            build_function=build_function_name,
        )

        model_ramcount, history_ramcount = trainer.train(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epoch,
            batch_size=batch,
            build_function_name=build_function_name,
        )
        print(model_ramcount.summary())
        return model_ramcount

    def train_classification_model(
        self,
        processed_train_data: pd.DataFrame,
        processed_test_data: pd.DataFrame,
        features_to_train: List[str],
        build_function_name: str,
        epoch: int,
        batch: int,
        model_type: str = "binary",
    ) -> tf.keras.Model:
        """
        Trains a classification model.

        Parameters:
        - processed_train_data (pd.DataFrame): Preprocessed training data.
        - processed_test_data (pd.DataFrame): Preprocessed testing data.
        - features_to_train (List[str]): List of features to use for training.
        - build_function_name (str): Name of the build function for the model.
        - epoch (int): Number of epochs for training.
        - batch (int): Batch size for training.
        - model_type (str, optional): Type of classification model. Defaults to "binary".

        Returns:
        - tf.keras.Model: Trained model.

        Raises:
        - TypeError: If any of the parameters are not of the correct type.
        """
        if not isinstance(processed_train_data, pd.DataFrame) or not isinstance(processed_test_data, pd.DataFrame):
            raise TypeError("Data must be pandas DataFrames")
        if not isinstance(features_to_train, list) or not isinstance(build_function_name, str):
            raise TypeError("features_to_train must be a list and build_function_name must be a string")
        if not isinstance(epoch, int) or not isinstance(batch, int):
            raise TypeError("epoch and batch must be integers")
        if not isinstance(model_type, str):
            raise TypeError("model_type must be a string")

        X_train, y_train = (
            processed_train_data[features_to_train],
            processed_train_data[self.model_target],
        )
        X_val, y_val = (
            processed_test_data[features_to_train],
            processed_test_data[self.model_target],
        )

        # Determine output shape and related settings based on the model type
        if model_type == "binary":
            output_shape = 1
            loss_function = "binary_crossentropy"
            metrics = [tf.keras.metrics.BinaryAccuracy()]
        else:  # assume 'multiclass'
            output_shape = len(y_train.unique())  # for one-hot encoded classes, check y_train.shape[1]
            loss_function = "categorical_crossentropy"
            metrics = [tf.keras.metrics.CategoricalAccuracy()]

        print("Output shape:", output_shape)

        trainer = ModelTrainer(
            MultiOutputModel,
            input_shape=X_train.shape[1],
            output_shape=output_shape,
            loss_function=loss_function,
            metrics=metrics,
            build_function=build_function_name,
        )

        model, history = trainer.train(X_train, y_train, X_val, y_val, epochs=epoch, batch_size=batch)
        return model

    def regression_prediction(
        self,
        trained_model: tf.keras.Model,
        processed_future_data: pd.DataFrame,
        features_to_train: List[str],
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Makes predictions using a trained regression model.

        Parameters:
        - trained_model (tf.keras.Model): Trained model.
        - processed_future_data (pd.DataFrame): Preprocessed future data.
        - features_to_train (List[str]): List of features to use for prediction.

        Returns:
        - Tuple[pd.DataFrame, np.ndarray]: DataFrame with predictions and the raw prediction array.

        Raises:
        - TypeError: If any of the parameters are not of the correct type.
        """
        if not isinstance(trained_model, tf.keras.Model):
            raise TypeError("trained_model must be a tf.keras.Model")
        if not isinstance(processed_future_data, pd.DataFrame):
            raise TypeError("processed_future_data must be a pandas DataFrame")
        if not isinstance(features_to_train, list):
            raise TypeError("features_to_train must be a list")

        X_test, y_test = (
            processed_future_data[features_to_train],
            processed_future_data[self.model_target],
        )
        print("y_test shape:", y_test.shape)

        # Evaluate the model
        y_pred = trained_model.predict(X_test)

        pred_names = [
            f"Predicted_{element}" for element in self.model_target
        ]  # ['Predicted_RAMCOUNT', 'Predicted_cputime_HS']
        predicted_df = processed_future_data.copy()
        print(self.model_target)
        predicted_df[self.model_target] = y_test
        for i in range(len(pred_names)):
            print(i, pred_names[i])
            predicted_df[pred_names[i]] = y_pred[:, i]

        print("Raw prediction shape:", y_pred.shape)
        print(predicted_df.head())
        return predicted_df, y_pred

    def classification_prediction(
        self,
        trained_model: tf.keras.Model,
        processed_future_data: pd.DataFrame,
        features_to_train: List[str],
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Makes predictions using a trained classification model.

        Parameters:
        - trained_model (tf.keras.Model): Trained model.
        - processed_future_data (pd.DataFrame): Preprocessed future data.
        - features_to_train (List[str]): List of features to use for prediction.

        Returns:
        - Tuple[pd.DataFrame, np.ndarray]: DataFrame with predictions and the raw prediction array.

        Raises:
        - TypeError: If any of the parameters are not of the correct type.
        """
        if not isinstance(trained_model, tf.keras.Model):
            raise TypeError("trained_model must be a tf.keras.Model")
        if not isinstance(processed_future_data, pd.DataFrame):
            raise TypeError("processed_future_data must be a pandas DataFrame")
        if not isinstance(features_to_train, list):
            raise TypeError("features_to_train must be a list")

        # Prepare test data
        X_test = processed_future_data[features_to_train]
        y_test = processed_future_data[self.model_target]

        # Predict class labels
        y_pred = trained_model.predict(X_test)

        # Convert numerical predictions back to 'low' or 'high'
        y_pred_text = np.where(y_pred > 0.5, "high", "low")  # Assuming binary classification

        # Create predictions DataFrame
        pred_names = [f"Predicted_{element}" for element in self.model_target]
        predicted_df = processed_future_data.copy()
        predicted_df[self.model_target] = y_test  # Keep actual values (which are already 0 or 1)
        predicted_df[pred_names] = y_pred_text  # Store predicted values as 'low' or 'high'

        # Convert actual values back to 'low' or 'high' for consistency
        predicted_df[self.model_target] = predicted_df[self.model_target].replace({0: "low", 1: "high"})
        return predicted_df, y_pred_text


class ColumnTransformer:
    """
    Transforms specific columns in a DataFrame to standardized formats.

    Attributes:
    - None
    """

    def convert_processingtype(self, processingtype: str) -> str:
        """
        Converts PROCESSINGTYPE to 'P' by extracting relevant parts.

        Parameters:
        - processingtype (str): Input processing type.

        Returns:
        - str: Converted processing type.

        Raises:
        - TypeError: If processingtype is not a string.
        """
        if not isinstance(processingtype, str):
            raise TypeError("processingtype must be a string")

        if processingtype is not None and re.search(r"-.*-", processingtype):
            return "-".join(processingtype.split("-")[-2:])
        return processingtype

    def convert_transhome(self, transhome: str) -> Optional[str]:
        """
        Converts TRANSHOME to 'F' by extracting relevant parts.

        Parameters:
        - transhome (str): Input transhome.

        Returns:
        - Optional[str]: Converted transhome or None if input is None.

        Raises:
        - TypeError: If transhome is not a string or None.
        """
        if transhome is None:
            return None
        if not isinstance(transhome, str):
            raise TypeError("transhome must be a string or None")

        if "AnalysisBase" in transhome:
            return "AnalysisBase"
        elif "AnalysisTransforms-" in transhome:
            part_after_dash = transhome.split("-")[1]
            return part_after_dash.split("_")[0]
        elif "/" in transhome:
            return transhome.split("/")[0]
        else:
            return transhome.split("-")[0]

    def convert_corecount(self, corecount: int) -> str:
        """
        Converts CORECOUNT to 'Core' based on its value.

        Parameters:
        - corecount (int): Input core count.

        Returns:
        - str: Converted core count ('S' for single core, 'M' for multi-core).

        Raises:
        - TypeError: If corecount is not an integer.
        """
        if not isinstance(corecount, int):
            raise TypeError("corecount must be an integer")

        return "S" if corecount == 1 else "M"

    def transform_features(self, df: pd.DataFrame, selected_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Applies transformations to the input DataFrame.

        Parameters:
        - df (pd.DataFrame): Input DataFrame.
        - selected_columns (Optional[List[str]]): Optional list of columns to select after transformation. If None, returns all columns.

        Returns:
        - pd.DataFrame: Transformed DataFrame.

        Raises:
        - TypeError: If df is not a DataFrame or if selected_columns is not a list or None.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if selected_columns is not None and not isinstance(selected_columns, list):
            raise TypeError("selected_columns must be a list or None")

        df["P"] = df["PROCESSINGTYPE"].apply(self.convert_processingtype)
        df["F"] = df["TRANSHOME"].apply(self.convert_transhome)
        df["CORE"] = df["CORECOUNT"].apply(self.convert_corecount)

        #KEEP_F_TAG = [
        #    "Athena",
        #    "AnalysisBase",
        #    "AtlasOffline",
        #    "AthAnalysis",
        #    "AthSimulation",
        #    "MCProd",
        #    "AthGeneration",
        #    "AthDerivation",
        ###]
        #KEEP_P_TAG = [
        #    "jedi-run",
        #    "deriv",
        #    "athena-trf",
        #    "jedi-athena",
        #    "simul",
        #    "pile",
        #    "merge",
        #    "evgen",
        #    "reprocessing",
        #    "recon",
        #    "eventIndex",
        #]

        KEEP_P_TAG = ['deriv', 'jedi-run', 'merge', 'jedi-athena', 'simul', 'pile', 'evgen', 'recon', 'reprocessing', 'athena-trf', 'run-evp', 'athena-evp']
        KEEP_F_TAG =['AthDerivation', 'AnalysisBase', 'Athena', 'AthAnalysis','AtlasOffline', 'AthGeneration', 'AthSimulation','MCProd']


        df["P"] = df["P"].apply(lambda x: x if x in KEEP_P_TAG else "others")
        df["F"] = df["F"].apply(lambda x: x if x in KEEP_F_TAG else "others")

        if selected_columns is not None:
            return df[selected_columns]
        else:
            return df


class ModelHandlerInProd:
    """
    Handles loading and using a machine learning model for predictions.

    Attributes:
    - model_sequence (str): Sequence number of the model.
    - target_name (str): Name of the target variable.
    - model: Loaded model instance.
    - scaler: Loaded scaler instance.
    """

    def __init__(self, model_sequence: str, target_name: str):
        """
        Initializes the ModelHandlerInProd with model sequence and target name.

        Parameters:
        - model_sequence (str): Sequence number of the model.
        - target_name (str): Name of the target variable.

        Raises:
        - TypeError: If model_sequence or target_name are not strings.
        """
        if not isinstance(model_sequence, str) or not isinstance(target_name, str):
            raise TypeError("model_sequence and target_name must be strings")

        self.model_sequence = model_sequence
        self.target_name = target_name
        self.model = None
        self.scaler = None

    def load_model_and_scaler(self, base_path: str = None) -> None:
        """
        Loads the model and scaler using an absolute path.

        Parameters:
        - base_path (str): The base directory where models are stored. If None, defaults to the current working directory.

        Raises:
        - Exception: If there's an issue loading the model or scaler.
        """
        try:
            # Ensure base_path is absolute; default to current working directory if not provided
            if base_path is None:
                base_path = os.getcwd()
            model_storage_path = os.path.abspath(os.path.join(base_path, f"ModelStorage/model{self.model_sequence}/"))

            # Load scaler and model
            self.scaler = joblib.load(os.path.join(model_storage_path, "scaler.pkl"))
            model_name = f"model{self.model_sequence}_{self.target_name}"
            model_full_path = os.path.join(model_storage_path, model_name)

            # Assuming TFSMLayer is a custom class for loading TensorFlow models
            self.model = TFSMLayer(model_full_path, call_endpoint="serving_default")

            print(f"Model and scaler for {self.target_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading model and scaler: {e}")

    def inherited_preprocessor(
        self,
        df: pd.DataFrame,
        numerical_features: List[str],
        category_sequence: List[str],
        unique_elements_categories: List[List],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preprocesses the data using the loaded scaler.

        Parameters:
        - df (pd.DataFrame): DataFrame to preprocess.
        - numerical_features (List[str]): List of numerical feature names.
        - category_sequence (List[str]): List of categorical feature names.
        - unique_elements_categories (List[List]): List of unique categories for each categorical feature.

        Returns:
        - Tuple[pd.DataFrame, List[str]]: Preprocessed DataFrame and list of features to train.

        Raises:
        - TypeError: If df is not a DataFrame or if numerical_features/category_sequence/unique_elements_categories are not lists.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if (
            not isinstance(numerical_features, list)
            or not isinstance(category_sequence, list)
            or not isinstance(unique_elements_categories, list)
        ):
            raise TypeError("numerical_features, category_sequence, and unique_elements_categories must be lists")

        required_columns = numerical_features + category_sequence
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns in input DataFrame: {missing_columns}")
            return None, None  # Or raise an exception based on your use case

        # Perform preprocessing as before
        pprocessor = LiveDataPreprocessor()  # Instantiate if required
        processed_df, encoded_columns = pprocessor.preprocess(
            df,
            numerical_features,
            category_sequence,
            unique_elements_categories,
            self.scaler,
        )
        features_to_train = encoded_columns + numerical_features
        return processed_df, features_to_train

    def make_predictions(self, df: pd.DataFrame, features_to_train: List[str]) -> np.ndarray:
        """
        Makes predictions using the loaded model.

        Parameters:
        - df (pd.DataFrame): DataFrame containing input data.
        - features_to_train (List[str]): List of features to use for prediction.

        Returns:
        - np.ndarray: Array of predicted values.

        Raises:
        - TypeError: If df is not a DataFrame or if features_to_train is not a list.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not isinstance(features_to_train, list):
            raise TypeError("features_to_train must be a list")

        predictions = self.model(df[features_to_train])
        # Extract the tensor from the predictions dictionary
        predicted_tensor = predictions["output_0"]  # Adjust the key based on actual output
        predicted_values = predicted_tensor.numpy()  # Convert tensor to NumPy array

        # Check if this is a classification model (model 5)
        if self.model_sequence == "5":
            # For classification, we want to return class labels
            if predicted_values.ndim > 1 and predicted_values.shape[1] > 1:
                # Multi-class classification
                predicted_labels = np.argmax(predicted_values, axis=1)
            else:
                # Binary classification
                predicted_labels = (predicted_values > 0.5).astype(int)

            # Convert 0/1 to 'low'/'high'
            return np.where(predicted_labels == 0, "low", "high")
        else:
            # For regression models (1-4), return the values directly
            if predicted_values.ndim > 1:
                predicted_values = predicted_values[:, 0]
            return predicted_values


########################  For Production ############################


class ModelManager:
    """
    Manages multiple machine learning models by loading and providing access to them.

    Attributes:
    - models (Dict[str, ModelHandlerInProd]): Dictionary of loaded models.
    - base_path (str): Base path for model files.
    - models_loaded (bool): Flag indicating whether models have been loaded.
    """

    def __init__(self, base_path: str):
        """
        Initializes the ModelManager with a base path for model files.

        Parameters:
        - base_path (str): Path to the directory containing model files.

        Raises:
        - TypeError: If base_path is not a string.
        """
        if not isinstance(base_path, str):
            raise TypeError("base_path must be a string")

        self.models = {}
        self.base_path = base_path
        self.models_loaded = False  # New attribute to track model loading status

    def load_models(self) -> None:
        """
        Loads all models specified in the model configurations.

        Raises:
        - Exception: If there's an issue loading any model.
        """
        model_configs = [
            ("1", "ramcount"),
            ("2", "cputime_low"),
            ("3", "cputime_high"),
            ("4", "cpu_eff"),
            ("5", "io"),
        ]
        for sequence, target_name in model_configs:
            try:
                model = ModelHandlerInProd(model_sequence=sequence, target_name=target_name)
                model.load_model_and_scaler(self.base_path)
                self.models[sequence] = model
            except Exception as e:
                print(f"Error loading model {sequence}: {e}")
                raise
        self.models_loaded = True  # Set to True after loading models

    def are_models_loaded(self) -> bool:
        """
        Checks if models have been loaded.

        Returns:
        - bool: True if models are loaded, False otherwise.
        """
        return self.models_loaded

    def get_model(self, sequence: str) -> ModelHandlerInProd:
        """
        Retrieves a model by its sequence number.

        Parameters:
        - sequence (str): Sequence number of the model.

        Returns:
        - ModelHandlerInProd: The requested model.

        Raises:
        - ValueError: If models are not loaded.
        - KeyError: If the sequence number does not match any loaded model.
        """
        if not self.models_loaded:
            raise ValueError("Models are not loaded")
        if sequence not in self.models:
            raise KeyError(f"Model with sequence {sequence} not found")
        return self.models[sequence]


class PredictionPipeline:
    """
    Manages the prediction pipeline using models from a ModelManager.

    Attributes:
    - model_manager (ModelManager): Instance of ModelManager for accessing models.
    - numerical_features (List[str]): List of numerical feature names.
    - category_sequence (List[str]): List of categorical feature names.
    - unique_elements_categories (List[List]): List of unique categories for each categorical feature.
    """

    def __init__(self, model_manager: "ModelManager"):
        """
        Initializes the PredictionPipeline with a ModelManager instance.

        Parameters:
        - model_manager (ModelManager): Instance of ModelManager for accessing models.

        Raises:
        - TypeError: If model_manager is not an instance of ModelManager.
        """
        if not isinstance(model_manager, ModelManager):
            raise TypeError("model_manager must be an instance of ModelManager")

        self.model_manager = model_manager
        self.numerical_features = [
            "TOTAL_NFILES",
            "TOTAL_NEVENTS",
            "DISTINCT_DATASETNAME_COUNT",
        ]
        self.category_sequence = [
            "PRODSOURCELABEL",
            "P",
            "F",
            "CORE",
            "CPUTIMEUNIT",
        ]

        self.unique_elements_categories = [
            ["managed", "user"],
            [
                "deriv",
                "pile",
                "jedi-run",
                "simul",
                "athena-trf",
                "jedi-athena",
                "recon",
                "reprocessing",
                "evgen",
                "others",
                "merge",
                "run-evp",
                "athena-evp",
            ],
            [
                "Athena",
                "AtlasOffline",
                "others",
                "AthDerivation",
                "AnalysisBase",
                "AthAnalysis",
                "AthSimulation",
                "AthGeneration",
                "MCProd",
            ],
            ["M", "S"],
            ["HS06sPerEvent", "mHS06sPerEvent"],
            ]

    def make_predictions_for_model(
        self, model_sequence: str, features: List[str], input_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Makes predictions using a specified model.

        Parameters:
        - model_sequence (str): Sequence number of the model to use.
        - features (List[str]): List of feature names to use for prediction.
        - input_df (pd.DataFrame): DataFrame containing input data.

        Returns:
        - Tuple[pd.DataFrame, np.ndarray]: DataFrame with predictions and the raw prediction array.

        Raises:
        - ValueError: If the model with the specified sequence does not exist.
        - Exception: If there's an error processing the data or making predictions.
        """
        try:
            mh = self.model_manager.get_model(model_sequence)
            if mh is None:
                raise ValueError(f"Model with sequence {model_sequence} not found")

            processed_data, features_to_train = mh.inherited_preprocessor(
                input_df[features],
                self.numerical_features,
                self.category_sequence,
                self.unique_elements_categories,
            )
            return mh.make_predictions(processed_data, features_to_train)
        except Exception as e:
            print(f"Error processing data with model sequence {model_sequence}: {e}")
            return None
