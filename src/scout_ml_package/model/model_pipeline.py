# src/scout_ml_package/model/model_pipeline.py
import os
import re

import joblib
import numpy as np
import tensorflow as tf
from keras.layers import TFSMLayer

from scout_ml_package.data import (
    LiveDataPreprocessor,
    NewDataPreprocessor,
    TrainingDataPreprocessor,
)
from scout_ml_package.model import MultiOutputModel
from scout_ml_package.model.base_model import ModelTrainer  # , ModelPipeline

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class TrainingPipeline:
    def __init__(self, numerical_features, categorical_features, category_list, m_target):
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.category_list = category_list
        self.model_target = m_target

    def preprocess_data(self, train_df, test_df, future_data):
        training_preprocessor = TrainingDataPreprocessor()
        processed_train_data, encoded_columns, fitted_scaler = (
            training_preprocessor.preprocess(
                train_df,
                self.numerical_features,
                self.categorical_features,
                self.category_list,
            )
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
        processed_train_data,
        processed_test_data,
        features_to_train,
        build_function_name,
        epoch,
        batch,
    ):

        X_train, y_train = (
            processed_train_data[features_to_train],
            processed_train_data[self.model_target],
        )
        X_val, y_val = (
            processed_test_data[features_to_train],
            processed_test_data[self.model_target],
        )

        # print("y_train shape:", y_train.shape)
        # print("X_train shape:", X_train.columns)
        # 1D output handling
        output_shape = y_train.shape[1] if y_train.ndim > 1 else 1
        print("Output shape:", output_shape)

        trainer = ModelTrainer(
            MultiOutputModel,
            input_shape=X_train.shape[1],
            output_shape=output_shape,
            loss_function="mse",
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
            build_function="test_build",
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
        processed_train_data,
        processed_test_data,
        features_to_train,
        build_function_name,
        epoch,
        batch,
        model_type="binary",
    ):
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
            output_shape = len(
                y_train.unique()
            )  # for one-hot encoded classes, check y_train.shape[1]
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

        model, history = trainer.train(
            X_train, y_train, X_val, y_val, epochs=epoch, batch_size=batch
        )
        return model

    def regression_prediction(
        self, trained_model, processed_future_data, features_to_train
    ):
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
        self, trained_model, processed_future_data, features_to_train
    ):
        # Prepare test data
        X_test = processed_future_data[features_to_train]
        y_test = processed_future_data[self.model_target]

        # Predict class labels
        y_pred = trained_model.predict(X_test)

        # Convert numerical predictions back to 'low' or 'high'
        y_pred_text = np.where(y_pred == 0, "low", "high")

        # Create predictions DataFrame
        pred_names = [f"Predicted_{element}" for element in self.model_target]
        predicted_df = processed_future_data.copy()
        predicted_df[self.model_target] = (
            y_test  # Keep actual values (which are already 0 or 1)
        )
        predicted_df[pred_names] = (
            y_pred_text  # Store predicted values as 'low' or 'high'
        )

        # Convert actual values back to 'low' or 'high' for consistency
        predicted_df[self.model_target] = predicted_df[self.model_target].replace(
            {0: "low", 1: "high"}
        )
        return predicted_df, y_pred_text


import re
import pandas as pd


class ColumnTransformer:
    def convert_processingtype(self, processingtype):
        """
        Convert PROCESSINGTYPE to 'P'.

        :param processingtype: Input processing type.
        :return: Converted processing type.
        """
        if processingtype is not None and re.search(r"-.*-", processingtype):
            return "-".join(processingtype.split("-")[-2:])
        return processingtype

    def convert_transhome(self, transhome):
        """
        Convert TRANSHOME to 'F'.

        :param transhome: Input transhome.
        :return: Converted transhome.
        """
        if transhome is None:
            return None

        if "AnalysisBase" in transhome:
            return "AnalysisBase"
        elif "AnalysisTransforms-" in transhome:
            part_after_dash = transhome.split("-")[1]
            return part_after_dash.split("_")[0]
        elif "/" in transhome:
            return transhome.split("/")[0]
        else:
            return transhome.split("-")[0]

    def convert_corecount(self, corecount):
        """
        Convert CORECOUNT to 'Core'.

        :param corecount: Input core count.
        :return: Converted core count.
        """
        return "S" if corecount == 1 else "M"

    def transform_features(
        self, df: pd.DataFrame, selected_columns: list = None
    ) -> pd.DataFrame:
        """
        Apply transformations to the input DataFrame.

        :param df: Input DataFrame.
        :param selected_columns: Optional list of columns to select after transformation. If None, returns all columns.
        :return: Transformed DataFrame.
        """
        df["P"] = df["PROCESSINGTYPE"].apply(self.convert_processingtype)
        df["F"] = df["TRANSHOME"].apply(self.convert_transhome)
        df["CORE"] = df["CORECOUNT"].apply(self.convert_corecount)

        if selected_columns is not None:
            return df[selected_columns]
        else:
            return df


# PredictionPipeline--> ModelManager--> ModelHandlerInProd--> TrainingPipeline--> ModelTrainer
class ModelHandlerInProd:
    def __init__(self, model_sequence: str, target_name: str):
        self.model_sequence = model_sequence
        self.target_name = target_name
        self.model = None
        self.scaler = None

    def load_model_and_scaler(self, base_path: str = None):
        """
        Load the model and scaler using an absolute path.

        Args:
            base_path (str): The base directory where models are stored. If None, defaults to the current working directory.
        """
        try:
            # Ensure base_path is absolute; default to current working directory if not provided
            if base_path is None:
                base_path = os.getcwd()
            model_storage_path = os.path.abspath(
                os.path.join(base_path, f"ModelStorage/model{self.model_sequence}/")
            )

            # Load scaler and model
            self.scaler = joblib.load(os.path.join(model_storage_path, "scaler.pkl"))
            model_name = f"model{self.model_sequence}_{self.target_name}"
            model_full_path = os.path.join(model_storage_path, model_name)

            # Assuming TFSMLayer is a custom class for loading TensorFlow models
            self.model = TFSMLayer(model_full_path, call_endpoint="serving_default")

            print(f"Model and scaler for {self.target_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading model and scaler: {e}")

    def preprocess_data(
        self,
        df,
        numerical_features,
        category_sequence,
        unique_elements_categories,
    ):
        """Preprocess the data using the loaded scaler."""
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

    # def transform_features(self, df):
    #     # Convert PROCESSINGTYPE to 'P'
    #     def convert_processingtype(processingtype):
    #         if processingtype is not None and re.search(r"-.*-", processingtype):
    #             return "-".join(processingtype.split("-")[-2:])
    #         return processingtype
    #
    #     # Convert TRANSHOME to 'F'
    #     def convert_transhome(transhome):
    #         # Check if transhome is None
    #         if transhome is None:
    #             return None  # or handle as needed
    #
    #         if "AnalysisBase" in transhome:
    #             return "AnalysisBase"
    #         elif "AnalysisTransforms-" in transhome:
    #             # Extract the part after 'AnalysisTransforms-'
    #             part_after_dash = transhome.split("-")[1]
    #             return part_after_dash.split("_")[
    #                 0
    #             ]  # Assuming you want the first part before any underscore
    #         elif "/" in transhome:
    #             # Handle cases like AthGeneration/2022-11-09T1600
    #             return transhome.split("/")[0]
    #         else:
    #             # For all other cases, split by '-', return the first segment
    #             return transhome.split("-")[0]
    #
    #     # Convert CORECOUNT to 'Core'
    #     def convert_corecount(corecount):
    #         return "S" if corecount == 1 else "M"
    #
    #     # Apply transformations
    #     df["P"] = df["PROCESSINGTYPE"].apply(convert_processingtype)
    #     df["F"] = df["TRANSHOME"].apply(convert_transhome)
    #     df["CORE"] = df["CORECOUNT"].apply(convert_corecount)
    #
    #     # Return selected columns
    #     numerical_features = [
    #         "TOTAL_NFILES",
    #         "TOTAL_NEVENTS",
    #         "DISTINCT_DATASETNAME_COUNT",
    #     ]
    #     categorical_features = [
    #         "PRODSOURCELABEL",
    #         "P",
    #         "F",
    #         "CORE",
    #         "CPUTIMEUNIT",
    #     ]
    #     ["JEDITASKID"] + numerical_features + categorical_features
    #
    #     return df

    def make_predictions(self, df, features_to_train):
        """Make predictions using the loaded model."""
        predictions = self.model(df[features_to_train])
        # Extract the tensor from the predictions dictionary
        predicted_tensor = predictions[
            "output_0"
        ]  # Adjust the key based on actual output
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


############################


class ModelManager:
    def __init__(self, base_path):
        self.models = {}
        self.base_path = base_path

    def load_models(self):
        model_configs = [
            ("1", "ramcount"),
            ("2", "cputime_low"),
            ("3", "cputime_high"),
            ("4", "cpu_eff"),
            ("5", "io"),
        ]
        for sequence, target_name in model_configs:
            model = ModelHandlerInProd(model_sequence=sequence, target_name=target_name)
            model.load_model_and_scaler(self.base_path)
            self.models[sequence] = model
        # logger.info("All models loaded successfully")

    def get_model(self, sequence):
        return self.models.get(sequence)


class PredictionPipeline:

    def __init__(self, model_manager):
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
                "simul",
                "jedi-run",
                "deriv",
                "pile",
                "reprocessing",
                "merge",
                "jedi-athena",
                "athena-trf",
                "evgen",
                "eventIndex",
                "others",
                "recon",
            ],
            [
                "Athena",
                "AnalysisBase",
                "AthDerivation",
                "AthAnalysis",
                "AthGeneration",
                "AtlasOffline",
                "AthSimulation",
                "others",
                "MCProd",
            ],
            ["M", "S"],
            ["HS06sPerEvent", "mHS06sPerEvent"],
        ]

    # def transform_features(self, df):
    #     # Convert PROCESSINGTYPE to 'P'
    #     def convert_processingtype(processingtype):
    #         if processingtype is not None and re.search(r"-.*-", processingtype):
    #             return "-".join(processingtype.split("-")[-2:])
    #         return processingtype
    #
    #     # Convert TRANSHOME to 'F'
    #     def convert_transhome(transhome):
    #         # Check if transhome is None
    #         if transhome is None:
    #             return None  # or handle as needed
    #
    #         if "AnalysisBase" in transhome:
    #             return "AnalysisBase"
    #         elif "AnalysisTransforms-" in transhome:
    #             # Extract the part after 'AnalysisTransforms-'
    #             part_after_dash = transhome.split("-")[1]
    #             return part_after_dash.split("_")[
    #                 0
    #             ]  # Assuming you want the first part before any underscore
    #         elif "/" in transhome:
    #             # Handle cases like AthGeneration/2022-11-09T1600
    #             return transhome.split("/")[0]
    #         else:
    #             # For all other cases, split by '-', return the first segment
    #             return transhome.split("-")[0]
    #
    #     # Convert CORECOUNT to 'Core'
    #     def convert_corecount(corecount):
    #         return "S" if corecount == 1 else "M"
    #
    #     # Apply transformations
    #     df["P"] = df["PROCESSINGTYPE"].apply(convert_processingtype)
    #     df["F"] = df["TRANSHOME"].apply(convert_transhome)
    #     df["CORE"] = df["CORECOUNT"].apply(convert_corecount)
    #
    #     # Return selected columns
    #     numerical_features = [
    #         "TOTAL_NFILES",
    #         "TOTAL_NEVENTS",
    #         "DISTINCT_DATASETNAME_COUNT",
    #     ]
    #     categorical_features = [
    #         "PRODSOURCELABEL",
    #         "P",
    #         "F",
    #         "CORE",
    #         "CPUTIMEUNIT",
    #     ]
    #     ["JEDITASKID"] + numerical_features + categorical_features
    #
    #     return df

    def make_predictions_for_model(self, model_sequence, features, input_df):
        try:
            mh = self.model_manager.get_model(model_sequence)
            if mh is None:
                raise ValueError(f"Model with sequence {model_sequence} not found")

            processed_data, features_to_train = mh.transform_features(
                input_df[features],
                self.numerical_features,
                self.category_sequence,
                self.unique_elements_categories,
            )
            return mh.make_predictions(processed_data, features_to_train)
        except Exception as e:
            print(f"Error processing data with model sequence {model_sequence}: {e}")
            return None
