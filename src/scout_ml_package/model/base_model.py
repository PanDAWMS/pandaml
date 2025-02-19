# # src/scout_ml_package/model/base_model.py
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    MaxPooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from scout_ml_package.data import NewDataPreprocessor, TrainingDataPreprocessor

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class DeviceInfo:
    @staticmethod
    def print_device_info():
        """Print information about available GPUs and CPUs."""
        print("Checking available devices...")
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"Number of GPUs available: {len(gpus)}")
            for gpu in gpus:
                print(f" - GPU: {gpu}")
        else:
            print("No GPUs found. Using CPU instead.")

        cpus = tf.config.list_physical_devices("CPU")
        print(f"Number of logical CPUs available: {len(cpus)}")
        for cpu in cpus:
            print(f" - CPU: {cpu}")


class MultiOutputModel:
    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        loss_function: str = "mse",
        regularizer: tf.keras.regularizers.Regularizer = l2(0.01),
        optimizer: tf.keras.optimizers.Optimizer = Adam(),
    ):
        """
        Initialize the MultiOutputModel.

        Args:
            input_shape (int): The input shape of the model.
            output_shape (int): The output shape of the model.
            loss_function (str): The loss function to use. Defaults to 'mse'.
            regularizer (tf.keras.regularizers.Regularizer): The regularizer to use. Defaults to l2(0.01).
            optimizer (tf.keras.optimizers.Optimizer): The optimizer to use. Defaults to Adam().
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.loss_function = self.get_loss_function(loss_function)
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.model = None

    def build_cputime(self) -> Model:
        """Build and compile the model for CPU time prediction."""
        inputs = Input(shape=(self.input_shape,))
        x = tf.keras.layers.Reshape((self.input_shape, 1))(inputs)
        x = self._add_conv_block(
            x, filters=128, kernel_size=7, activation="relu", pool_size=2
        )
        x = self._add_conv_block(
            x, filters=32, kernel_size=3, activation="relu", pool_size=2
        )
        x = Flatten()(x)
        x = self._add_dense_block(x, units=512, dropout_rate=0.5, activation="relu")
        x = self._add_dense_block(x, units=256, dropout_rate=0.4, activation="relu")
        x = self._add_dense_block(x, units=64, dropout_rate=0.3, activation="sigmoid")
        outputs = Dense(self.output_shape, activation="relu")(x)

        model = Model(inputs, outputs)
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=["mean_absolute_error", "mean_squared_error"],
        )
        self.model = model
        return model

    def custom_activation_low(self, x):
        return tf.clip_by_value(x, 0.4, 8.5)

    def build_cputime_low(self) -> Model:
        """Build and compile the model for CPU time prediction."""
        inputs = Input(shape=(self.input_shape,))
        x = tf.keras.layers.Reshape((self.input_shape, 1))(inputs)
        x = self._add_conv_block(
            x, filters=128, kernel_size=3, activation="relu", pool_size=2
        )
        x = Flatten()(x)
        x = self._add_dense_block(x, units=128, dropout_rate=0.4, activation="swish")
        x = self._add_dense_block(x, units=64, dropout_rate=0.3, activation="relu")
        outputs = Dense(self.output_shape)(x)
        outputs = tf.keras.layers.Lambda(self.custom_activation_low)(outputs)
        model = Model(inputs, outputs)
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=["RootMeanSquaredError", "mean_squared_error"],
        )
        self.model = model
        return model

    def weighted_mae(self, y_true, y_pred):
        mae = tf.abs(y_true - y_pred)
        # Progressive weighting for different ranges
        weights = tf.where(
            y_true > 3000,
            4.0,
            tf.where(y_true > 2000, 2.5, tf.where(y_true > 1000, 2.0, 1.0)),
        )
        return tf.reduce_mean(weights * mae)

    def custom_activation_high(self, x):
        return tf.clip_by_value(x, 10, 5000)

    def build_cputime_high0(self) -> Model:
        """Build and compile the model for CPU time prediction."""
        inputs = Input(shape=(self.input_shape,))
        x = tf.keras.layers.Reshape((self.input_shape, 1))(inputs)
        x = self._add_conv_block(
            x, filters=512, kernel_size=3, activation="swish", pool_size=2
        )
        x = Flatten()(x)
        x = self._add_dense_block(x, units=512, dropout_rate=0.4, activation="swish")
        x = self._add_dense_block(x, units=256, dropout_rate=0.3, activation="relu")
        x = self._add_dense_block(x, units=128, dropout_rate=0.3, activation="relu")
        outputs = Dense(self.output_shape, activation="linear")(x)
        outputs = tf.keras.layers.Lambda(self.custom_activation_high)(outputs)
        model = Model(inputs, outputs)
        model.compile(
            # tf.keras.optimizers.Adam(learning_rate=0.001), #
            optimizer=self.optimizer,
            loss=self.weighted_mae,  # self.loss_function,#self.weighted_mae, #
            metrics=["RootMeanSquaredError"],
        )
        self.model = model
        return model

    def build_cputime_high(self) -> Model:
        """Build and compile the model for CPU time prediction."""
        inputs = Input(shape=(self.input_shape,))
        x = tf.keras.layers.Reshape((self.input_shape, 1))(inputs)

        x = self._add_conv_block(
            x, filters=512, kernel_size=5, activation="relu", pool_size=2
        )
        x = self._add_conv_block(
            x, filters=128, kernel_size=3, activation="relu", pool_size=2
        )

        x = Flatten()(x)
        x = self._add_dense_block(x, units=512, dropout_rate=0.5, activation="relu")

        x = self._add_dense_block(x, units=256, dropout_rate=0.5, activation="relu")
        x = self._add_dense_block(x, units=128, dropout_rate=0.3, activation="relu")
        outputs = Dense(self.output_shape, activation="linear")(x)
        model = Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # self.optimizer,#
            loss=self.loss_function,
            metrics=["RootMeanSquaredError"],
        )
        self.model = model
        return model

    def test_build(self) -> Model:
        inputs = Input(shape=(self.input_shape,))
        x = tf.keras.layers.Reshape((self.input_shape, 1))(inputs)

        x = self._add_conv_block(
            x, filters=256, kernel_size=3, activation="elu", pool_size=2
        )
        x = Flatten()(x)
        # x = self._add_dense_block(x, units=256, dropout_rate=0.4, activation='elu')

        outputs = Dense(self.output_shape, activation="relu")(x)
        model = Model(inputs, outputs)

        model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=["mean_absolute_error"],
        )
        self.model = model
        return model

    def build_ramcount(self) -> Model:
        inputs = Input(shape=(self.input_shape,))
        x = tf.keras.layers.Reshape((self.input_shape, 1))(inputs)
        x = self._add_conv_block(
            x, filters=256, kernel_size=3, activation="relu", pool_size=2
        )
        x = self._add_conv_block(
            x, filters=128, kernel_size=5, activation="relu", pool_size=2
        )
        x = Flatten()(x)
        x = self._add_dense_block(x, units=256, dropout_rate=0.5, activation="relu")
        x = self._add_dense_block(x, units=128, dropout_rate=0.3, activation="relu")
        outputs = Dense(self.output_shape)(x)
        model = Model(inputs, outputs)
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=["mean_absolute_error"],
        )
        self.model = model
        return model

    def custom_loss_cpu_eff(self, y_true, y_pred):
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        range_penalty = tf.reduce_mean(
            tf.maximum(0.0, 1.0 - y_pred) + tf.maximum(0.0, y_pred - 99.0)
        )
        return mse + 10.0 * range_penalty

    def build_cpu_eff(self) -> Model:
        inputs = Input(shape=(self.input_shape,))
        x = tf.keras.layers.Reshape((self.input_shape, 1))(inputs)
        x = self._add_conv_block(
            x, filters=256, kernel_size=3, activation="relu", pool_size=2
        )
        x = self._add_conv_block(
            x, filters=128, kernel_size=3, activation="relu", pool_size=2
        )
        x = Flatten()(x)
        x = self._add_dense_block(x, units=512, dropout_rate=0.4, activation="relu")
        x = self._add_dense_block(x, units=256, dropout_rate=0.3, activation="relu")
        x = Dense(1)(x)
        outputs = Lambda(lambda x: tf.clip_by_value(x, 1, 99))(x)
        model = Model(inputs, outputs)

        model.compile(
            optimizer=self.optimizer,
            loss=self.custom_loss_cpu_eff,
            metrics=["RootMeanSquaredError", "mean_absolute_error"],
        )
        self.model = model
        return model

    def build_io(self) -> Model:
        inputs = Input(shape=(self.input_shape,))
        x = tf.keras.layers.Reshape((self.input_shape, 1))(inputs)

        # Convolutional layers with increasing complexity
        x = self._add_conv_block(x, filters=512, kernel_size=7)
        x = self._add_conv_block(x, filters=128, kernel_size=3)
        x = Flatten()(x)
        # x = self._add_dense_block(x, units=512, dropout_rate=0.5,act='sigmoid')
        x = self._add_dense_block(x, units=256, dropout_rate=0.4, activation="sigmoid")
        x = self._add_dense_block(x, units=64, dropout_rate=0.3, activation="relu")

        # Use output_shape here
        outputs = Dense(self.output_shape, activation="relu")(x)
        model = Model(inputs, outputs)
        model.compile(
            optimizer=self.optimizer,
            loss=self.loss_function,
            metrics=["accuracy"],
        )

        self.model = model
        return model

    def _add_conv_block(
        self,
        x: tf.Tensor,
        filters: int,
        kernel_size: int,
        activation: str = "relu",
        pool_size: int = 2,
    ) -> tf.Tensor:
        """Add a convolutional block to the model."""
        x = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation=activation,
            padding="same",
        )(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=pool_size)(x)
        return x

    def _add_dense_block(
        self,
        x: tf.Tensor,
        units: int,
        dropout_rate: float,
        activation: str = "relu",
    ) -> tf.Tensor:
        """Add a dense block to the model."""
        x = Dense(units, activation=activation, kernel_regularizer=self.regularizer)(x)
        x = Dropout(dropout_rate)(x)
        x = BatchNormalization()(x)
        return x

    @staticmethod
    def get_loss_function(loss_function: str) -> Union[str, Callable]:
        """Get the loss function based on the provided string."""
        loss_functions = {
            "mse": tf.keras.losses.MeanSquaredError(),
            "mae": tf.keras.losses.MeanAbsoluteError(),
            "rmse": lambda y_true, y_pred: tf.sqrt(
                tf.reduce_mean(tf.square(y_true - y_pred))
            ),
            "huber": tf.keras.losses.Huber(delta=7.0),
            "categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy(),
            "binary_crossentropy": tf.keras.losses.BinaryCrossentropy(),
        }
        if loss_function not in loss_functions:
            raise ValueError(f"Unknown loss function: {loss_function}")
        return loss_functions[loss_function]


class ModelTrainer:
    def __init__(
        self,
        model_class: type,
        input_shape: int,
        loss_function: str = "mse",
        metrics: Optional[List[Union[str, Callable]]] = None,
        build_function: str = "build",
        *args,
        **kwargs,
    ):
        """
        Initialize the ModelTrainer.

        Args:
            model_class (type): The class of the model to train.
            input_shape (int): The input shape of the model.
            loss_function (str): The loss function to use. Defaults to 'mse'.
            metrics (Optional[List[Union[str, Callable]]]): The metrics to use for evaluation. Defaults to None.
            build_function (str): The name of the build function to use. Defaults to 'build'.
            *args: Additional positional arguments for the model class.
            **kwargs: Additional keyword arguments for the model class.
        """
        self.model_instance = model_class(
            input_shape, loss_function=loss_function, *args, **kwargs
        )
        self.loss_function = loss_function
        self.metrics = metrics or []
        self.history = None
        self.build_function = build_function
        DeviceInfo.print_device_info()

    def train(
        self,
        X_train: tf.Tensor,
        y_train: tf.Tensor,
        X_val: tf.Tensor,
        y_val: tf.Tensor,
        epochs: int = 50,
        batch_size: int = 32,
        build_function_name: str = None,
    ) -> Tuple[Model, tf.keras.callbacks.History]:
        """
        Train the model.

        Args:
            X_train (tf.Tensor): The training input data.
            y_train (tf.Tensor): The training target data.
            X_val (tf.Tensor): The validation input data.
            y_val (tf.Tensor): The validation target data.
            epochs (int): The number of epochs to train for. Defaults to 50.
            batch_size (int): The batch size to use for training. Defaults to 32.
            build_function_name (str): The name of the build function to use. Defaults to None.

        Returns:
            Tuple[Model, tf.keras.callbacks.History]: The trained model and training history.
        """
        build_function_name = build_function_name or self.build_function
        strategy = (
            tf.distribute.MirroredStrategy()
            if tf.config.list_physical_devices("GPU")
            else tf.distribute.get_strategy()
        )

        with strategy.scope():
            model = getattr(self.model_instance, build_function_name)()
            model.compile(
                optimizer=self.model_instance.optimizer,
                loss=self.loss_function,
                metrics=self.metrics,
            )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.4, patience=5, min_lr=1e-6
        )
        self.history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
        )

        return model, self.history


class TrainedModel:
    def __init__(self, model: Model, model_name: str):
        """
        Initialize the TrainedModel.

        Args:
            model (Model): The trained model.
            model_name (str): The name of the model.
        """
        self.model = model
        self.model_name = model_name

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate the model's performance.

        Args:
            X_test (pd.DataFrame): The test input data.
            y_test (pd.DataFrame): The test target data.

        Returns:
            pd.DataFrame: A DataFrame containing evaluation metrics.
        """
        y_pred = self.predict(X_test)
        metrics = {}

        for i in range(y_pred.shape[1]):
            mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            metrics[f"Target {i + 1} (MSE)"] = mse
            metrics[f"Target {i + 1} (RMSE)"] = rmse
            metrics[f"Target {i + 1} (MAE)"] = mae
            metrics[f"Target {i + 1} (R²)"] = r2

        overall_mse = mean_squared_error(y_test, y_pred, multioutput="uniform_average")
        overall_rmse = np.sqrt(overall_mse)
        overall_mae = mean_absolute_error(y_test, y_pred, multioutput="uniform_average")
        overall_r2 = r2_score(y_test, y_pred, multioutput="uniform_average")
        metrics["Overall (MSE)"] = overall_mse
        metrics["Overall (RMSE)"] = overall_rmse
        metrics["Overall (MAE)"] = overall_mae
        metrics["Overall (R²)"] = overall_r2

        return pd.DataFrame(metrics.items(), columns=["Metric", "Value"])

    def save(self, save_path: str):
        """Save the trained model."""
        self.model.save(f"{save_path}/{self.model_name}.keras")
        print(f"Model '{self.model_name}' saved at {save_path}/{self.model_name}.keras")


class ModelPipeline:
    def __init__(
        self,
        training_data: pd.DataFrame,
        selected_columns: List[str],
        test_size: float = 0.15,
    ):
        """
        Initialize the ModelPipeline.

        Args:
            training_data (pd.DataFrame): The training data.
            selected_columns (List[str]): List of columns to use for training.
            test_size (float): Proportion of data to use for testing. Defaults to 0.15.
        """
        self.training_data = training_data
        self.selected_columns = selected_columns
        self.test_size = test_size

    def print_X_train_shape(self, X_train: pd.DataFrame):
        """Print the shape of X_train."""
        print("Shape of X_train:", X_train.shape)

    def split_data(self):
        """Split the data into training and testing sets."""
        from scout_ml_package.data.data_manager import DataSplitter

        splitter = DataSplitter(self.training_data, self.selected_columns)
        self.train_df, self.test_df = splitter.split_data(test_size=self.test_size)

    def preprocess_data(
        self,
        model_target,
        numerical_features,
        categorical_features,
        category_list,
    ):
        self.model_target = model_target
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.features = numerical_features + categorical_features
        self.category_list = category_list

        training_preprocessor = TrainingDataPreprocessor()
        self.processed_train_data, self.encoded_columns, self.fitted_scaler = (
            training_preprocessor.preprocess(
                self.train_df,
                self.numerical_features,
                self.categorical_features,
                self.category_list,
            )
        )

        # After processing data, print the shape of X_train
        X_train = self.processed_train_data[
            self.encoded_columns + self.numerical_features
        ]
        self.print_X_train_shape(X_train)

    def preprocess_new_data(self, test_df, future_data):
        new_data_preprocessor = NewDataPreprocessor()
        self.processed_test_data = new_data_preprocessor.preprocess(
            test_df,
            self.numerical_features,
            self.categorical_features,
            self.category_list,
            self.fitted_scaler,
            self.encoded_columns + self.model_target + ["JEDITASKID"],
        )

        self.processed_future_data = new_data_preprocessor.preprocess(
            future_data,
            self.numerical_features,
            self.categorical_features,
            self.category_list,
            self.fitted_scaler,
            self.encoded_columns + self.model_target + ["JEDITASKID"],
        )
