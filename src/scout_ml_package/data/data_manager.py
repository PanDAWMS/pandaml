# # src/scout_ml_package/data/data_manager.py

from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


class HistoricalDataProcessor:
    def __init__(self, task_data_path: str, additional_data_path: str = None):
        """
        Initialize the HistoricalDataProcessor with paths to historical Parquet files.

        Args:
            task_data_path (str): Path to the task data file.
            additional_data_path (str, optional): Path to additional historical data file to merge with.
        """
        self.task_data = pd.read_parquet(task_data_path)
        self.additional_data = (
            pd.read_parquet(additional_data_path) if additional_data_path else None
        )
        self.merged_data = None

    def filtered_data(self) -> pd.DataFrame:
        """
        Process and filter the data.

        Returns:
            pd.DataFrame: Filtered and processed data.
        """
        if self.additional_data is None:
            return self.task_data

        self.task_data["Process_Head"] = (
            self.task_data["TASKNAME"]
            .str.split(".")
            .str[2]
            .str.replace(r"[_\.]", " ", regex=True)
        )
        self.task_data["Tags"] = (
            self.task_data["TASKNAME"]
            .str.split(".")
            .str[-1]
            .str.replace(r"[_\.]", " ", regex=True)
        )

        self.merged_data = pd.merge(
            self.additional_data,
            self.task_data[["JEDITASKID", "Process_Head", "Tags"]],
            on="JEDITASKID",
            how="left",
        )

        self.merged_data = self.merged_data.drop(
            columns=["PROCESSINGTYPE", "P50", "F50", "PRED_RAM", "TRANSHOME"],
            errors="ignore",
        )
        self.merged_data["IOIntensity"] = self.merged_data["IOINTENSITY"].apply(
            lambda x: "low" if x < 500 else "high"
        )

        return self.merged_data[
            self.merged_data["PRODSOURCELABEL"].isin(["user", "managed"])
            & (self.merged_data["RAMCOUNT"] > 100)
            & (self.merged_data["RAMCOUNT"] < 6000)
            & (self.merged_data["CPU_EFF"] > 30)
            & (self.merged_data["CPU_EFF"] < 100)
            & (self.merged_data["cputime_HS"] > 0.1)
            & (self.merged_data["cputime_HS"] < 3000)
        ]


class DataSplitter:
    def __init__(self, filtered_data: pd.DataFrame, selected_columns: List[str]):
        """
        Initialize the DataSplitter with filtered data and selected columns.

        Args:
            filtered_data (pd.DataFrame): The DataFrame obtained after processing.
            selected_columns (List[str]): List of column names to select from the DataFrame.
        """
        self.merged_data = filtered_data
        self.selected_columns = selected_columns

    def split_data(
        self, test_size: float = 0.30, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the processed data into training and testing datasets.

        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Controls the shuffling applied to the data before applying the split.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Training and testing datasets.

        Raises:
            ValueError: If data has not been processed or 'JEDITASKID' is not in selected columns.
        """
        if self.merged_data is None or self.merged_data.empty:
            raise ValueError(
                "Data has not been processed yet. Please provide valid processed data."
            )

        if "JEDITASKID" not in self.selected_columns:
            raise ValueError("The selected columns must include 'JEDITASKID'.")

        df_train, df_test = train_test_split(
            self.merged_data[self.selected_columns].dropna(),
            test_size=test_size,
            random_state=random_state,
        )

        return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


class ModelTrainingInput:
    def __init__(
        self,
        df_train: pd.DataFrame,
        features: List[str],
        target_cols: List[str],
    ):
        """
        Initialize the ModelTrainingInput with training dataset, features, and target columns.

        Args:
            df_train (pd.DataFrame): The training dataset.
            features (List[str]): List of feature column names.
            target_cols (List[str]): List of target column names.
        """
        self.df_train = df_train
        self.features = features
        self.target_cols = target_cols

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare the data for model training.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Feature data and target data for training.
        """
        X_train = self.df_train[self.features]
        y_train = self.df_train[self.target_cols]
        return X_train, y_train


class CategoricalEncoder:
    @staticmethod
    def get_unique_values(df: pd.DataFrame, target_columns: list) -> list:
        """
        Retrieve the unique values for specified columns in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame from which unique values are extracted.
            target_columns (list): List of column names to extract unique values from.

        Returns:
            list: A list containing unique values for each specified column.
        """
        return [df[col].unique().tolist() for col in target_columns]

    @staticmethod
    def one_hot_encode(
        df: pd.DataFrame, columns_to_encode: list, category_list: list
    ) -> tuple:
        """
        Perform one-hot encoding on specified categorical columns in a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the data to be encoded.
            columns_to_encode (list): List of column names to encode.
            category_list (list): List of unique categories for each column.

        Returns:
            tuple: A DataFrame with the original data and the new one-hot encoded features,
                   and a list of new feature names.
        """
        encoder = OneHotEncoder(categories=category_list, sparse_output=False)
        encoded_features = encoder.fit_transform(df[columns_to_encode])
        encoded_feature_names = encoder.get_feature_names_out(columns_to_encode)
        encoded_df = pd.DataFrame(
            encoded_features, columns=encoded_feature_names, index=df.index
        )
        encoded_df = pd.concat([df, encoded_df], axis=1)
        return encoded_df, encoded_feature_names.tolist()


class BaseDataPreprocessor:
    def __init__(self):
        """Initialize the BaseDataPreprocessor."""
        self.scaler = MinMaxScaler()

    def _fit_and_transform(self, df: pd.DataFrame, numerical_features: list):
        """Fit and transform numerical features."""
        if numerical_features:
            self.scaler.fit(df[numerical_features])
            df[numerical_features] = self.scaler.transform(df[numerical_features])
        return df

    def _encode_features(
        self, df: pd.DataFrame, categorical_features: list, category_list: list
    ):
        """Encode categorical features."""
        return CategoricalEncoder.one_hot_encode(df, categorical_features, category_list)


class TrainingDataPreprocessor(BaseDataPreprocessor):
    def preprocess(
        self,
        df: pd.DataFrame,
        numerical_features: list,
        categorical_features: list,
        category_list: list,
    ) -> tuple:
        """
        Preprocess training data.

        Args:
            df (pd.DataFrame): The training data.
            numerical_features (list): List of numerical feature names.
            categorical_features (list): List of categorical feature names.
            category_list (list): List of unique categories for each categorical feature.

        Returns:
            tuple: Preprocessed DataFrame, encoded column names, and fitted scaler.
        """
        df = self._fit_and_transform(df, numerical_features)
        df, encoded_columns = self._encode_features(
            df, categorical_features, category_list
        )
        return df, encoded_columns, self.scaler


class NewDataPreprocessor(BaseDataPreprocessor):
    def preprocess(
        self,
        new_data: pd.DataFrame,
        numerical_features: list,
        categorical_features: list,
        category_list: list,
        scaler: MinMaxScaler,
        encoded_columns: list,
    ) -> pd.DataFrame:
        """
        Preprocess new data for predictions using the fitted scaler from training data.

        Args:
            new_data (pd.DataFrame): The new data to preprocess.
            numerical_features (list): List of numerical feature names.
            categorical_features (list): List of categorical feature names.
            category_list (list): List of unique categories for each categorical feature.
            scaler (MinMaxScaler): Fitted scaler from training data.
            encoded_columns (list): List of encoded column names.

        Returns:
            pd.DataFrame: Preprocessed new data.
        """
        new_data[numerical_features] = scaler.transform(new_data[numerical_features])
        new_data, _ = self._encode_features(new_data, categorical_features, category_list)
        for col in encoded_columns:
            if col not in new_data.columns:
                new_data[col] = 0
        return new_data[encoded_columns + numerical_features]


class LiveDataPreprocessor(BaseDataPreprocessor):
    def preprocess(
        self,
        live_data: pd.DataFrame,
        numerical_features: list,
        categorical_features: list,
        category_list: list,
        scaler: MinMaxScaler,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preprocess live data for predictions using the fitted scaler from training data.

        Args:
            live_data (pd.DataFrame): The live data to preprocess.
            numerical_features (list): List of numerical feature names.
            categorical_features (list): List of categorical feature names.
            category_list (list): List of unique categories for each categorical feature.
            scaler (MinMaxScaler): Fitted scaler from training data.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Preprocessed live data and list of encoded column names.
        """
        live_data = live_data.copy()
        live_data[numerical_features] = scaler.transform(live_data[numerical_features])
        live_data, encoded_columns = self._encode_features(
            live_data, categorical_features, category_list
        )
        return live_data[encoded_columns + numerical_features], encoded_columns
