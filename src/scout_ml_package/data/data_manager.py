# # src/scout_ml_package/data/data_manager.py

import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Optional


class HistoricalDataProcessor:
    """
    Processes historical data by merging and filtering task data with additional data.

    Attributes:
    - task_data (pd.DataFrame): DataFrame containing task data.
    - additional_data (pd.DataFrame): Optional DataFrame containing additional historical data.
    - merged_data (pd.DataFrame): Merged DataFrame after processing.
    """

    def __init__(self, task_data_path: str, additional_data_path: Optional[str] = None):
        """
        Initializes the HistoricalDataProcessor with paths to historical Parquet files.

        Parameters:
        - task_data_path (str): Path to the task data file.
        - additional_data_path (str, optional): Path to additional historical data file to merge with. Defaults to None.

        Raises:
        - FileNotFoundError: If either of the specified files does not exist.
        """
        self.task_data = pd.read_parquet(task_data_path)
        self.additional_data = (
            pd.read_parquet(additional_data_path) if additional_data_path else None
        )
        self.merged_data = None

    def filtered_data(self) -> pd.DataFrame:
        """
        Processes and filters the data.

        Returns:
        - pd.DataFrame: Filtered and processed data.

        Notes:
        - If additional data is provided, it merges with task data and applies filters.
        - If no additional data is provided, it returns the task data as is.
        """
        if self.additional_data is None:
            return self.task_data

        # Extract process head and tags from task names
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

        # Merge task data with additional data
        self.merged_data = pd.merge(
            self.additional_data,
            self.task_data[["JEDITASKID", "Process_Head", "Tags"]],
            on="JEDITASKID",
            how="left",
        )

        # Drop unnecessary columns
        self.merged_data = self.merged_data.drop(
            columns=["PROCESSINGTYPE", "P50", "F50", "PRED_RAM", "TRANSHOME"],
            errors="ignore",
        )

        # Classify IO intensity
        self.merged_data["IOIntensity"] = self.merged_data["IOINTENSITY"].apply(
            lambda x: "low" if x < 500 else "high"
        )

        # Apply filters
        filtered_data = self.merged_data[
            self.merged_data["PRODSOURCELABEL"].isin(["user", "managed"])
            & (self.merged_data["RAMCOUNT"] > 100)
            & (self.merged_data["RAMCOUNT"] < 6000)
            & (self.merged_data["CPU_EFF"] > 30)
            & (self.merged_data["CPU_EFF"] < 100)
            & (self.merged_data["cputime_HS"] > 0.1)
            & (self.merged_data["cputime_HS"] < 3000)
        ]

        return filtered_data


class DataSplitter:
    """
    Splits processed data into training and testing datasets.

    Attributes:
    - merged_data (pd.DataFrame): Processed data to be split.
    - selected_columns (List[str]): List of column names to select from the data.
    """

    def __init__(self, filtered_data: pd.DataFrame, selected_columns: List[str]):
        """
        Initializes the DataSplitter with filtered data and selected columns.

        Parameters:
        - filtered_data (pd.DataFrame): The DataFrame obtained after processing.
        - selected_columns (List[str]): List of column names to select from the DataFrame.

        Raises:
        - TypeError: If filtered_data is not a DataFrame or selected_columns is not a list.
        """
        if not isinstance(filtered_data, pd.DataFrame):
            raise TypeError("filtered_data must be a pandas DataFrame")
        if not isinstance(selected_columns, list):
            raise TypeError("selected_columns must be a list of strings")

        self.merged_data = filtered_data
        self.selected_columns = selected_columns

    def split_data(
        self, test_size: float = 0.30, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the processed data into training and testing datasets.

        Parameters:
        - test_size (float): Proportion of the dataset to include in the test split. Defaults to 0.30.
        - random_state (int): Controls the shuffling applied to the data before applying the split. Defaults to 42.

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame]: Training and testing datasets.

        Raises:
        - ValueError: If data has not been processed or 'JEDITASKID' is not in selected columns.
        """
        if self.merged_data is None or self.merged_data.empty:
            raise ValueError(
                "Data has not been processed yet. Please provide valid processed data."
            )

        if "JEDITASKID" not in self.selected_columns:
            raise ValueError("The selected columns must include 'JEDITASKID'.")

        # Ensure selected columns exist in the data
        if not all(col in self.merged_data.columns for col in self.selected_columns):
            raise ValueError("All selected columns must exist in the data.")

        df_train, df_test = train_test_split(
            self.merged_data[self.selected_columns].dropna(),
            test_size=test_size,
            random_state=random_state,
        )

        return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


class ModelTrainingInput:
    """
    Prepares data for model training by separating features and target columns.

    Attributes:
    - df_train (pd.DataFrame): The training dataset.
    - features (List[str]): List of feature column names.
    - target_cols (List[str]): List of target column names.
    """

    def __init__(
        self,
        df_train: pd.DataFrame,
        features: List[str],
        target_cols: List[str],
    ):
        """
        Initializes the ModelTrainingInput with training dataset, features, and target columns.

        Parameters:
        - df_train (pd.DataFrame): The training dataset.
        - features (List[str]): List of feature column names.
        - target_cols (List[str]): List of target column names.

        Raises:
        - TypeError: If df_train is not a DataFrame or if features/target_cols are not lists.
        """
        if not isinstance(df_train, pd.DataFrame):
            raise TypeError("df_train must be a pandas DataFrame")
        if not isinstance(features, list) or not isinstance(target_cols, list):
            raise TypeError("features and target_cols must be lists of strings")

        self.df_train = df_train
        self.features = features
        self.target_cols = target_cols

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepares the data for model training by separating features and target columns.

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame]: Feature data and target data for training.

        Raises:
        - ValueError: If any feature or target column does not exist in the DataFrame.
        """
        # Check if all feature and target columns exist in the DataFrame
        if not all(
            col in self.df_train.columns for col in self.features + self.target_cols
        ):
            raise ValueError(
                "All feature and target columns must exist in the DataFrame."
            )

        X_train = self.df_train[self.features]
        y_train = self.df_train[self.target_cols]
        return X_train, y_train


class CategoricalEncoder:
    """
    Provides methods for handling categorical data encoding.

    Attributes:
    - None

    Methods:
    - get_unique_values: Retrieves unique values for specified columns in a DataFrame.
    - one_hot_encode: Performs one-hot encoding on categorical columns.
    """

    @staticmethod
    def get_unique_values(df: pd.DataFrame, target_columns: List[str]) -> List[List]:
        """
        Retrieves the unique values for specified columns in a DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame from which unique values are extracted.
        - target_columns (List[str]): List of column names to extract unique values from.

        Returns:
        - List[List]: A list containing unique values for each specified column.

        Raises:
        - TypeError: If df is not a DataFrame or if target_columns is not a list.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not isinstance(target_columns, list):
            raise TypeError("target_columns must be a list of strings")

        return [df[col].unique().tolist() for col in target_columns]

    @staticmethod
    def one_hot_encode(
        df: pd.DataFrame, columns_to_encode: List[str], category_list: List[List]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Performs one-hot encoding on specified categorical columns in a DataFrame.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing the data to be encoded.
        - columns_to_encode (List[str]): List of column names to encode.
        - category_list (List[List]): List of unique categories for each column.

        Returns:
        - Tuple[pd.DataFrame, List[str]]: A DataFrame with the original data and the new one-hot encoded features,
                                           and a list of new feature names.

        Raises:
        - TypeError: If df is not a DataFrame or if columns_to_encode/category_list are not lists.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not isinstance(columns_to_encode, list) or not isinstance(category_list, list):
            raise TypeError("columns_to_encode and category_list must be lists")

        encoder = OneHotEncoder(categories=category_list, sparse_output=False)
        encoded_features = encoder.fit_transform(df[columns_to_encode])
        encoded_feature_names = encoder.get_feature_names_out(columns_to_encode)
        encoded_df = pd.DataFrame(
            encoded_features, columns=encoded_feature_names, index=df.index
        )
        encoded_df = pd.concat([df, encoded_df], axis=1)
        return encoded_df, encoded_feature_names.tolist()


class BaseDataPreprocessor:
    """
    Base class for data preprocessing, providing methods for scaling and encoding.

    Attributes:
    - scaler (MinMaxScaler): Scaler for numerical features.
    """

    def __init__(self):
        """
        Initializes the BaseDataPreprocessor with a MinMaxScaler.
        """
        self.scaler = MinMaxScaler()

    def _fit_and_transform(
        self, df: pd.DataFrame, numerical_features: List[str]
    ) -> pd.DataFrame:
        """
        Fits and transforms numerical features using the scaler.

        Parameters:
        - df (pd.DataFrame): DataFrame containing numerical features.
        - numerical_features (List[str]): List of numerical feature names.

        Returns:
        - pd.DataFrame: DataFrame with scaled numerical features.

        Raises:
        - TypeError: If df is not a DataFrame or if numerical_features is not a list.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not isinstance(numerical_features, list):
            raise TypeError("numerical_features must be a list of strings")

        if numerical_features:
            self.scaler.fit(df[numerical_features])
            df[numerical_features] = self.scaler.transform(df[numerical_features])
        return df

    def _encode_features(
        self, df: pd.DataFrame, categorical_features: List[str], category_list: List[List]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Encodes categorical features using one-hot encoding.

        Parameters:
        - df (pd.DataFrame): DataFrame containing categorical features.
        - categorical_features (List[str]): List of categorical feature names.
        - category_list (List[List]): List of unique categories for each categorical feature.

        Returns:
        - Tuple[pd.DataFrame, List[str]]: DataFrame with encoded features and list of new feature names.

        Raises:
        - TypeError: If df is not a DataFrame or if categorical_features/category_list are not lists.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not isinstance(categorical_features, list) or not isinstance(
            category_list, list
        ):
            raise TypeError("categorical_features and category_list must be lists")

        return CategoricalEncoder.one_hot_encode(df, categorical_features, category_list)


class TrainingDataPreprocessor(BaseDataPreprocessor):
    """
    Preprocesses training data by scaling numerical features and encoding categorical features.

    Attributes:
    - None
    """

    def preprocess(
        self,
        df: pd.DataFrame,
        numerical_features: List[str],
        categorical_features: List[str],
        category_list: List[List],
    ) -> Tuple[pd.DataFrame, List[str], MinMaxScaler]:
        """
        Preprocesses training data.

        Parameters:
        - df (pd.DataFrame): The training data.
        - numerical_features (List[str]): List of numerical feature names.
        - categorical_features (List[str]): List of categorical feature names.
        - category_list (List[List]): List of unique categories for each categorical feature.

        Returns:
        - Tuple[pd.DataFrame, List[str], MinMaxScaler]: Preprocessed DataFrame, encoded column names, and fitted scaler.

        Raises:
        - TypeError: If df is not a DataFrame or if numerical_features/categorical_features/category_list are not lists.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if (
            not isinstance(numerical_features, list)
            or not isinstance(categorical_features, list)
            or not isinstance(category_list, list)
        ):
            raise TypeError(
                "numerical_features, categorical_features, and category_list must be lists"
            )

        df = self._fit_and_transform(df, numerical_features)
        df, encoded_columns = self._encode_features(
            df, categorical_features, category_list
        )
        return df, encoded_columns, self.scaler


class NewDataPreprocessor(BaseDataPreprocessor):
    """
    Preprocesses new data for predictions using a fitted scaler.

    Attributes:
    - None
    """

    def preprocess(
        self,
        new_data: pd.DataFrame,
        numerical_features: List[str],
        categorical_features: List[str],
        category_list: List[List],
        scaler: MinMaxScaler,
        encoded_columns: List[str],
    ) -> pd.DataFrame:
        """
        Preprocesses new data for predictions.

        Parameters:
        - new_data (pd.DataFrame): The new data to preprocess.
        - numerical_features (List[str]): List of numerical feature names.
        - categorical_features (List[str]): List of categorical feature names.
        - category_list (List[List]): List of unique categories for each categorical feature.
        - scaler (MinMaxScaler): Fitted scaler from training data.
        - encoded_columns (List[str]): List of encoded column names.

        Returns:
        - pd.DataFrame: Preprocessed new data.

        Raises:
        - TypeError: If new_data is not a DataFrame or if numerical_features/categorical_features/category_list/encoded_columns are not lists.
        """
        if not isinstance(new_data, pd.DataFrame):
            raise TypeError("new_data must be a pandas DataFrame")
        if (
            not isinstance(numerical_features, list)
            or not isinstance(categorical_features, list)
            or not isinstance(category_list, list)
            or not isinstance(encoded_columns, list)
        ):
            raise TypeError(
                "numerical_features, categorical_features, category_list, and encoded_columns must be lists"
            )

        new_data[numerical_features] = scaler.transform(new_data[numerical_features])
        new_data, _ = self._encode_features(new_data, categorical_features, category_list)
        for col in encoded_columns:
            if col not in new_data.columns:
                new_data[col] = 0
        return new_data[encoded_columns + numerical_features]


class LiveDataPreprocessor(BaseDataPreprocessor):
    """
    Preprocesses live data for predictions using a fitted scaler.

    Attributes:
    - None
    """

    def preprocess(
        self,
        live_data: pd.DataFrame,
        numerical_features: List[str],
        categorical_features: List[str],
        category_list: List[List],
        scaler: MinMaxScaler,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Preprocesses live data for predictions.

        Parameters:
        - live_data (pd.DataFrame): The live data to preprocess.
        - numerical_features (List[str]): List of numerical feature names.
        - categorical_features (List[str]): List of categorical feature names.
        - category_list (List[List]): List of unique categories for each categorical feature.
        - scaler (MinMaxScaler): Fitted scaler from training data.

        Returns:
        - Tuple[pd.DataFrame, List[str]]: Preprocessed live data and list of encoded column names.

        Raises:
        - TypeError: If live_data is not a DataFrame or if numerical_features/categorical_features/category_list are not lists.
        """
        if not isinstance(live_data, pd.DataFrame):
            raise TypeError("live_data must be a pandas DataFrame")
        if (
            not isinstance(numerical_features, list)
            or not isinstance(categorical_features, list)
            or not isinstance(category_list, list)
        ):
            raise TypeError(
                "numerical_features, categorical_features, and category_list must be lists"
            )

        live_data = live_data.copy()
        live_data[numerical_features] = scaler.transform(live_data[numerical_features])
        live_data, encoded_columns = self._encode_features(
            live_data, categorical_features, category_list
        )
        return live_data[encoded_columns + numerical_features], encoded_columns
