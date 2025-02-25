import logging
import time
from typing import List, Generator
import pandas as pd

from scout_ml_package.utils.logger import Logger

# Get the logger instance using the singleton pattern
logger = Logger(
    "demo_logger", "/data/model-data/logs", "prediction_v1.log", log_level=logging.DEBUG
).get_logger()


# class FakeListener:
#     def __init__(self, task_ids, delay=3):
#         self.task_ids = task_ids
#         self.delay = delay
#
#     def demo_task_listener(self):
#         """
#         A demo function that simulates a listener sending task IDs with a fixed delay.
#
#         Yields:
#         int: A task ID from the list.
#         """
#         for task_id in self.task_ids:
#             # Wait for the specified delay
#             time.sleep(self.delay)
#             yield task_id

#
# class DataValidator:
#     @classmethod
#     def check_predictions(cls, df, column, acceptable_ranges):
#         min_val, max_val = acceptable_ranges[column]
#         if df[column].min() < min_val or df[column].max() > max_val:
#             raise ValueError(
#                 f"Predictions for {column} are outside the acceptable range of {acceptable_ranges[column]}"
#             )
#         return True
#
#     @classmethod
#     def validate_prediction(cls, df, column, acceptable_ranges, jeditaskid):
#         """
#         Validates predictions for a given column and logs the result.
#
#         Parameters:
#         - df: DataFrame containing predictions.
#         - column: Column name to validate.
#         - acceptable_ranges: Acceptable ranges for validation.
#         - jeditaskid: ID for logging purposes.
#
#         Returns:
#         - bool: True if validation succeeds, False otherwise.
#         """
#         try:
#             cls.check_predictions(df, column, acceptable_ranges)
#             # logger.info(f"{column} predictions validated successfully.")
#             return True
#         except ValueError as ve:
#             logger.error(f"{column} validation failed for JEDITASKID {jeditaskid}: {ve}")
#             return False
#         except Exception as e:
#             logger.error(
#                 f"Unexpected error during {column} validation for JEDITASKID {jeditaskid}: {e}"
#             )
#             return False
#
#     @classmethod
#     def validate_ctime_prediction(cls, df, jeditaskid, additional_ctime_ranges):
#         """
#         Validates CTIME predictions using alternative ranges.
#
#         Parameters:
#         - df: DataFrame containing predictions.
#         - jeditaskid: ID for logging purposes.
#         - additional_ctime_ranges: Alternative ranges for CTIME validation.
#
#         Returns:
#         - bool: True if validation succeeds, False otherwise.
#         """
#         try:
#             if df["CPUTIMEUNIT"].values[0] == "mHS06sPerEvent":
#                 cls.check_predictions(
#                     df, "CTIME", {"CTIME": additional_ctime_ranges["low"]}
#                 )
#                 logger.info("Validation passed with low CTIME range.")
#                 return True
#             else:
#                 cls.check_predictions(
#                     df, "CTIME", {"CTIME": additional_ctime_ranges["high"]}
#                 )
#                 logger.info("Validation passed with high CTIME range.")
#                 return True
#         except ValueError as ve:
#             logger.error(f"Validation failed with all ranges: {ve}")
#             return False


# class DummyData:
#     @classmethod
#     def fetch_data(cls):
#         # Simulate data retrieval
#         data = {
#             "JEDITASKID": [27766704, 27746332],
#             "PRODSOURCELABEL": ["managed", "user"],
#             "PROCESSINGTYPE": ["deriv", "panda-client-1.4.98-jedi-run"],
#             "TRANSHOME": [
#                 "AthDerivation-21.2.77.0",
#                 "AnalysisTransforms-AnalysisBase_21.2.197",
#             ],
#             "CPUTIMEUNIT": ["HS06sPerEvent", "mHS06sPerEvent"],
#             "CORECOUNT": [8, 1],
#             "TOTAL_NFILES": [290000, 11237955],
#             "TOTAL_NEVENTS": [23, 260],
#             "DISTINCT_DATASETNAME_COUNT": [1, 3],
#         }
#         return pd.DataFrame(data)


class FakeListener:
    """
    Simulates a listener that sends task IDs with a fixed delay.

    Attributes:
    - task_ids (List[int]): A list of task IDs to be sent.
    - delay (int): The delay in seconds between sending each task ID.
    """

    def __init__(self, task_ids: List[int], delay: int = 3):
        """
        Initializes the FakeListener with a list of task IDs and a delay.

        Parameters:
        - task_ids (List[int]): A list of task IDs.
        - delay (int, optional): The delay in seconds. Defaults to 3.

        Raises:
        - TypeError: If task_ids is not a list or if delay is not an integer.
        """
        if not isinstance(task_ids, list) or not all(
            isinstance(task_id, int) for task_id in task_ids
        ):
            raise TypeError("task_ids must be a list of integers")
        if not isinstance(delay, int):
            raise TypeError("delay must be an integer")

        self.task_ids = task_ids
        self.delay = delay

    def demo_task_listener(self) -> Generator[int, None, None]:
        """
        A generator function that simulates a listener sending task IDs with a fixed delay.

        Yields:
        - int: A task ID from the list.

        Notes:
        - This function uses time.sleep to introduce a delay between yielding each task ID.
        """
        for task_id in self.task_ids:
            # Wait for the specified delay
            time.sleep(self.delay)
            yield task_id


class DataValidator:
    """
    Provides methods for validating data predictions.

    Attributes:
    - None

    Methods:
    - check_predictions: Checks if predictions are within acceptable ranges.
    - validate_prediction: Validates predictions for a given column and logs the result.
    - validate_ctime_prediction: Validates CTIME predictions using alternative ranges.
    """

    @classmethod
    def check_predictions(
        cls, df: pd.DataFrame, column: str, acceptable_ranges: dict
    ) -> bool:
        """
        Checks if predictions are within acceptable ranges.

        Parameters:
        - df (pd.DataFrame): DataFrame containing predictions.
        - column (str): Column name to check.
        - acceptable_ranges (dict): Dictionary with acceptable ranges for each column.

        Returns:
        - bool: True if validation succeeds (not used, raises ValueError on failure).

        Raises:
        - ValueError: If predictions are outside the acceptable range.
        """
        min_val, max_val = acceptable_ranges[column]
        if df[column].min() < min_val or df[column].max() > max_val:
            raise ValueError(
                f"Predictions for {column} are outside the acceptable range of {acceptable_ranges[column]}"
            )
        return True

    @classmethod
    def validate_prediction(
        cls, df: pd.DataFrame, column: str, acceptable_ranges: dict, jeditaskid: int
    ) -> bool:
        """
        Validates predictions for a given column and logs the result.

        Parameters:
        - df (pd.DataFrame): DataFrame containing predictions.
        - column (str): Column name to validate.
        - acceptable_ranges (dict): Dictionary with acceptable ranges for each column.
        - jeditaskid (int): ID for logging purposes.

        Returns:
        - bool: True if validation succeeds, False otherwise.
        """
        try:
            cls.check_predictions(df, column, acceptable_ranges)
            logger.info(f"{column} predictions validated successfully.")
            return True
        except ValueError as ve:
            logger.error(f"{column} validation failed for JEDITASKID {jeditaskid}: {ve}")
            return False
        except Exception as e:
            logger.error(
                f"Unexpected error during {column} validation for JEDITASKID {jeditaskid}: {e}"
            )
            return False

    @classmethod
    def validate_ctime_prediction(
        cls, df: pd.DataFrame, jeditaskid: int, additional_ctime_ranges: dict
    ) -> bool:
        """
        Validates CTIME predictions using alternative ranges.

        Parameters:
        - df (pd.DataFrame): DataFrame containing predictions.
        - jeditaskid (int): ID for logging purposes.
        - additional_ctime_ranges (dict): Dictionary with alternative ranges for CTIME validation.

        Returns:
        - bool: True if validation succeeds, False otherwise.
        """
        try:
            if df["CPUTIMEUNIT"].values[0] == "mHS06sPerEvent":
                cls.check_predictions(
                    df, "CTIME", {"CTIME": additional_ctime_ranges["low"]}
                )
                logger.info("Validation passed with low CTIME range.")
                return True
            else:
                cls.check_predictions(
                    df, "CTIME", {"CTIME": additional_ctime_ranges["high"]}
                )
                logger.info("Validation passed with high CTIME range.")
                return True
        except ValueError as ve:
            logger.error(f"Validation failed with all ranges: {ve}")
            return False


class DummyData:
    """
    Provides dummy data for testing purposes.

    Attributes:
    - None

    Methods:
    - fetch_data: Simulates data retrieval by returning a DataFrame.
    """

    @classmethod
    def fetch_data(cls) -> pd.DataFrame:
        """
        Simulates data retrieval by returning a DataFrame with dummy data.

        Returns:
        - pd.DataFrame: A DataFrame containing dummy data.
        """
        # Simulate data retrieval
        data = {
            "JEDITASKID": [27766704, 27746332],
            "PRODSOURCELABEL": ["managed", "user"],
            "PROCESSINGTYPE": ["deriv", "panda-client-1.4.98-jedi-run"],
            "TRANSHOME": [
                "AthDerivation-21.2.77.0",
                "AnalysisTransforms-AnalysisBase_21.2.197",
            ],
            "CPUTIMEUNIT": ["HS06sPerEvent", "mHS06sPerEvent"],
            "CORECOUNT": [8, 1],
            "TOTAL_NFILES": [290000, 11237955],
            "TOTAL_NEVENTS": [23, 260],
            "DISTINCT_DATASETNAME_COUNT": [1, 3],
        }
        return pd.DataFrame(data)
