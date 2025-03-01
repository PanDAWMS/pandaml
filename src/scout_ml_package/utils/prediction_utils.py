import queue
import pandas as pd
from datetime import datetime
import sys
import time
from typing import List


from scout_ml_package.data.fetch_db_data import DatabaseFetcher
from scout_ml_package.model.model_pipeline import (
    ModelManager,
    PredictionPipeline,
    ColumnTransformer,
)
from scout_ml_package.utils.validator import DataValidator

import logging
from typing import Optional


class PredictionUtils:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger if logger else logging.getLogger(__name__)

    def fetch_and_process(
        self,
        task_id_queue: queue.Queue,
        input_db: DatabaseFetcher,
        output_db: DatabaseFetcher,
        model_manager: ModelManager,
        cols_to_write: List[str],
    ) -> None:
        """
        Fetches task IDs from a queue and processes them using a model.

        Parameters:
        - task_id_queue (queue.Queue): Queue containing task IDs to process.
        - input_db (object): Input database object.
        - output_db (object): Output database object.
        - model_manager (ModelManager): Instance of ModelManager for accessing models.
        - cols_to_write (List[str]): List of columns to write to the output database.

        Raises:
        - TypeError: If task_id_queue is not a queue.Queue or if cols_to_write is not a list.
        """
        if not isinstance(task_id_queue, queue.Queue):
            raise TypeError("task_id_queue must be a queue.Queue")
        if not isinstance(cols_to_write, list):
            raise TypeError("cols_to_write must be a list")

        self.logger.info("Task processing thread started")
        while True:
            try:
                task_id = task_id_queue.get(timeout=0.1)  # Check every 100 ms
                if task_id is not None:
                    # logger.info(f"Processing task ID: {task_id}")
                    self.process_single_task(task_id, input_db, output_db, model_manager, cols_to_write)
                else:
                    # print("No task ID received.")
                    self.logger.info("No task ID received.")
            except queue.Empty:
                pass
            except Exception as e:
                self.logger.error(f"Error processing task ID: {e}")
                print(f"Error processing task ID: {e}")

    def handle_error(
        self,
        task_id: int,
        r: pd.DataFrame,
        error_message: str,
        cols_to_write: List[str],
        submission_date: datetime,
        output_db: DatabaseFetcher,
    ) -> None:
        """
        Handles errors by logging them and writing error information to the database.

        Parameters:
        - task_id (int): ID of the task that encountered an error.
        - r (pd.DataFrame): DataFrame containing task parameters. If None or not a DataFrame, an empty DataFrame is used.
        - error_message (str): Message describing the error.
        - cols_to_write (List[str]): List of columns to include in the error data.
        - submission_date (str): Date when the error occurred.
        - output_db (object): Output database object.

        Raises:
        - TypeError: If task_id is not an integer, or if error_message is not a string, or if cols_to_write is not a list.
        """
        if not isinstance(task_id, int):
            raise TypeError("task_id must be an integer")
        if not isinstance(error_message, str):
            raise TypeError("error_message must be a string")
        if not isinstance(cols_to_write, list):
            raise TypeError("cols_to_write must be a list")
        if not isinstance(submission_date, datetime):
            raise TypeError("submission_date must be a datetime")

        try:
            error_df = r.copy() if isinstance(r, pd.DataFrame) else pd.DataFrame()
            error_df["ERROR"] = error_message

            # Add dummy columns if necessary to match the schema of the main table
            for col in cols_to_write:
                if col not in error_df.columns:
                    error_df[col] = None

            error_df["SUBMISSION_DATE"] = submission_date

            # Write error data to the database
            output_db.write_data(
                error_df[cols_to_write + ["ERROR", "SUBMISSION_DATE"]],
                "ATLAS_PANDA.PANDAMLTEST",
            )
            self.logger.info(f"Error logged successfully for JEDITASKID: {task_id}")
        except Exception as e:
            self.logger.exception(f"Failed to handle error for JEDITASKID: {task_id}: {e}")

    def get_prediction(self, model_manager: ModelManager, r: pd.DataFrame, task_id: int) -> pd.DataFrame:
        """
        Generates predictions for a task using multiple models.

        Parameters:
        - model_manager (ModelManager): Instance of ModelManager for accessing models.
        - r (pd.DataFrame): DataFrame containing task parameters.
        - task_id (str): ID of the task.

        Returns:
        - pd.DataFrame: DataFrame with predictions.

        Raises:
        - TypeError: If model_manager is not an instance of ModelManager or if r is not a DataFrame.
        """
        if not isinstance(model_manager, ModelManager):
            raise TypeError("model_manager must be an instance of ModelManager")
        if not isinstance(r, pd.DataFrame):
            raise TypeError("r must be a pandas DataFrame")
        if not isinstance(task_id, int):
            raise TypeError("task_id must be a integer")

        start_time = time.time()

        if not model_manager.are_models_loaded():
            self.logger.info("Models are not loaded. Loading models...")
            model_manager.load_models()
            self.logger.info("Models re-loaded successfully.")

        if r is None or r.empty:
            self.logger.error(f"DataFrame is empty or input data is None for task ID {task_id}.")
            return None

        jeditaskid = r["JEDITASKID"].values[0]
        processor = PredictionPipeline(model_manager)
        base_df = ColumnTransformer().transform_features(r)

        # Model 1: RAMCOUNT
        features = ["JEDITASKID"] + processor.numerical_features + processor.category_sequence
        base_df.loc[:, "RAMCOUNT"] = processor.make_predictions_for_model("1", features, base_df)

        if not DataValidator.validate_prediction(
            base_df, "RAMCOUNT", DataValidator.acceptable_ranges, jeditaskid, self.logger
        ):
            self.logger.error(f"RAMCOUNT validation failed for JEDITASKID {jeditaskid}.")
            return "M1 failure"

        # Update features for subsequent models
        processor.numerical_features.append("RAMCOUNT")
        features = ["JEDITASKID"] + processor.numerical_features + processor.category_sequence

        # Model 2/3: CTIME
        try:
            if base_df["CPUTIMEUNIT"].values[0] == "mHS06sPerEvent":
                base_df.loc[:, "CTIME"] = processor.make_predictions_for_model("2", features, base_df)
            else:
                base_df.loc[:, "CTIME"] = processor.make_predictions_for_model("3", features, base_df)

            if not DataValidator.validate_ctime_prediction(
                base_df, jeditaskid, DataValidator.additional_ctime_ranges, self.logger
            ):
                self.logger.error(f"CTIME validation failed for JEDITASKID {jeditaskid}.")
                cpu_unit = base_df["CPUTIMEUNIT"].values[0]
                if cpu_unit == "mHS06sPerEvent":
                    return "M2 failure"
                else:
                    return "M3 failure"
        except Exception as e:
            self.logger.error(f"CTIME prediction failed for JEDITASKID {jeditaskid}: {str(e)}")
            cpu_unit = base_df["CPUTIMEUNIT"].values[0]
            if cpu_unit == "mHS06sPerEvent":
                return f"{jeditaskid}M2 failure: {str(e)}"
            else:
                return f"{jeditaskid}M3 failure: {str(e)}"

        # Update features for subsequent models
        processor.numerical_features.append("CTIME")
        features = ["JEDITASKID"] + processor.numerical_features + processor.category_sequence

        # Model 4: CPU_EFF
        try:
            base_df.loc[:, "CPU_EFF"] = processor.make_predictions_for_model("4", features, base_df)
            if not DataValidator.validate_prediction(
                base_df, "CPU_EFF", DataValidator.acceptable_ranges, jeditaskid, self.logger
            ):
                self.logger.error(f"CPU_EFF validation failed for JEDITASKID {jeditaskid}.")
                return f"{jeditaskid} M4 failure: Validation failed."
        except Exception as e:
            self.logger.error(f"{jeditaskid} M4 failure: {str(e)}")
            return f"M4 failure: {str(e)}"

        # Update features for subsequent models
        processor.numerical_features.append("CPU_EFF")
        features = ["JEDITASKID"] + processor.numerical_features + processor.category_sequence

        # Model 5: IOINTENSITY
        try:
            base_df.loc[:, "IOINTENSITY"] = processor.make_predictions_for_model("5", features, base_df)
        except Exception as e:
            self.logger.error(f"{jeditaskid}M5 failure: {str(e)}")
            return f"M5 failure: {str(e)}"

        self.logger.info(f"JEDITASKID {jeditaskid} processed successfully in {time.time() - start_time:.2f} seconds")
        base_df[["RAMCOUNT", "CTIME", "CPU_EFF"]] = base_df[["RAMCOUNT", "CTIME", "CPU_EFF"]].round(3)
        return base_df

    def process_single_task(
        self,
        task_id: int,
        input_db: DatabaseFetcher,
        output_db: DatabaseFetcher,
        model_manager: ModelManager,
        cols_to_write: List[str],
    ) -> None:
        """
        Processes a single task by fetching its parameters, generating predictions,
        and handling errors appropriately.

        Parameters:
        - task_id (int): ID of the task to process.
        - input_db (DatabaseFetcher): Input database object.
        - output_db (DatabaseFetcher): Output database object.
        - model_manager (ModelManager): Instance of ModelManager for accessing models.
        - cols_to_write (List[str]): List of columns to write to the output database.

        Raises:
        - TypeError: If task_id is not an integer or if cols_to_write is not a list.
        """
        if not isinstance(task_id, int):
            raise TypeError("task_id must be an integer")
        if not isinstance(cols_to_write, list):
            raise TypeError("cols_to_write must be a list")

        submission_date = datetime.now()
        try:
            self.logger.info(f"Processing task ID: {task_id}")

            # Fetch task parameters
            retry_count = 0
            max_retries = 3
            task_params_fetched = False
            while retry_count < max_retries and not task_params_fetched:
                try:
                    r = input_db.fetch_task_param(task_id)
                    task_params_fetched = True
                except Exception as e:
                    if hasattr(e, "args") and "DPY-1001" in e.args[0].message:
                        self.logger.error(f"Database connection error fetching task parameters: {e}. Retrying...")
                        retry_count += 1
                        time.sleep(5)  # Wait before retrying
                    else:
                        self.logger.error(f"Error fetching task parameters for JEDITASKID: {task_id}: {e}")
                        raise

            if task_params_fetched and isinstance(r, pd.DataFrame) and not r.empty and not r.isnull().all().any():
                self.logger.info(f"Task parameters fetched successfully for JEDITASKID: {task_id}")
                # Generate prediction
                try:
                    result = self.get_prediction(model_manager, r, task_id)
                    if isinstance(result, pd.DataFrame):
                        self.logger.info(f"Prediction completed successfully for JEDITASKID: {task_id}")
                        submission_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        # Process and write results to the output database
                        retry_count = 0
                        write_successful = False
                        while retry_count < max_retries and not write_successful:
                            try:
                                result = result[cols_to_write].copy()
                                result["SUBMISSION_DATE"] = datetime.now()
                                output_db.write_data(result, "ATLAS_PANDA.PANDAMLTEST")
                                write_successful = True
                            except Exception as e:
                                if hasattr(e, "args") and "DPY-1001" in e.args[0].message:
                                    self.logger.error(f"Database connection error writing results: {e}. Retrying...")
                                    retry_count += 1
                                    time.sleep(5)  # Wait before retrying
                                else:
                                    self.logger.error(f"Error writing results for JEDITASKID: {task_id}: {e}")
                                    raise

                        if write_successful:
                            # Prepare success message
                            message = {
                                "taskid": result["JEDITASKID"].values[0],
                                "PRODSOURCELABEL": result["PRODSOURCELABEL"].values[0],
                                "status": "success",
                                "RAMCOUNT": result["RAMCOUNT"].values[0],
                                "CTIME": result["CTIME"].values[0],
                                "CPU_EFF": result["CPU_EFF"].values[0],
                                "IOINTENSITY": result["IOINTENSITY"].values[0],
                                "submission_time": submission_date,
                            }
                            self.logger.info(f"Success message: {message}")

                        else:
                            self.logger.error("All retries failed writing results. Exiting.")
                            sys.exit(1)

                    else:
                        # Handle non-DataFrame results as an error
                        message = {
                            "taskid": task_id,
                            "status": "failure",
                            "submission_time": submission_date,
                        }
                        self.logger.info(f"Failure message: {message}")
                        raise ValueError(f"Prediction failed for JEDITASKID: {task_id}. Result: {result}")

                except Exception as e:
                    if hasattr(e, "args") and "DPY-1001" in e.args[0].message:
                        self.logger.error(
                            f"Database connection error during prediction: {e}. Exiting to trigger service restart."
                        )
                        sys.exit(1)  # Exit with a non-zero status to trigger restart
                    else:
                        self.logger.error(f"Oracle interface error for JEDITASKID: {task_id}: {e}")
                        self.handle_error(task_id, r, str(e), cols_to_write, submission_date, output_db)
                        raise

            else:
                # Handle invalid or empty DataFrame `r`
                error_message = f"Invalid or empty DataFrame fetched for JEDITASKID: {task_id}"
                self.logger.error(error_message)
                self.handle_error(
                    task_id,
                    r if isinstance(r, pd.DataFrame) else None,
                    error_message,
                    cols_to_write,
                    submission_date,
                    output_db,
                )

        except Exception as e:
            self.logger.error(f"Error processing task ID: {e}")
            raise
