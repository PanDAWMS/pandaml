import pandas as pd
from datetime import datetime
import sys
import queue
import threading
import time
import warnings
import pandas as pd
import logging
import sys
from datetime import datetime
import time
import pandas as pd
import logging
from typing import List

warnings.filterwarnings("ignore", category=UserWarning, module="pandas.io.sql")

last_logged = time.time()
from scout_ml_package.data.fetch_db_data import DatabaseFetcher
from scout_ml_package.model.model_pipeline import (
    ModelManager,
    PredictionPipeline,
    ColumnTransformer,
)
from scout_ml_package.utils.logger import Logger
from scout_ml_package.utils.validator import DataValidator
from scout_ml_package.utils.message import ConfigLoader, MyListener  # TaskIDListener

logger = Logger("demo_logger", "/data/model-data/logs", "prediction_v1.log")
# Acceptable ranges for each prediction
acceptable_ranges = {
    # Adjust these ranges based on your domain knowledge
    "RAMCOUNT": (100, 10000),
    "CTIME": (0.1, 10000),
    "CPU_EFF": (0, 100),
}

additional_ctime_ranges = {
    "low": (0.1, 10),
    "high": (10, 10000),
}


def fetch_and_process(task_id_queue):
    print("Task processing thread started")
    while True:
        try:
            task_id = task_id_queue.get(timeout=0.1)  # Check every 100 ms
            if task_id is not None:
                print(f"Processing task ID: {task_id}")
                logger.info(f"Processing task ID: {task_id}")
                process_single_task(
                    task_id, input_db, output_db, model_manager, cols_to_write
                )
            else:
                print("No task ID received.")
                logger.info("No task ID received.")
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error processing task ID: {e}")
            print(f"Error processing task ID: {e}")


#
# def get_prediction(model_manager, r, task_id):
#     start_time = time.time()
#
#     if not model_manager.are_models_loaded():
#         logger.info("Models are not loaded. Loading models...")
#         model_manager.load_models()
#         logger.info("Models re-loaded successfully.")
#
#     if r is None or r.empty:
#         logger.error(f"DataFrame is empty or input data is None {task_id}.")
#         return None
#
#     jeditaskid = r["JEDITASKID"].values[0]
#     processor = PredictionPipeline(model_manager)
#     base_df = ColumnTransformer().transform_features(r)
#     # print(base_df)
#     # print(base_df.columns)
#
#     # Model 1: RAMCOUNT
#     features = ["JEDITASKID"] + processor.numerical_features + processor.category_sequence
#     base_df.loc[:, "RAMCOUNT"] = processor.make_predictions_for_model(
#         "1", features, base_df
#     )
#
#     if not DataValidator.validate_prediction(
#         base_df, "RAMCOUNT", acceptable_ranges, jeditaskid
#     ):
#         logger.error(f"RAMCOUNT validation failed for JEDITASKID {jeditaskid}.")
#         return f"M1 failure."
#
#     # Update features for subsequent models
#     processor.numerical_features.append("RAMCOUNT")
#     features = ["JEDITASKID"] + processor.numerical_features + processor.category_sequence
#
#     # Model 2/3: CTIME
#     try:
#         if base_df["CPUTIMEUNIT"].values[0] == "mHS06sPerEvent":
#             base_df.loc[:, "CTIME"] = processor.make_predictions_for_model(
#                 "2", features, base_df
#             )
#         else:
#             base_df.loc[:, "CTIME"] = processor.make_predictions_for_model(
#                 "3", features, base_df
#             )
#
#         if not DataValidator.validate_ctime_prediction(
#             base_df, jeditaskid, additional_ctime_ranges
#         ):
#             logger.error(f"CTIME validation failed for JEDITASKID {jeditaskid}.")
#             cpu_unit = base_df["CPUTIMEUNIT"].values[0]
#             if cpu_unit == "mHS06sPerEvent":
#                 return f"M2 failure"
#             else:
#                 return f"M3 failure"
#     except Exception as e:
#         logger.error(f"CTIME prediction failed for JEDITASKID {jeditaskid}: {str(e)}")
#         cpu_unit = base_df["CPUTIMEUNIT"].values[0]
#         if cpu_unit == "mHS06sPerEvent":
#             return f"{jeditaskid}M2 failure: {str(e)}"
#         else:
#             return f"{jeditaskid}M3 failure: {str(e)}"
#
#     # Update features for subsequent models
#     processor.numerical_features.append("CTIME")
#     features = ["JEDITASKID"] + processor.numerical_features + processor.category_sequence
#
#     # Model 4: CPU_EFF
#     try:
#         base_df.loc[:, "CPU_EFF"] = processor.make_predictions_for_model(
#             "4", features, base_df
#         )
#         if not DataValidator.validate_prediction(
#             base_df, "CPU_EFF", acceptable_ranges, jeditaskid
#         ):
#             logger.error(f"CPU_EFF validation failed for JEDITASKID {jeditaskid}.")
#             return f"{jeditaskid}M4 failure: Validation failed."
#     except Exception as e:
#         logger.error(f"{jeditaskid} M4 failure: {str(e)}")
#         return f"M4 failure: {str(e)}"
#
#     # Update features for subsequent models
#     processor.numerical_features.append("CPU_EFF")
#     features = ["JEDITASKID"] + processor.numerical_features + processor.category_sequence
#
#     # Model 5: IOINTENSITY
#     try:
#         base_df.loc[:, "IOINTENSITY"] = processor.make_predictions_for_model(
#             "5", features, base_df
#         )
#     except Exception as e:
#         logger.error(f"{jeditaskid}M5 failure: {str(e)}")
#         return f"M5 failure: {str(e)}"
#
#     logger.info(
#         f"JEDITASKID {jeditaskid} processed successfully in {time.time() - start_time:.2f} seconds"
#     )
#     base_df[["RAMCOUNT", "CTIME", "CPU_EFF"]] = base_df[
#         ["RAMCOUNT", "CTIME", "CPU_EFF"]
#     ].round(3)
#     return base_df


def handle_error(task_id, r, error_message, cols_to_write, submission_date, output_db):
    """
    Handles errors by logging them and writing error information to the database.
    """
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
        logger.info(f"Error logged successfully for JEDITASKID: {task_id}")
    except Exception as e:
        logger.exception(f"Failed to handle error for JEDITASKID: {task_id}: {e}")


#
# def process_single_task(task_id, input_db, output_db, model_manager, cols_to_write):
#     """
#     Processes a single task by fetching its parameters, generating predictions,
#     and handling errors appropriately.
#     """
#     # submission_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     submission_date = datetime.now()
#     try:
#         logger.info(f"Processing task ID: {task_id}")
#
#         # Fetch task parameters
#         r = input_db.fetch_task_param(task_id)
#         if isinstance(r, pd.DataFrame) and not r.empty and not r.isnull().all().any():
#             logger.info(f"Task parameters fetched successfully for JEDITASKID: {task_id}")
#             # Generate prediction
#             try:
#                 result = get_prediction(model_manager, r, task_id)
#                 if isinstance(result, pd.DataFrame):
#                     logger.info(
#                         f"Prediction completed successfully for JEDITASKID: {task_id}"
#                     )
#                     submission_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#                     # Process and write results to the output database
#                     result = result[cols_to_write].copy()
#                     result["SUBMISSION_DATE"] = datetime.now()
#                     # result["SUBMISSION_DATE"] = result["SUBMISSION_DATE"].dt.strftime('%Y-%m-%d %H:%M:%S')
#                     output_db.write_data(result, "ATLAS_PANDA.PANDAMLTEST")
#
#                     # Prepare success message
#                     message = {
#                         "taskid": result["JEDITASKID"].values[0],
#                         "status": "success",
#                         "RAMCOUNT": result["RAMCOUNT"].values[0],
#                         "CTIME": result["CTIME"].values[0],
#                         "CPU_EFF": result["CPU_EFF"].values[0],
#                         "IOINTENSITY": result["IOINTENSITY"].values[0],
#                         "submission_time": submission_date,
#                     }
#                     logger.info(f"Success message: {message}")
#
#                 else:
#                     # Handle non-DataFrame results as an error
#                     message = {
#                         "taskid": result["JEDITASKID"].values[0],
#                         "status": "failure",
#                         "submission_time": submission_date,
#                     }
#                     logger.info(f"Failure message: {message}")
#                     raise ValueError(
#                         f"Prediction failed for JEDITASKID: {task_id}. Result: {result}"
#                     )
#
#             except Error as e:
#                 (error_obj,) = e.args
#                 if "DPY-1001" in error_obj.message:
#                     logger.error(
#                         f"Database connection error: {e}. Exiting to trigger service restart."
#                     )
#                     sys.exit(1)  # Exit with a non-zero status to trigger restart
#                 else:
#                     logger.error(f"Oracle interface error for JEDITASKID: {task_id}: {e}")
#                     handle_error(
#                         task_id, r, str(e), cols_to_write, submission_date, output_db
#                     )
#
#             except Exception as e:
#                 logger.error(f"Error during prediction for JEDITASKID: {task_id}: {e}")
#                 handle_error(
#                     task_id, r, str(e), cols_to_write, submission_date, output_db
#                 )
#         else:
#             # Handle invalid or empty DataFrame `r`
#             error_message = (
#                 f"Invalid or empty DataFrame fetched for JEDITASKID: {task_id}"
#             )
#             logger.error(error_message)
#             handle_error(
#                 task_id,
#                 r if isinstance(r, pd.DataFrame) else None,
#                 error_message,
#                 cols_to_write,
#                 submission_date,
#                 output_db,
#             )
#
#     except Exception as e:
#         logger.error(f"Error processing task ID: {e}")


def get_prediction(
    model_manager: "ModelManager", r: pd.DataFrame, task_id: int
) -> pd.DataFrame:
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
    if not isinstance(task_id, str):
        raise TypeError("task_id must be a string")

    start_time = time.time()

    if not model_manager.are_models_loaded():
        logging.info("Models are not loaded. Loading models...")
        model_manager.load_models()
        logging.info("Models re-loaded successfully.")

    if r is None or r.empty:
        logging.error(f"DataFrame is empty or input data is None for task ID {task_id}.")
        return None

    jeditaskid = r["JEDITASKID"].values[0]
    processor = PredictionPipeline(model_manager)
    base_df = ColumnTransformer().transform_features(r)

    # Model 1: RAMCOUNT
    features = ["JEDITASKID"] + processor.numerical_features + processor.category_sequence
    base_df.loc[:, "RAMCOUNT"] = processor.make_predictions_for_model(
        "1", features, base_df
    )

    if not DataValidator.validate_prediction(
        base_df, "RAMCOUNT", acceptable_ranges, jeditaskid
    ):
        logging.error(f"RAMCOUNT validation failed for JEDITASKID {jeditaskid}.")
        return f"M1 failure."

    # Update features for subsequent models
    processor.numerical_features.append("RAMCOUNT")
    features = ["JEDITASKID"] + processor.numerical_features + processor.category_sequence

    # Model 2/3: CTIME
    try:
        if base_df["CPUTIMEUNIT"].values[0] == "mHS06sPerEvent":
            base_df.loc[:, "CTIME"] = processor.make_predictions_for_model(
                "2", features, base_df
            )
        else:
            base_df.loc[:, "CTIME"] = processor.make_predictions_for_model(
                "3", features, base_df
            )

        if not DataValidator.validate_ctime_prediction(
            base_df, jeditaskid, additional_ctime_ranges
        ):
            logging.error(f"CTIME validation failed for JEDITASKID {jeditaskid}.")
            cpu_unit = base_df["CPUTIMEUNIT"].values[0]
            if cpu_unit == "mHS06sPerEvent":
                return f"M2 failure"
            else:
                return f"M3 failure"
    except Exception as e:
        logging.error(f"CTIME prediction failed for JEDITASKID {jeditaskid}: {str(e)}")
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
        base_df.loc[:, "CPU_EFF"] = processor.make_predictions_for_model(
            "4", features, base_df
        )
        if not DataValidator.validate_prediction(
            base_df, "CPU_EFF", acceptable_ranges, jeditaskid
        ):
            logging.error(f"CPU_EFF validation failed for JEDITASKID {jeditaskid}.")
            return f"{jeditaskid}M4 failure: Validation failed."
    except Exception as e:
        logging.error(f"{jeditaskid} M4 failure: {str(e)}")
        return f"M4 failure: {str(e)}"

    # Update features for subsequent models
    processor.numerical_features.append("CPU_EFF")
    features = ["JEDITASKID"] + processor.numerical_features + processor.category_sequence

    # Model 5: IOINTENSITY
    try:
        base_df.loc[:, "IOINTENSITY"] = processor.make_predictions_for_model(
            "5", features, base_df
        )
    except Exception as e:
        logging.error(f"{jeditaskid}M5 failure: {str(e)}")
        return f"M5 failure: {str(e)}"

    logging.info(
        f"JEDITASKID {jeditaskid} processed successfully in {time.time() - start_time:.2f} seconds"
    )
    base_df[["RAMCOUNT", "CTIME", "CPU_EFF"]] = base_df[
        ["RAMCOUNT", "CTIME", "CPU_EFF"]
    ].round(3)
    return base_df


def process_single_task(
    task_id: str,
    input_db: object,
    output_db: object,
    model_manager: "ModelManager",
    cols_to_write: List[str],
) -> None:
    """
    Processes a single task by fetching its parameters, generating predictions,
    and handling errors appropriately.

    Parameters:
    - task_id (str): ID of the task to process.
    - input_db (object): Input database object.
    - output_db (object): Output database object.
    - model_manager (ModelManager): Instance of ModelManager for accessing models.
    - cols_to_write (List[str]): List of columns to write to the output database.

    Raises:
    - TypeError: If task_id is not a string or if cols_to_write is not a list.
    """
    if not isinstance(task_id, str):
        raise TypeError("task_id must be a string")
    if not isinstance(cols_to_write, list):
        raise TypeError("cols_to_write must be a list")

    submission_date = datetime.now()
    try:
        logging.info(f"Processing task ID: {task_id}")

        # Fetch task parameters
        r = input_db.fetch_task_param(task_id)
        if isinstance(r, pd.DataFrame) and not r.empty and not r.isnull().all().any():
            logging.info(
                f"Task parameters fetched successfully for JEDITASKID: {task_id}"
            )
            # Generate prediction
            try:
                result = get_prediction(model_manager, r, task_id)
                if isinstance(result, pd.DataFrame):
                    logging.info(
                        f"Prediction completed successfully for JEDITASKID: {task_id}"
                    )
                    submission_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    # Process and write results to the output database
                    result = result[cols_to_write].copy()
                    result["SUBMISSION_DATE"] = datetime.now()
                    # result["SUBMISSION_DATE"] = result["SUBMISSION_DATE"].dt.strftime('%Y-%m-%d %H:%M:%S')
                    output_db.write_data(result, "ATLAS_PANDA.PANDAMLTEST")

                    # Prepare success message
                    message = {
                        "taskid": result["JEDITASKID"].values[0],
                        "status": "success",
                        "RAMCOUNT": result["RAMCOUNT"].values[0],
                        "CTIME": result["CTIME"].values[0],
                        "CPU_EFF": result["CPU_EFF"].values[0],
                        "IOINTENSITY": result["IOINTENSITY"].values[0],
                        "submission_time": submission_date,
                    }
                    logging.info(f"Success message: {message}")

                else:
                    # Handle non-DataFrame results as an error
                    message = {
                        "taskid": task_id,
                        "status": "failure",
                        "submission_time": submission_date,
                    }
                    logging.info(f"Failure message: {message}")
                    raise ValueError(
                        f"Prediction failed for JEDITASKID: {task_id}. Result: {result}"
                    )

            except Exception as e:
                if hasattr(e, "args") and "DPY-1001" in e.args[0].message:
                    logging.error(
                        f"Database connection error: {e}. Exiting to trigger service restart."
                    )
                    sys.exit(1)  # Exit with a non-zero status to trigger restart
                else:
                    logging.error(
                        f"Oracle interface error for JEDITASKID: {task_id}: {e}"
                    )
                    handle_error(
                        task_id, r, str(e), cols_to_write, submission_date, output_db
                    )

        else:
            # Handle invalid or empty DataFrame `r`
            error_message = (
                f"Invalid or empty DataFrame fetched for JEDITASKID: {task_id}"
            )
            logging.error(error_message)
            handle_error(
                task_id,
                r if isinstance(r, pd.DataFrame) else None,
                error_message,
                cols_to_write,
                submission_date,
                output_db,
            )

    except Exception as e:
        logging.error(f"Error processing task ID: {e}")


if __name__ == "__main__":
    base_path = "/data/model-data/"  # "/data/test/"
    model_manager = ModelManager(base_path)
    model_manager.load_models()

    input_db = DatabaseFetcher("database")
    output_db = DatabaseFetcher("output_database")

    cols_to_write = [
        "JEDITASKID",
        "PRODSOURCELABEL",
        "PROCESSINGTYPE",
        "TRANSHOME",
        "CPUTIMEUNIT",
        "CORECOUNT",
        "TOTAL_NFILES",
        "TOTAL_NEVENTS",
        "DISTINCT_DATASETNAME_COUNT",
        "RAMCOUNT",
        "CTIME",
        "CPU_EFF",
        "IOINTENSITY",
    ]

    # Load configuration
    config_path = (
        "/data/model-data/configs/config.ini"  # Replace with your config file path
    )
    config_loader = ConfigLoader(config_path)
    # Create queues to hold task IDs
    task_id_queue = queue.Queue()
    # task_queue = queue.Queue()

    # Create and start listener
    listener = MyListener(
        task_id_queue,
        config_loader.mb_server_host_port,
        config_loader.vhost,
        config_loader.username,
        config_loader.passcode,
        config_loader.queue_name,
    )

    # Start thread
    fetch_thread = threading.Thread(target=fetch_and_process, args=(task_id_queue,))
    fetch_thread.daemon = True
    fetch_thread.start()

    # Keep the main thread running

    print_time = time.time()
    while True:
        if time.time() - print_time >= 120:  # Check if 2 minutes have passed
            print("Main thread alive", end="\r")
            print_time = time.time()  # Reset the timer
        time.sleep(1)  # Sleep for 1 second

    print("All tasks processed")
    input_db.close_connection()
    output_db.close_connection()
