import pandas as pd
from datetime import datetime
import sys
import queue
import threading
from oracledb import Error
import time

last_logged = time.time()
from scout_ml_package.data.fetch_db_data import DatabaseFetcher
from scout_ml_package.model.model_pipeline import ModelManager, PredictionPipeline
from scout_ml_package.utils.logger import Logger
from scout_ml_package.utils.validator import DataValidator
from scout_ml_package.utils.message import TaskIDListener

# logger = Logger("demo_logger", "/data/model-data/logs", "pred.log").get_logger()
logger = Logger("demo_logger", "/data/model-data/logs", "demo.log")
# Define acceptable ranges for each prediction
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


def get_prediction(model_manager, r, task_id):
    start_time = time.time()

    if r is None or r.empty:
        logger.error(f"DataFrame is empty or input data is None {task_id}.")
        return None

    jeditaskid = r["JEDITASKID"].values[0]
    processor = PredictionPipeline(model_manager)
    base_df = processor.preprocess_data(r)

    # Model 1: RAMCOUNT
    features = ["JEDITASKID"] + processor.numerical_features + processor.category_sequence
    base_df.loc[:, "RAMCOUNT"] = processor.make_predictions_for_model(
        "1", features, base_df
    )

    if not DataValidator.validate_prediction(
        base_df, "RAMCOUNT", acceptable_ranges, jeditaskid
    ):
        logger.error(f"RAMCOUNT validation failed for JEDITASKID {jeditaskid}.")
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
            logger.error(f"CTIME validation failed for JEDITASKID {jeditaskid}.")
            cpu_unit = base_df["CPUTIMEUNIT"].values[0]
            if cpu_unit == "mHS06sPerEvent":
                return f"M2 failure"
            else:
                return f"M3 failure"
    except Exception as e:
        logger.error(f"CTIME prediction failed for JEDITASKID {jeditaskid}: {str(e)}")
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
            logger.error(f"CPU_EFF validation failed for JEDITASKID {jeditaskid}.")
            return f"{jeditaskid}M4 failure: Validation failed."
    except Exception as e:
        logger.error(f"{jeditaskid} M4 failure: {str(e)}")
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
        logger.error(f"{jeditaskid}M5 failure: {str(e)}")
        return f"M5 failure: {str(e)}"

    logger.info(
        f"JEDITASKID {jeditaskid} processed successfully in {time.time() - start_time:.2f} seconds"
    )
    base_df[["RAMCOUNT", "CTIME", "CPU_EFF"]] = base_df[
        ["RAMCOUNT", "CTIME", "CPU_EFF"]
    ].round(3)
    return base_df


def fetch_and_enqueue(listener, task_queue):
    global last_logged
    while True:
        try:
            task_id = listener.get_task_id()
            if task_id is not None:
                task_queue.put(task_id)
                logger.info(f"Added task ID {task_id} to queue")
                last_logged = time.time()  # Reset timer when a task ID is received
            else:
                current_time = time.time()
                if current_time - last_logged >= 120:  # Check if 60 seconds have passed
                    logger.info("No task ID received.")
                    last_logged = current_time  # Update last logged time
        except Exception as e:
            logger.error(f"Error fetching task ID: {e}")


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


def process_task_v1(task_id, input_db, output_db, model_manager, cols_to_write):
    """
    Processes a single task by fetching its parameters, generating predictions,
    and handling errors appropriately.
    """
    # submission_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    submission_date = datetime.now()
    try:
        logger.info(f"Processing task ID: {task_id}")

        # Fetch task parameters
        r = input_db.fetch_task_param(task_id)
        if isinstance(r, pd.DataFrame) and not r.empty and not r.isnull().all().any():
            logger.info(f"Task parameters fetched successfully for JEDITASKID: {task_id}")
            # Generate prediction
            try:
                result = get_prediction(model_manager, r, task_id)
                if isinstance(result, pd.DataFrame):
                    logger.info(
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
                    }
                    logger.info(f"Success message: {message}")

                else:
                    # Handle non-DataFrame results as an error
                    raise ValueError(
                        f"Prediction failed for JEDITASKID: {task_id}. Result: {result}"
                    )

            except Error as e:
                (error_obj,) = e.args
                if "DPY-1001" in error_obj.message:
                    logger.error(
                        f"Database connection error: {e}. Exiting to trigger service restart."
                    )
                    sys.exit(1)  # Exit with a non-zero status to trigger restart
                else:
                    logger.error(f"Oracle interface error for JEDITASKID: {task_id}: {e}")
                    handle_error(
                        task_id, r, str(e), cols_to_write, submission_date, output_db
                    )

            except Exception as e:
                logger.error(f"Error during prediction for JEDITASKID: {task_id}: {e}")
                handle_error(
                    task_id, r, str(e), cols_to_write, submission_date, output_db
                )
        else:
            # Handle invalid or empty DataFrame `r`
            error_message = (
                f"Invalid or empty DataFrame fetched for JEDITASKID: {task_id}"
            )
            logger.error(error_message)
            handle_error(
                task_id,
                r if isinstance(r, pd.DataFrame) else None,
                error_message,
                cols_to_write,
                submission_date,
                output_db,
            )

    except Exception as e:
        logger.error(f"Error processing task ID: {e}")


# Example usage (replace with actual implementations)
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

    listener = TaskIDListener(config_file="/data/model-data/configs/config.ini")
    listener.start_listening()

    task_queue = queue.Queue()

    # Start a thread to fetch and enqueue task IDs
    #threading.Thread(target=self.listen_for_tasks).start()
    fetch_thread = threading.Thread(target=fetch_and_enqueue, args=(listener, task_queue))
    fetch_thread.daemon = (
        True  # Allow the main thread to exit even if this thread is still running
    )
    fetch_thread.start()

    while True:
        try:
            if not task_queue.empty():
                task_id = task_queue.get()
                logger.info(f"Processing task ID: {task_id}")
                print(f"Received JEDITASKID: {task_id}")
                logger.info("Calling process_task_v1...")
                process_task_v1(
                    task_id, input_db, output_db, model_manager, cols_to_write
                )
                logger.info("Finished processing task ID.")
            else:
                logger.info("Task queue is empty. Sleeping for 60 seconds...")
                time.sleep(60)
        except Exception as e:
            logger.error(f"Error processing task ID: {e}")

    print("All tasks processed")
    input_db.close_connection()
    output_db.close_connection()
