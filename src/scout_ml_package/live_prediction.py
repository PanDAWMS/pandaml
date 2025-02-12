import pandas as pd
from datetime import datetime
import queue
import threading
import sys

# from scout_ml_package.utils.logger import configure_logger
from scout_ml_package.data.fetch_db_data import DatabaseFetcher
from scout_ml_package.model.model_pipeline import ModelManager
from scout_ml_package.utils.logger import Logger
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


def fetch_and_enqueue(listener, task_queue):
    """
    Fetches task IDs and adds them to the queue.
    """
    while True:
        task_id = listener.get_task_id()
        if task_id is not None:
            task_queue.put(task_id)
            logger.info(f"Task ID {task_id} added to queue")


def process_tasks(task_queue, input_db, output_db, model_manager, cols_to_write):
    """
    Processes tasks by fetching them from the queue, fetching task parameters,
    and handling predictions or errors appropriately.
    """
    submission_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    while True:
        try:
            task_id = task_queue.get(timeout=1)  # Wait up to 1 second for a task
            logger.info(f"Processing task ID: {task_id}")

            # Fetch task parameters
            r = input_db.fetch_task_param(task_id)
            if isinstance(r, pd.DataFrame) and not r.empty:
                logger.info(
                    f"Task parameters fetched successfully for JEDITASKID: {task_id}"
                )

                # Generate prediction
                try:
                    result = get_prediction(model_manager, r, task_id)
                    if isinstance(result, pd.DataFrame):
                        logger.info(
                            f"Prediction completed successfully for JEDITASKID: {task_id}"
                        )

                        # Process and write results to the output database
                        result = result[cols_to_write].copy()
                        result["SUBMISSION_DATE"] = submission_date
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

                except oracledb.exceptions.InterfaceError as e:
                    if "DPY-1001" in str(e):
                        logger.error(
                            f"Database connection error: {e}. Exiting to trigger service restart."
                        )
                        sys.exit(1)  # Exit with a non-zero status to trigger restart
                    else:
                        logger.error(
                            f"Oracle interface error for JEDITASKID: {task_id}: {e}"
                        )
                        handle_error(
                            task_id, r, str(e), cols_to_write, submission_date, output_db
                        )

                except Exception as e:
                    logger.error(
                        f"Error during prediction for JEDITASKID: {task_id}: {e}"
                    )
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

            task_queue.task_done()  # Mark the task as done

        except queue.Empty:
            # Handle the case where the queue is empty
            pass


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
    listener = TaskIDListener(
        mb_server_host_port=("aipanda100.cern.ch", 61613),
        queue_name="/queue/new_task_notif",
        vhost="/",
        username="panda",
        passcode="panda",
    )
    listener.start_listening()

    task_queue = queue.Queue()

    # Start a thread to fetch and enqueue task IDs
    fetch_thread = threading.Thread(target=fetch_and_enqueue, args=(listener, task_queue))
    fetch_thread.daemon = (
        True  # Allow the main thread to exit even if this thread is still running
    )
    fetch_thread.start()

    # Start processing tasks
    process_tasks(task_queue, input_db, output_db, model_manager, cols_to_write)

    print("All tasks processed")
    input_db.close_connection()
    output_db.close_connection()
