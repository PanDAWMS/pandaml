# Standard library imports
import queue
import threading
import time
import warnings
from datetime import datetime

# Third-party imports

# Local application imports
from scout_ml_package.data.fetch_db_data import DatabaseFetcher
from scout_ml_package.model.model_pipeline import (
    ModelManager,
)
from scout_ml_package.utils.prediction_utils import PredictionUtils
from scout_ml_package.utils.logger import Logger
from scout_ml_package.utils.message import ConfigLoader, MyListener

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas.io.sql")
last_logged = time.time()

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
logname = f"prediction_{current_datetime}.log"
# Create a custom logger
logger = Logger("scout_logger", "/data/model-data/logs", logname)


if __name__ == "__main__":
    base_path = "/data/model-data/"  # "/data/test/"
    model_manager = ModelManager(base_path)
    model_manager.load_models()

    input_db = DatabaseFetcher("database")
    output_db = DatabaseFetcher("output_database")
    pred_utils = PredictionUtils(logger)
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
    config_path = "/data/model-data/configs/config.ini"  # path to config file
    config_loader = ConfigLoader(config_path)
    # Create queues to hold task IDs
    task_id_queue = queue.Queue()

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
    fetch_thread = threading.Thread(
        target=pred_utils.fetch_and_process,
        args=(task_id_queue, input_db, output_db, model_manager, cols_to_write),
    )
    fetch_thread.daemon = True
    fetch_thread.start()

    # Keep the main thread running

    print_time = time.time()
    while True:
        if time.time() - print_time >= 120:  # Check if 2 minutes have passed
            print("Main thread alive", end="\r")
            print_time = time.time()  # Reset the timer
        time.sleep(1)  # Sleep for 1 second
