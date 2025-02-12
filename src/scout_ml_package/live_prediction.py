import time

import pandas as pd

# from scout_ml_package.utils.logger import configure_logger
from scout_ml_package.data.fetch_db_data import DatabaseFetcher
from scout_ml_package.model.model_pipeline import ModelManager, PredictionPipeline
from scout_ml_package.utils.logger import Logger
from scout_ml_package.utils.validator import DataValidator, DummyData, FakeListener

# logger = configure_logger('prediction_logger', '/data/model-data/logs/prediction_logger.log')
# logger = configure_logger('demo_logger', '/data/model-data/logs', 'pred.log')


logger = Logger("demo_logger", "/data/model-data/logs", "pred.log").get_logger()

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


def get_prediction(model_manager, r):
    start_time = time.time()

    if r is None or r.empty:
        logger.error("DataFrame is empty or input data is None.")
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


if __name__ == "__main__":
    df = DummyData.fetch_data()
    # base_path = "/Users/tasnuvachowdhury/Desktop/PROD/pandaml-test/src/"
    base_path = "/data/model-data/"  # "/data/test/"
    model_manager = ModelManager(base_path)
    model_manager.load_models()

    # db_fetcher = DatabaseFetcher()
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

    sample_tasks = [
        30752901,
        27766704,
        30749131,
    ]  # [27766704, 27746332, 30749131, 30752901]
    listener = FakeListener(sample_tasks, delay=200)  # Pass delay here
    for jeditaskid in listener.demo_task_listener():  # No arguments needed here
        print(f"Received JEDITASKID: {jeditaskid}")
        r = input_db.fetch_task_param(jeditaskid)
        # r = df[df['JEDITASKID'] == jeditaskid].copy()
        print(r)
        result = get_prediction(model_manager, r)
        print(result)

        if isinstance(result, pd.DataFrame):
            # logger.info("Processing completed successfully")
            print(result.columns)
            result = result[cols_to_write]
            output_db.write_data(result, "ATLAS_PANDA.PANDAMLTEST")
        else:
            logger.error(f"Processing failed: {result}")
            print(r)
            error_df = r.copy()  # Copy the original DataFrame
            error_df["ERROR"] = result  # Add the error message as a new column

            # Add dummy columns if necessary to match the schema of the main table
            for col in cols_to_write:
                if col not in error_df.columns:
                    error_df[col] = None
            output_db.write_data(
                error_df[cols_to_write + ["ERROR"]], "ATLAS_PANDA.PANDAMLTEST"
            )
        print("Next ID")
        print(result)

    print("All tasks processed")
    input_db.close_connection()
    output_db.close_connection()
