# src/scout_ml_package/train_model.py

import pandas as pd
import numpy as np
import re
from scout_ml_package.data import (
    DataSplitter,
)
from scout_ml_package.model.model_pipeline import (
    TrainingPipeline,
)
from scout_ml_package.model.base_model import DeviceInfo
from scout_ml_package.utils import ErrorMetricsPlotter

import joblib



def preprocess_data(df):
    """
    Preprocesses the data by converting specific columns.
    """

    # Convert PROCESSINGTYPE to 'P'
    def convert_processingtype(processingtype):
        if processingtype is not None and re.search(r"-.*-", processingtype):
            return "-".join(processingtype.split("-")[-2:])
        return processingtype

    # Convert TRANSHOME to 'F'
    def convert_transhome(transhome):
        if transhome is None:
            return None

        if "AnalysisBase" in transhome:
            return "AnalysisBase"
        elif "AnalysisTransforms-" in transhome:
            part_after_dash = transhome.split("-")[1]
            return part_after_dash.split("_")[0]
        elif "/" in transhome:
            return transhome.split("/")[0]
        else:
            return transhome.split("-")[0]

    # Convert CORECOUNT to 'Core'
    def convert_corecount(corecount):
        return "S" if corecount == 1 else "M"

    df["CORE"] = df["CORECOUNT"].apply(convert_corecount)
    df["P"] = df["PROCESSINGTYPE"].apply(convert_processingtype)
    df["F"] = df["TRANSHOME"].apply(convert_transhome)
    df['CTIME'] = np.where(
        df['CPUTIMEUNIT'] == 'mHS06sPerEvent',
        df['CPUTIME'] / 1000,
        np.where(
            df['CPUTIMEUNIT'] == 'HS06sPerEvent',
            df['CPUTIME'],
            None
        )
    )
    df['CTIME'] = df['CTIME'].astype('float64')
    #KEEP_F_TAG = ['Athena', 'AnalysisBase', 'AtlasOffline', 'AthAnalysis', 'AthSimulation', 'MCProd', 'AthGeneration', 'AthDerivation']
    #KEEP_P_TAG = ['jedi-run', 'deriv', 'athena-trf', 'jedi-athena', 'simul', 'pile', 'merge', 'evgen', 'reprocessing', 'recon']
    KEEP_P_TAG = ['deriv', 'jedi-run', 'merge', 'jedi-athena', 'simul', 'pile', 'evgen', 'recon', 'reprocessing', 'athena-trf', 'run-evp', 'athena-evp']
    KEEP_F_TAG =['AthDerivation', 'AnalysisBase', 'Athena', 'AthAnalysis','AtlasOffline', 'AthGeneration', 'AthSimulation','MCProd']
    df['P'] = df['P'].apply(lambda x: x if x in KEEP_P_TAG else 'others')
    df['F'] = df['F'].apply(lambda x: x if x in KEEP_F_TAG else 'others')
    return df


def main():
    DeviceInfo.print_device_info()
    #DeviceInfo.set_cpu_usage_to_fraction(0.7) 
    # Define base path and load data
    base_path = "/data/"
    data = pd.read_parquet(f"{base_path}merged_files/features.parquet")

    # Preprocess data
    df_ = preprocess_data(data)
    print(df_.columns)
    print(df_.head())
    # Filter data
    df_ = df_[
        (df_["PRODSOURCELABEL"].isin(["user", "managed"]))
        & (df_["CTIME"] > 50)
        & (df_["CTIME"] < 5000)
        & (df_["RAMCOUNT"] < 8000)
        & (df_["RAMCOUNT"] > 100)
        & (df_["CPUTIMEUNIT"] == "HS06sPerEvent")
    ]

    # Split data into training and future sets
    training_data = df_.sample(frac=0.68, random_state=42)
    future_data = df_[~df_.index.isin(training_data.index)]
    future_data = future_data[future_data["CPUTIMEUNIT"] == "HS06sPerEvent"].copy()

    # Define target variable and categorical features
    target_var = ["CTIME"]
    categorical_features = ["PRODSOURCELABEL", "P", "F", "CORE", "CPUTIMEUNIT"]

    # Define unique categories for each categorical feature
    category_list = [
        ["managed", "user"],
        [
            "deriv",
            "pile",
            "jedi-run",
            "simul",
            "athena-trf",
            "jedi-athena",
            "recon",
            "reprocessing",
            "evgen",
            "others",
            "merge",
        ],
        [
            "Athena",
            "AtlasOffline",
            "others",
            "AthDerivation",
            "AnalysisBase",
            "AthAnalysis",
            "AthSimulation",
            "AthGeneration",
            "MCProd",
        ],
        ["M", "S"],
        ["HS06sPerEvent", "mHS06sPerEvent"],
    ]

    # Define selected columns
    selected_columns = [
        "JEDITASKID",
        "PRODSOURCELABEL",
        "P",
        "F",
        "CPUTIMEUNIT",
        "CORE",
        "TOTAL_NFILES",
        "TOTAL_NEVENTS",
        "DISTINCT_DATASETNAME_COUNT",
        "RAMCOUNT",
        "CTIME",
    ]

    # Split training data into train and test sets
    splitter = DataSplitter(training_data, selected_columns)
    train_df, test_df = splitter.split_data(test_size=0.15)

    # Define numerical features
    numerical_features = ["TOTAL_NFILES", "TOTAL_NEVENTS", "DISTINCT_DATASETNAME_COUNT", "RAMCOUNT"]
    print("numerical_features: ",numerical_features )
    # Create training pipeline
    pipeline = TrainingPipeline(numerical_features, categorical_features, category_list, target_var)

    # Preprocess data
    (
        processed_train_data,
        processed_test_data,
        processed_future_data,
        encoded_columns,
        fitted_scalar,
    ) = pipeline.preprocess_data(train_df, test_df, future_data)

    # Define features to train
    features_to_train = encoded_columns + numerical_features

    # Train model
    tuned_model = pipeline.train_model(
        processed_train_data,
        processed_test_data,
        features_to_train,
        "build_cputime_high",
        epoch=1,
        batch=300,
    )

    # Make predictions
    predictions, y_pred = pipeline.regression_prediction(tuned_model, processed_future_data, features_to_train)

    # Define model storage path and name
    model_seq = "3"
    target_name = "cputime_high"
    model_storage_path = f"/data/test-data/ModelStorage/model{model_seq}/"
    model_name = f"model{model_seq}_{target_name}"
    plot_directory_name = f"/data/test-data/ModelStorage/plots/model{model_seq}"

    # Save model
    joblib.dump(fitted_scalar, f"{model_storage_path}/scaler.pkl")

    model_full_path=model_storage_path + model_name
    print(model_full_path)
    tuned_model.export(model_full_path)

    actual_column_name = "CTIME"
    predicted_column_name = "Predicted_CTIME"
    predictions = predictions.dropna()

    print(predictions[predicted_column_name].max())
    print(predictions[predicted_column_name].value_counts())
    # Create an instance of the ErrorMetricsPlotter class
    plotter = ErrorMetricsPlotter(
        predictions,
        actual_column=actual_column_name,
        predicted_column=predicted_column_name,
        plot_directory=plot_directory_name,
    )

    # Print error metrics
    plotter.print_metrics()


if __name__ == "__main__":
    main()
