# Standard library imports
import os
from typing import Tuple

# Third-party imports
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    root_mean_squared_error,
    r2_score,
)


class ErrorMetricsPlotter:
    """
    Calculates and plots error metrics for actual vs predicted values.

    Attributes:
    - df (pd.DataFrame): DataFrame containing actual and predicted values.
    - actual_column (str): Name of the column with actual values.
    - predicted_column (str): Name of the column with predicted values.
    - plot_directory (str): Directory where plots will be saved.
    - actual (pd.Series): Series of actual values.
    - predicted (pd.Series): Series of predicted values.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        actual_column: str = "RAMCOUNT",
        predicted_column: str = "Predicted_RAMCOUNT",
        plot_directory: str = "plots",
    ):
        """
        Initializes the ErrorMetricsPlotter with a DataFrame and column names.

        Parameters:
        - df (pd.DataFrame): DataFrame containing actual and predicted values.
        - actual_column (str, optional): Name of the column with actual values. Defaults to "RAMCOUNT".
        - predicted_column (str, optional): Name of the column with predicted values. Defaults to "Predicted_RAMCOUNT".
        - plot_directory (str, optional): Directory where plots will be saved. Defaults to "plots".

        Raises:
        - TypeError: If df is not a DataFrame or if actual_column/predicted_column are not strings.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not isinstance(actual_column, str) or not isinstance(predicted_column, str):
            raise TypeError("actual_column and predicted_column must be strings")
        if not isinstance(plot_directory, str):
            raise TypeError("plot_directory must be a string")

        self.df = df
        self.actual_column = actual_column
        self.predicted_column = predicted_column
        self.plot_directory = plot_directory
        self.actual = df[actual_column]
        self.predicted = df[predicted_column]

        # Ensure the plot directory exists
        os.makedirs(self.plot_directory, exist_ok=True)

    def calculate_metrics(self) -> Tuple[float, float, float, float]:
        """
        Calculates error metrics: MAE, RMSE, R², and MAPE.

        Returns:
        - Tuple[float, float, float, float]: MAE, RMSE, R², and MAPE.
        """
        mae = mean_absolute_error(self.actual, self.predicted)
        rmse = root_mean_squared_error(self.actual, self.predicted)
        r2 = r2_score(self.actual, self.predicted)

        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = (
            abs((self.actual - self.predicted) / self.actual).replace([float("inf"), -float("inf")], 0).fillna(0)
        ).mean() * 100
        return mae, rmse, r2, mape

    def print_metrics(self) -> None:
        """
        Prints error metrics.
        """
        mae, rmse, r2, mape = self.calculate_metrics()
        print(f"Metrics for {self.predicted_column} compared to {self.actual_column}:")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"R-squared (R²): {r2}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

    def plot_metrics(self) -> None:
        """
        Plots error metrics and saves the plot to the specified directory.
        """
        mae, rmse, r2, mape = self.calculate_metrics()

        plt.figure(figsize=(18, 6))

        # Plot Actual vs Predicted
        plt.subplot(1, 3, 1)
        plt.scatter(self.actual, self.predicted, alpha=0.7)
        plt.plot(
            [self.actual.min(), self.actual.max()],
            [self.actual.min(), self.actual.max()],
            "r--",
        )
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted")

        metrics_text = f"MAE: {mae:.2f}\n" f"RMSE: {rmse:.2f}\n" f"R²: {r2:.2f}\n" f"MAPE: {mape:.2f}%"
        plt.text(
            0.05,
            0.95,
            metrics_text,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            backgroundcolor="white",
        )

        # Plot Histogram of Errors
        errors = self.actual - self.predicted
        plt.subplot(1, 3, 2)
        plt.hist(errors, bins=20, color="gray", edgecolor="black", alpha=0.7)
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.title("Histogram of Prediction Errors")

        # Plot Distributions of Actual vs Predicted
        plt.subplot(1, 3, 3)
        plt.hist(
            self.actual,
            bins=20,
            alpha=0.5,
            label="Actual",
            color="blue",
            edgecolor="black",
        )
        plt.hist(
            self.predicted,
            bins=20,
            alpha=0.5,
            label="Predicted",
            color="orange",
            edgecolor="black",
        )
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Distribution of Actual and Predicted Values")
        plt.legend(loc="upper right")

        plot_path = os.path.join(self.plot_directory, f"plot_{self.predicted_column}.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

        # Optionally display the plot
        # plt.show()


class ClassificationMetricsPlotter:
    """
    Plots and calculates metrics for classification models.

    Attributes:
    - df (pd.DataFrame): DataFrame containing actual and predicted values.
    - actual_column (str): Name of the column with actual values.
    - predicted_column (str): Name of the column with predicted values.
    - plot_directory (str): Directory where plots will be saved.
    - actual (pd.Series): Series of actual values.
    - predicted (pd.Series): Series of predicted values.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        actual_column: str,
        predicted_column: str,
        plot_directory: str = "plots",
    ):
        """
        Initializes the ClassificationMetricsPlotter with a DataFrame and column names.

        Parameters:
        - df (pd.DataFrame): DataFrame containing actual and predicted values.
        - actual_column (str): Name of the column with actual values.
        - predicted_column (str): Name of the column with predicted values.
        - plot_directory (str, optional): Directory where plots will be saved. Defaults to "plots".

        Raises:
        - TypeError: If df is not a DataFrame or if actual_column/predicted_column are not strings.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")
        if not isinstance(actual_column, str) or not isinstance(predicted_column, str):
            raise TypeError("actual_column and predicted_column must be strings")
        if not isinstance(plot_directory, str):
            raise TypeError("plot_directory must be a string")

        self.df = df
        self.actual_column = actual_column
        self.predicted_column = predicted_column
        self.plot_directory = plot_directory
        self.actual = df[actual_column]
        self.predicted = df[predicted_column]

        # Ensure the plot directory exists
        os.makedirs(self.plot_directory, exist_ok=True)

    def plot_confusion_matrix(self) -> None:
        """
        Plots the confusion matrix for actual vs predicted values.
        """
        # Compute confusion matrix
        cm = confusion_matrix(self.actual, self.predicted)

        # Display the confusion matrix using matplotlib
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)

        # Save the confusion matrix plot
        plot_path = os.path.join(
            self.plot_directory,
            f"confusion_matrix_{self.predicted_column}.png",
        )
        plt.title("Confusion Matrix")
        plt.savefig(plot_path)
        plt.close()
        print(f"Confusion matrix plot saved to {plot_path}")

    def calculate_and_print_metrics(self) -> Tuple[float, float, float, float]:
        """
        Calculates and prints evaluation metrics for classification.

        Returns:
        - Tuple[float, float, float, float]: Accuracy, Precision, Recall, F1 Score.
        """
        # Calculate evaluation metrics for classification
        accuracy = accuracy_score(self.actual, self.predicted)
        precision = precision_score(self.actual, self.predicted, average="weighted")  # Use 'binary' for binary tasks
        recall = recall_score(self.actual, self.predicted, average="weighted")  # Use 'binary' for binary tasks
        f1 = f1_score(self.actual, self.predicted, average="weighted")  # Use 'binary' for binary tasks

        # Print the evaluation metrics
        print(f"Metrics for {self.predicted_column} compared to {self.actual_column}:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1 Score: {f1:.2f}")

        # Optional: Print classification report for detailed metrics per class
        report = classification_report(
            self.actual, self.predicted, target_names=["Class 0", "Class 1"]
        )  # Modify target_names as needed
        print("Classification Report:\n", report)

        # Additional metrics like confusion matrix can also be printed if needed:
        cm = confusion_matrix(self.actual, self.predicted)
        print("Confusion Matrix:\n", cm)
        return accuracy, precision, recall, f1
