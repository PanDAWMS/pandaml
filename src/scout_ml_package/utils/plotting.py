import os
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


class ErrorMetricsPlotter:
    def __init__(
        self,
        df: pd.DataFrame,
        actual_column: str = "RAMCOUNT",
        predicted_column: str = "Predicted_RAMCOUNT",
        plot_directory: str = "plots",
    ):
        self.df = df
        self.actual_column = actual_column
        self.predicted_column = predicted_column
        self.plot_directory = plot_directory
        self.actual = df[actual_column]
        self.predicted = df[predicted_column]

        # Ensure the plot directory exists
        os.makedirs(self.plot_directory, exist_ok=True)

    def calculate_metrics(self):
        mae = mean_absolute_error(self.actual, self.predicted)
        rmse = root_mean_squared_error(self.actual, self.predicted)
        r2 = r2_score(self.actual, self.predicted)

        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = (
            abs((self.actual - self.predicted) / self.actual)
            .replace([float("inf"), -float("inf")], 0)
            .fillna(0)
        ).mean() * 100
        return mae, rmse, r2, mape

    def print_metrics(self):
        mae, rmse, r2, mape = self.calculate_metrics()
        print(
            f"Metrics for {self.predicted_column} compared to {self.actual_column}:"
        )
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"R-squared (R²): {r2}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape}%")

    def plot_metrics(self):
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

        metrics_text = (
            f"MAE: {mae:.2f}\n"
            f"RMSE: {rmse:.2f}\n"
            f"R²: {r2:.2f}\n"
            f"MAPE: {mape:.2f}%"
        )
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

        plot_path = os.path.join(
            self.plot_directory, f"plot_{self.predicted_column}.png"
        )
        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Plot saved to {plot_path}")

        # Optionally display the plot
        # plt.show()


class ClassificationMetricsPlotter:
    def __init__(
        self,
        df: pd.DataFrame,
        actual_column: str,
        predicted_column: str,
        plot_directory: str = "plots",
    ):
        self.df = df
        self.actual_column = actual_column
        self.predicted_column = predicted_column
        self.plot_directory = plot_directory
        self.actual = df[actual_column]
        self.predicted = df[predicted_column]

        # Ensure the plot directory exists
        os.makedirs(self.plot_directory, exist_ok=True)

    def plot_confusion_matrix(self):
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

    # def plot_classification_report(self):
    #     """Add more plots or adjustments based on your needs, such as precision-recall curves, etc."""
    #     # This method can be expanded to include other classification visualizations
    #     pass

    def calculate_and_print_metrics(self):
        # Calculate evaluation metrics for classification
        accuracy = accuracy_score(self.actual, self.predicted)
        precision = precision_score(
            self.actual, self.predicted, average="weighted"
        )  # Use 'binary' for binary tasks
        recall = recall_score(
            self.actual, self.predicted, average="weighted"
        )  # Use 'binary' for binary tasks
        f1 = f1_score(
            self.actual, self.predicted, average="weighted"
        )  # Use 'binary' for binary tasks

        # Print the evaluation metrics
        print(
            f"Metrics for {self.predicted_column} compared to {self.actual_column}:"
        )
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


# Example usage:
# df = pd.DataFrame(...)  # your DataFrame with actual and predicted columns
# plotter = ErrorMetricsPlotter(df)
# plotter.print_metrics()
# plotter.plot_metrics()


# class PredictionVisualizer:
#     def __init__(self, trained_model: TrainedModel):
#         """
#         Initialize the PredictionVisualizer.
#
#         Args:
#             trained_model (TrainedModel): The trained model to visualize predictions for.
#         """
#         self.trained_model = trained_model
#
#     def plot_predictions(
#         self,
#         X_test: pd.DataFrame,
#         y_test: pd.DataFrame,
#         num_samples: int = 100,
#     ):
#         """
#         Plot predictions against actual values.
#
#         Args:
#             X_test (pd.DataFrame): The test input data.
#             y_test (pd.DataFrame): The test target data.
#             num_samples (int): Number of samples to plot. Defaults to 100.
#         """
#         y_pred = self.trained_model.predict(X_test)
#         num_targets = y_pred.shape[1]
#         fig, axes = plt.subplots(num_targets, 2, figsize=(15, 5 * num_targets))
#
#         for i in range(num_targets):
#             ax1 = axes[i, 0]
#             ax2 = axes[i, 1]
#
#             # Scatter plot
#             sns.scatterplot(
#                 x=y_test.iloc[:num_samples, i],
#                 y=y_pred[:num_samples, i],
#                 ax=ax1,
#             )
#             ax1.set_xlabel("Actual")
#             ax1.set_ylabel("Predicted")
#             ax1.set_title(f"Target {i + 1} - Scatter Plot")
#
#             # Histogram plot
#             actual_values = y_test.iloc[:num_samples, i]
#             predicted_values = y_pred[:num_samples, i]
#             ax2.hist(
#                 [actual_values, predicted_values],
#                 bins=20,
#                 label=["Actual", "Predicted"],
#                 density=True,
#             )
#             ax2.set_xlabel("Value")
#             ax2.set_ylabel("Density")
#             ax2.set_title(f"Target {i + 1} - Histogram")
#             ax2.legend()
#
#         plt.tight_layout()
#         plt.show()