from unittest.mock import Mock, patch

import pytest
from pandas import DataFrame

from scout_ml_package.live_prediction import get_prediction
from scout_ml_package.model.model_pipeline import ModelManager


# Mocking setup
@pytest.fixture
def mock_model_manager():
    return Mock(spec=ModelManager)


@pytest.fixture
def mock_data():
    return DataFrame({"JEDITASKID": [12345], "other_column": ["value"]})


def test_get_prediction_empty_data(mock_model_manager):
    result = get_prediction(mock_model_manager, None)
    assert result is None


def test_get_prediction_valid_data(mock_model_manager, mock_data):
    # Mock the prediction pipeline to return valid data
    with patch(
        "scout_ml_package.model.model_pipeline.PredictionPipeline"
    ) as mock_pipeline:
        mock_pipeline.return_value.preprocess_data.return_value = mock_data
        mock_pipeline.return_value.make_predictions_for_model.side_effect = [
            100,
            10,
            50,
        ]  # Mock predictions

        result = get_prediction(mock_model_manager, mock_data)
        assert isinstance(result, DataFrame)


def test_get_prediction_ramcount_validation_failure(mock_model_manager, mock_data):
    # Mock the prediction pipeline to return invalid RAMCOUNT
    with patch(
        "scout_ml_package.model.model_pipeline.PredictionPipeline"
    ) as mock_pipeline:
        mock_pipeline.return_value.preprocess_data.return_value = mock_data
        mock_pipeline.return_value.make_predictions_for_model.side_effect = [
            0,
            10,
            50,
        ]  # Invalid RAMCOUNT

        result = get_prediction(mock_model_manager, mock_data)
        assert isinstance(result, str) and "M1 failure" in result


def test_get_prediction_ctime_validation_failure(mock_model_manager, mock_data):
    # Mock the prediction pipeline to return invalid CTIME
    with patch(
        "scout_ml_package.model.model_pipeline.PredictionPipeline"
    ) as mock_pipeline:
        mock_pipeline.return_value.preprocess_data.return_value = mock_data
        mock_pipeline.return_value.make_predictions_for_model.side_effect = [
            100,
            0,
            50,
        ]  # Invalid CTIME

        result = get_prediction(mock_model_manager, mock_data)
        assert isinstance(result, str) and "M2 failure" in result


def test_get_prediction_cpu_eff_validation_failure(mock_model_manager, mock_data):
    # Mock the prediction pipeline to return invalid CPU_EFF
    with patch(
        "scout_ml_package.model.model_pipeline.PredictionPipeline"
    ) as mock_pipeline:
        mock_pipeline.return_value.preprocess_data.return_value = mock_data
        mock_pipeline.return_value.make_predictions_for_model.side_effect = [
            100,
            10,
            150,
        ]  # Invalid CPU_EFF

        result = get_prediction(mock_model_manager, mock_data)
        assert isinstance(result, str) and "M4 failure" in result


def test_get_prediction_exception(mock_model_manager, mock_data):
    # Mock an exception during prediction
    with patch(
        "scout_ml_package.model.model_pipeline.PredictionPipeline"
    ) as mock_pipeline:
        mock_pipeline.return_value.make_predictions_for_model.side_effect = Exception(
            "Mocked exception"
        )

        result = get_prediction(mock_model_manager, mock_data)
        assert isinstance(result, str) and "failure" in result
