"""Unit tests for api/predict.py."""

# pylint: disable=redefined-outer-name

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from api.predict import (
    IMAGE_SIZE,
    IMAGENET_STD,
    IMAGENET_MEAN,
    ModelService,
    get_run_id,
    preprocess_image,
)

# ---------- fixtures ----------


def make_image_bytes(
    width: int = 256, height: int = 256, color=(100, 150, 200)
) -> bytes:
    """Create a fake JPEG image and return its bytes."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture()
def class_names() -> list[str]:
    return [
        "Apple___Apple_scab",
        "Apple___healthy",
        "Tomato___Late_blight",
        "Tomato___healthy",
    ]


@pytest.fixture()
def metadata(class_names) -> dict:
    return {
        "class_names": class_names,
        "class_to_index": {name: i for i, name in enumerate(class_names)},
        "index_to_class": {str(i): name for i, name in enumerate(class_names)},
        "num_classes": len(class_names),
    }


@pytest.fixture()
def mock_session(class_names):
    """Mock ONNX InferenceSession that returns uniform logits."""
    session = MagicMock()
    session.get_inputs.return_value = [MagicMock(name="image")]
    n = len(class_names)
    # logits that make class 2 (Tomato___Late_blight) the winner
    logits = np.zeros((1, n), dtype=np.float32)
    logits[0, 2] = 5.0
    session.run.return_value = [logits]
    return session


@pytest.fixture()
def model_service(mock_session, metadata) -> ModelService:
    return ModelService(mock_session, metadata, run_id="test_run_123")


# ---------- get_run_id ----------


def test_get_run_id_from_env_var(monkeypatch):
    monkeypatch.setenv("RUN_ID", "abc123")
    assert get_run_id() == "abc123"


def test_get_run_id_prefers_env_var_over_ssm(monkeypatch):
    """Env var must win even if SSM would return something different."""
    monkeypatch.setenv("RUN_ID", "from_env")
    # if SSM were called it would raise — this test verifies it isn't
    with patch("api.predict.boto3.client") as mock_boto:
        result = get_run_id()
    mock_boto.assert_not_called()
    assert result == "from_env"


def test_get_run_id_from_ssm(monkeypatch):
    monkeypatch.delenv("RUN_ID", raising=False)
    mock_ssm = MagicMock()
    mock_ssm.get_parameter.return_value = {"Parameter": {"Value": "ssm_run_id_456"}}
    with patch("api.predict.boto3.client", return_value=mock_ssm):
        result = get_run_id()
    assert result == "ssm_run_id_456"


def test_get_run_id_ssm_uses_correct_parameter(monkeypatch):
    monkeypatch.delenv("RUN_ID", raising=False)
    monkeypatch.setenv("SSM_PARAMETER_NAME", "/my/custom/param")
    mock_ssm = MagicMock()
    mock_ssm.get_parameter.return_value = {"Parameter": {"Value": "xyz"}}
    with patch("api.predict.boto3.client", return_value=mock_ssm):
        get_run_id()
    mock_ssm.get_parameter.assert_called_once_with(Name="/my/custom/param")


# ---------- preprocess_image ----------


def test_preprocess_image_output_shape():
    arr = preprocess_image(make_image_bytes())
    assert arr.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_preprocess_image_dtype():
    arr = preprocess_image(make_image_bytes())
    assert arr.dtype == np.float32


def test_preprocess_image_normalized():
    """Output values should be roughly in [-3, 3] after ImageNet normalization."""
    arr = preprocess_image(make_image_bytes())
    assert arr.min() > -4.0
    assert arr.max() < 4.0


def test_preprocess_image_small_input():
    arr = preprocess_image(make_image_bytes(width=32, height=32))
    assert arr.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_preprocess_image_large_input():
    arr = preprocess_image(make_image_bytes(width=1024, height=768))
    assert arr.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_preprocess_image_rgba_converted():
    """RGBA images must be converted to RGB without error."""
    buf = io.BytesIO()
    Image.new("RGBA", (128, 128)).save(buf, "PNG")
    arr = preprocess_image(buf.getvalue())
    assert arr.shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)


def test_preprocess_image_mean_subtracted():
    """A constant image of value 0.5 after /255 should be close to (0.5-mean)/std."""
    solid_color = (int(0.5 * 255), int(0.5 * 255), int(0.5 * 255))
    arr = preprocess_image(make_image_bytes(color=solid_color))
    expected_r = (0.5 - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    assert abs(arr[0, 0].mean() - expected_r) < 0.1


# ---------- ModelService ----------


def test_model_service_predict_returns_dict(model_service):
    result = model_service.predict(make_image_bytes())
    assert isinstance(result, dict)


def test_model_service_predict_keys(model_service):
    result = model_service.predict(make_image_bytes())
    assert {"class_name", "confidence", "top_k", "run_id"} <= result.keys()


def test_model_service_predict_correct_class(model_service):
    result = model_service.predict(make_image_bytes())
    assert result["class_name"] == "Tomato___Late_blight"


def test_model_service_predict_confidence_range(model_service):
    result = model_service.predict(make_image_bytes())
    assert 0.0 <= result["confidence"] <= 1.0


def test_model_service_predict_run_id(model_service):
    result = model_service.predict(make_image_bytes())
    assert result["run_id"] == "test_run_123"


def test_model_service_predict_top_k_length(model_service):
    result = model_service.predict(make_image_bytes(), top_k=3)
    assert len(result["top_k"]) == 3


def test_model_service_predict_top_k_sorted(model_service):
    result = model_service.predict(make_image_bytes(), top_k=4)
    confidences = [item["confidence"] for item in result["top_k"]]
    assert confidences == sorted(confidences, reverse=True)


def test_model_service_predict_probabilities_sum_to_one(model_service):
    result = model_service.predict(make_image_bytes(), top_k=4)
    total = sum(item["confidence"] for item in result["top_k"])
    assert abs(total - 1.0) < 1e-5


def test_model_service_class_names(model_service, class_names):
    assert model_service.class_names == class_names


def test_model_service_session_called(model_service, mock_session):
    model_service.predict(make_image_bytes())
    mock_session.run.assert_called_once()
