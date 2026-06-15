"""
Integration test for the crop disease detection API.

Sends a real HTTP request to a running Docker container and checks
the response structure and status codes.

Requires the container to be running:
    docker run -p 8000:8000 \
        -e MODEL_LOCATION=/var/model \
        -v /path/to/model:/var/model \
        crop-disease-inference:latest

Or via the Lambda RIE:
    docker run -p 8080:8080 \
        -e MODEL_LOCATION=/var/model \
        -v /path/to/model:/var/model \
        crop-disease-inference:latest

Run with:
    bash tests/integration/run.sh
"""

import io
import os

import requests
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")


def make_test_image(width: int = 224, height: int = 224) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(100, 150, 80)).save(buf, "JPEG")
    buf.seek(0)
    return buf.read()


# ---------- health endpoint ----------


def test_health_returns_200():
    response = requests.get(f"{API_URL}/health", timeout=10)
    assert response.status_code == 200


def test_health_response_structure():
    response = requests.get(f"{API_URL}/health", timeout=10)
    data = response.json()
    assert "status" in data
    assert "run_id" in data
    assert "num_classes" in data


def test_health_status_healthy():
    response = requests.get(f"{API_URL}/health", timeout=10)
    assert response.json()["status"] == "healthy"


def test_health_num_classes():
    response = requests.get(f"{API_URL}/health", timeout=10)
    assert response.json()["num_classes"] == 38


# ---------- predict endpoint ----------


def test_predict_returns_200():
    image_bytes = make_test_image()
    response = requests.post(
        f"{API_URL}/predict",
        files={"file": ("leaf.jpg", image_bytes, "image/jpeg")},
        timeout=30,
    )
    assert response.status_code == 200


def test_predict_response_structure():
    image_bytes = make_test_image()
    response = requests.post(
        f"{API_URL}/predict",
        files={"file": ("leaf.jpg", image_bytes, "image/jpeg")},
        timeout=30,
    )
    data = response.json()
    assert "class_name" in data
    assert "confidence" in data
    assert "top_k" in data
    assert "run_id" in data


def test_predict_confidence_range():
    image_bytes = make_test_image()
    response = requests.post(
        f"{API_URL}/predict",
        files={"file": ("leaf.jpg", image_bytes, "image/jpeg")},
        timeout=30,
    )
    confidence = response.json()["confidence"]
    assert 0.0 <= confidence <= 1.0


def test_predict_class_name_valid():
    image_bytes = make_test_image()
    response = requests.post(
        f"{API_URL}/predict",
        files={"file": ("leaf.jpg", image_bytes, "image/jpeg")},
        timeout=30,
    )
    class_name = response.json()["class_name"]
    assert isinstance(class_name, str)
    assert len(class_name) > 0


def test_predict_top_k_structure():
    image_bytes = make_test_image()
    response = requests.post(
        f"{API_URL}/predict",
        files={"file": ("leaf.jpg", image_bytes, "image/jpeg")},
        timeout=30,
    )
    top_k = response.json()["top_k"]
    assert isinstance(top_k, list)
    assert len(top_k) == 5
    for item in top_k:
        assert "class_name" in item
        assert "confidence" in item


def test_predict_unsupported_format_returns_422():
    response = requests.post(
        f"{API_URL}/predict",
        files={"file": ("file.txt", b"not an image", "text/plain")},
        timeout=10,
    )
    assert response.status_code == 422


def test_predict_png_accepted():
    buf = io.BytesIO()
    Image.new("RGB", (224, 224)).save(buf, "PNG")
    response = requests.post(
        f"{API_URL}/predict",
        files={"file": ("leaf.png", buf.getvalue(), "image/png")},
        timeout=30,
    )
    assert response.status_code == 200
