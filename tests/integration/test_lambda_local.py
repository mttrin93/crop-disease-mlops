"""
Local test script for the Lambda container via the RIE.

Builds a properly encoded API Gateway v1 event with multipart form data
and sends it to the Lambda Runtime Interface Emulator.

Usage:
    python test_lambda_local.py --image data/processed/test/Tomato___healthy/000004.jpg
    python test_lambda_local.py --health
"""

import json
import base64
import argparse

import requests

RIE_URL = "http://localhost:8080/2015-03-31/functions/function/invocations"
BOUNDARY = "boundary123"


def make_api_gateway_event(
    method: str,
    path: str,
    headers: dict,
    body: bytes | None = None,
) -> dict:
    """Build a minimal API Gateway v1 event that Mangum recognises."""
    encoded_body = None
    is_base64 = False

    if body is not None:
        encoded_body = base64.b64encode(body).decode()
        is_base64 = True

    return {
        "httpMethod": method,
        "path": path,
        "headers": headers,
        "multiValueHeaders": {k: [v] for k, v in headers.items()},
        "queryStringParameters": None,
        "multiValueQueryStringParameters": None,
        "pathParameters": None,
        "stageVariables": None,
        # requestContext is required for Mangum to identify the handler
        "requestContext": {
            "resourcePath": path,
            "httpMethod": method,
            "stage": "local",
            "requestId": "test-request-id",
            "identity": {"sourceIp": "127.0.0.1"},
        },
        "body": encoded_body,
        "isBase64Encoded": is_base64,
        "resource": path,
    }


def build_multipart_body(
    image_bytes: bytes, filename: str = "image.jpg"
) -> tuple[bytes, str]:
    """Build a multipart/form-data body for file upload."""
    content_type = "image/jpeg" if filename.endswith(".jpg") else "image/png"
    body = (
        (
            f"--{BOUNDARY}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n"
            f"\r\n"
        ).encode()
        + image_bytes
        + f"\r\n--{BOUNDARY}--\r\n".encode()
    )

    content_type_header = f"multipart/form-data; boundary={BOUNDARY}"
    return body, content_type_header


def test_health():
    """Test the /health endpoint."""
    print("Testing /health...")
    event = make_api_gateway_event(
        method="GET",
        path="/health",
        headers={"Accept": "application/json"},
    )
    response = requests.post(RIE_URL, json=event, timeout=30)
    result = response.json()
    print(f"Status: {response.status_code}")
    print(json.dumps(result, indent=2))
    return result


def test_predict(image_path: str):
    """Test the /predict endpoint with a local image."""
    print(f"Testing /predict with {image_path}...")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    filename = image_path.split("/")[-1]
    body, content_type = build_multipart_body(image_bytes, filename)

    event = make_api_gateway_event(
        method="POST",
        path="/predict",
        headers={
            "Content-Type": content_type,
            "Content-Length": str(len(body)),
        },
        body=body,
    )

    response = requests.post(RIE_URL, json=event, timeout=60)
    result = response.json()
    print(f"Status: {response.status_code}")
    print(json.dumps(result, indent=2))
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Test Lambda container locally")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--health", action="store_true", help="Test health endpoint")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.health:
        test_health()
    elif args.image:
        test_predict(args.image)
    else:
        # run both by default
        test_health()
        print()
        test_predict("data/processed/test/Tomato___healthy/000004.jpg")
