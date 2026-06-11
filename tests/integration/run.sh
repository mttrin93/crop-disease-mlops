#!/usr/bin/env bash
# Integration test runner for the crop disease inference service.
# Starts the Docker container, waits for it to be ready, runs tests, teardown.

set -e
# load .env
set -a
source .env
set +a


LOCAL_IMAGE_NAME="crop-disease-inference:latest"
#MODEL_LOCATION=${MODEL_LOCATION:-"$(pwd)/models"}
CONTAINER_NAME="crop-disease-test"
PORT=8000
echo ${MLFLOW_TRACKING_URI}

echo "Starting container ${LOCAL_IMAGE_NAME}..."
docker run --rm \
    -d \
    --name "${CONTAINER_NAME}" \
    -p "${PORT}:${PORT}" \
    -v ~/.aws:/root/.aws:ro \
    -e MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI} \
    --entrypoint uvicorn \
    "${LOCAL_IMAGE_NAME}" \
    api.main:app --host 0.0.0.0 --port "${PORT}"

# wait for the service to be ready (max 30 seconds)
echo "Waiting for service to be ready..."
for i in $(seq 1 30); do
    if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "Service is ready."
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "Service failed to start after 30s"
        docker logs "${CONTAINER_NAME}"
        docker rm -f "${CONTAINER_NAME}"
        exit 1
    fi
    sleep 1
done

# run integration tests
echo "Running integration tests..."
API_URL="http://localhost:${PORT}" \
    python -m pytest tests/integration/test_api.py -v

TEST_EXIT_CODE=$?

# teardown
echo "Stopping container..."
docker rm -f "${CONTAINER_NAME}"

exit ${TEST_EXIT_CODE}
