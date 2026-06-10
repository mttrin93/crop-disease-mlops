# Crop Disease Detection — Inference Service
#
# Dual-mode image:
#   Local Docker:  docker run -p 8000:8000 -e ... crop-disease-inference:latest
#   AWS Lambda:    CMD ["api.main.handler"] (default)
#
# The Lambda base image includes the Runtime Interface Emulator (RIE)
# for local testing: POST http://localhost:8080/2015-03-31/functions/function/invocations

FROM public.ecr.aws/lambda/python:3.12

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /var/task

# install dependencies into system Python (Lambda runtime uses it directly)
COPY api/pyproject.toml ./
RUN uv pip install --system .

# copy application code
COPY api/ ./api/

# Lambda handler — Mangum adapts FastAPI for API Gateway events
CMD ["api.main.handler"]
