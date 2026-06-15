# ── IAM role for Lambda ───────────────────────────────────────────────────────

resource "aws_iam_role" "lambda_execution" {
  name = "lambda-execution-${var.lambda_function_name}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
    }]
  })

  tags = { ManagedBy = "terraform" }
}

# CloudWatch Logs
resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# S3: read model artifacts
resource "aws_iam_role_policy" "lambda_s3" {
  name = "lambda-s3-${var.lambda_function_name}"
  role = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["s3:GetObject", "s3:ListBucket"]
      Resource = [
        "arn:aws:s3:::${var.model_bucket}",
        "arn:aws:s3:::${var.model_bucket}/*"
      ]
    }]
  })
}

# SSM: read RUN_ID parameter
resource "aws_iam_role_policy" "lambda_ssm" {
  name = "lambda-ssm-${var.lambda_function_name}"
  role = aws_iam_role.lambda_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = ["ssm:GetParameter"]
      Resource = "arn:aws:ssm:${var.aws_region}:*:parameter/crop-disease-mlops/*"
    }]
  })
}

# ── Lambda function ───────────────────────────────────────────────────────────

resource "aws_lambda_function" "inference" {
  function_name = var.lambda_function_name
  role          = aws_iam_role.lambda_execution.arn
  package_type  = "Image"
  image_uri     = var.image_uri
  timeout       = 60    # cold start with model download can take ~30s
  memory_size   = 1024  # ONNX inference needs enough memory

  environment {
    variables = {
      RUN_ID               = var.run_id
      MODEL_BUCKET         = var.model_bucket
      MLFLOW_TRACKING_URI  = var.mlflow_tracking_uri
      MLFLOW_EXPERIMENT_ID = var.mlflow_experiment_id
    }
  }

  tags = {
    Project     = "crop-disease-mlops"
    Environment = var.environment
    ManagedBy   = "terraform"
  }
}

# ── API Gateway v2 (HTTP API) ─────────────────────────────────────────────────

resource "aws_apigatewayv2_api" "main" {
  name          = "crop-disease-api-${var.environment}"
  protocol_type = "HTTP"
  description   = "Crop disease detection inference API"

  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["GET", "POST"]
    allow_headers = ["Content-Type", "Authorization"]
  }

  tags = { ManagedBy = "terraform" }
}

resource "aws_apigatewayv2_integration" "lambda" {
  api_id                 = aws_apigatewayv2_api.main.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.inference.invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "health" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_route" "predict" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.lambda.id}"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.main.id
  name        = "$default"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway.arn
    format = jsonencode({
      requestId        = "$context.requestId"
      sourceIp         = "$context.identity.sourceIp"
      requestTime      = "$context.requestTime"
      httpMethod       = "$context.httpMethod"
      routeKey         = "$context.routeKey"
      status           = "$context.status"
      responseLength   = "$context.responseLength"
      integrationError = "$context.integrationErrorMessage"
    })
  }
}

# Allow API Gateway to invoke Lambda
resource "aws_lambda_permission" "api_gateway" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.inference.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.main.execution_arn}/*/*"
}

# ── CloudWatch log groups ─────────────────────────────────────────────────────

resource "aws_cloudwatch_log_group" "lambda" {
  name              = "/aws/lambda/${var.lambda_function_name}"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/apigateway/crop-disease-${var.environment}"
  retention_in_days = 7
}
