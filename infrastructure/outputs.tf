output "model_bucket" {
  description = "S3 bucket name for data and model artifacts"
  value       = module.s3_bucket.name
}

output "ecr_repo" {
  description = "ECR repository name (used by CI/CD to tag and push images)"
  value       = module.ecr.repository_name
}

output "ecr_repo_url" {
  description = "Full ECR repository URL"
  value       = module.ecr.repository_url
}

output "lambda_function" {
  description = "Lambda function name (used by CD to update env vars)"
  value       = module.lambda.function_name
}

output "api_endpoint" {
  description = "Public HTTPS endpoint for the inference API"
  value       = module.lambda.api_endpoint
}

output "predictions_stream_name" {
  description = "Kept for compatibility — not used in this project"
  value       = ""
}
