variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "eu-west-1"
}

variable "environment" {
  description = "Deployment environment prefix (stg or prod)"
  type        = string
  default     = "stg"
}

variable "project_id" {
  description = "Project identifier used in resource names"
  type        = string
  default     = "crop-disease-mlops"
}

variable "model_bucket_name" {
  description = "Base name for the S3 bucket (account_id appended automatically)"
  type        = string
  default     = "crop-disease-models-mlflow-stg"
}

variable "ecr_repo_name" {
  description = "ECR repository base name"
  type        = string
  default     = "crop-disease-inference"
}

variable "lambda_function_name" {
  description = "Lambda function base name"
  type        = string
  default     = "crop-disease-predict"
}

variable "run_id" {
  description = "MLflow RUN_ID of the active production model"
  type        = string
  default     = ""
}

variable "mlflow_tracking_uri" {
  description = "MLflow tracking server URI (EC2 instance)"
  type        = string
  default     = ""
}

variable "mlflow_experiment_id" {
  description = "MLflow experiment ID"
  type        = string
  default     = ""
}

variable "docker_image_local_path" {
  description = "Path to Dockerfile for null_resource ECR bootstrap"
  type        = string
  default     = ""
}

variable "lambda_function_local_path" {
  description = "Path to lambda entry point for null_resource trigger hash"
  type        = string
  default     = ""
}
