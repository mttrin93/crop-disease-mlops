variable "lambda_function_name" {
  type = string
}

variable "image_uri" {
  description = "Full ECR image URI including tag"
  type        = string
}

variable "model_bucket" {
  description = "S3 bucket name for model artifacts"
  type        = string
}

variable "run_id" {
  description = "MLflow RUN_ID of the active model"
  type        = string
  default     = ""
}

variable "mlflow_tracking_uri" {
  description = "MLflow tracking server URI"
  type        = string
  default     = ""
}

variable "mlflow_experiment_id" {
  description = "MLflow experiment ID"
  type = string
  default = ""
}

variable "aws_region" {
  type = string
}

variable "environment" {
  type    = string
  default = "stg"
}
