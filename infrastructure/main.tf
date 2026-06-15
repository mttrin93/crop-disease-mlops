# ──────────────────────────────────────────────────────────────────────────────
# Crop Disease Detection — AWS Infrastructure
#
# Resources created:
#   - S3 bucket        : data + model artifacts + MLflow artifacts
#   - ECR repository   : Docker image for Lambda inference service
#   - Lambda function  : inference API (container image)
#   - API Gateway (v2) : HTTP endpoint → Lambda
#   - IAM roles        : Lambda execution role with least-privilege policies
#
# State stored in: s3://tf-state-crop-disease-mlops-1/crop-disease-mlops/stg.tfstate
#
# Usage:
#   terraform init
#   terraform plan
#   terraform apply
#   terraform destroy
# ──────────────────────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  backend "s3" {
    bucket  = "tf-state-crop-disease-mlops-1"
    key     = "crop-disease-mlops/stg.tfstate"
    region  = "eu-west-1"
    encrypt = true
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

locals {
  account_id  = data.aws_caller_identity.current.account_id
  name_prefix = "${var.environment}-${var.project_id}"
}

# ── S3 bucket (data + model artifacts + MLflow artifacts) ─────────────────────

module "s3_bucket" {
  source      = "./modules/s3"
  bucket_name = "${var.model_bucket_name}-${local.account_id}"
}

# ── ECR repository + initial Docker image push ────────────────────────────────

module "ecr" {
  source        = "./modules/ecr"
  ecr_repo_name = "${var.ecr_repo_name}_${var.project_id}"
  aws_region    = var.aws_region
  account_id    = local.account_id
  lambda_function_local_path = var.lambda_function_local_path
  docker_image_local_path = var.docker_image_local_path
}

# ── Lambda inference function + API Gateway ────────────────────────────────────

module "lambda" {
  source               = "./modules/lambda"
  lambda_function_name = "${var.lambda_function_name}_${var.project_id}"
  image_uri            = module.ecr.image_uri
  model_bucket         = module.s3_bucket.name
  run_id               = var.run_id
  mlflow_tracking_uri  = var.mlflow_tracking_uri
  mlflow_experiment_id = var.mlflow_experiment_id
  aws_region           = var.aws_region
  environment          = var.environment
  depends_on = [module.ecr]
}
