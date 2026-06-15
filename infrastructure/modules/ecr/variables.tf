variable "ecr_repo_name" {
  type = string
}

variable "aws_region" {
  type = string
}

variable "account_id" {
  type = string
}

variable "ecr_image_tag" {
    type        = string
    description = "ECR repo name"
    default = "latest"
}

variable "lambda_function_local_path" {
    type        = string
    description = "Local path to lambda function / python file"
}

variable "docker_image_local_path" {
    type        = string
    description = "Local path to Dockerfile"
}
