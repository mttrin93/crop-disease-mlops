resource "aws_ecr_repository" "main" {
  name                 = var.ecr_repo_name
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = {
    Project   = "crop-disease-mlops"
    ManagedBy = "terraform"
  }
}

# ── Bootstrap: build and push initial image so Lambda can be created ──────────
#
# In production CI/CD, image build/push is handled by GitHub Actions.
# This null_resource only runs on first apply (or when Dockerfile/main.py change)
# to solve the chicken-and-egg problem: Lambda needs an image to be created,
# but the image doesn't exist yet.
#
# Requires: Docker installed on the machine running Terraform.

resource "null_resource" "ecr_image_bootstrap" {
  triggers = {
    # path.root = infrastructure/ so ../ = project root
    #dockerfile = filemd5("${path.root}/../api/Dockerfile")
#    dockerfile = md5(file(var.docker_image_local_path))
    #entrypoint = filemd5("${path.root}/../api/main.py")
#    entrypoint = md5(file(var.lambda_function_local_path))
    dockerfile = filemd5("${path.root}/../api/Dockerfile")
    entrypoint = filemd5("${path.root}/../api/main.py")
  }

  provisioner "local-exec" {
    working_dir = "${path.root}/.."   # run from project root
    command     = <<EOF
      aws ecr get-login-password --region ${var.aws_region} | \
        docker login --username AWS --password-stdin \
        ${var.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com

      docker build -t ${aws_ecr_repository.main.repository_url}:${var.ecr_image_tag} \
        -f api/Dockerfile .

      docker push ${aws_ecr_repository.main.repository_url}:${var.ecr_image_tag}
    EOF
  }

  depends_on = [aws_ecr_repository.main]
}

# TODO: see the conversation on chatgpt and apply the changes here and in output
#data "aws_ecr_image" "main" {
#  depends_on = [
#    null_resource.ecr_image_bootstrap
#  ]

#  repository_name = aws_ecr_repository.main.name
#  image_tag       = var.ecr_image_tag
#}
