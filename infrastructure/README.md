# Infrastructure — Terraform Setup Guide

AWS infrastructure for the crop disease detection service, managed with Terraform.

---

## Resources created

| Resource | Name | Purpose |
|---|---|---|
| S3 bucket | `crop-disease-models-mlflow-stg-478544568263` | Data, model artifacts, MLflow store |
| ECR repository | `crop-disease-inference_crop-disease-mlops` | Docker image for Lambda |
| Lambda function | `crop-disease-predict_crop-disease-mlops` | Inference API (container image) |
| API Gateway (HTTP) | `crop-disease-api-stg` | Public HTTPS endpoint |
| IAM role | `lambda-execution-crop-disease-predict_...` | Least-privilege Lambda permissions |
| CloudWatch log groups | `/aws/lambda/...` `/aws/apigateway/...` | 7-day log retention |

---

## Prerequisites

- Terraform >= 1.0 installed
- AWS CLI configured (`aws configure`)
- Docker installed (for ECR image bootstrap)
- Docker image already built locally:
  ```bash
  docker build -t crop-disease-inference:latest -f api/Dockerfile .
  ```

---

## Step 1 — Create the Terraform state bucket (once, before init)

Terraform needs an S3 bucket to store its state file. This must exist **before**
`terraform init`, Terraform cannot create its own state storage.

```bash
aws s3 mb s3://tf-state-crop-disease-mlops-1 --region eu-west-1
```

The state bucket is referenced in `main.tf`:
```hcl
backend "s3" {
  bucket  = "tf-state-crop-disease-mlops-1"
  key     = "crop-disease-mlops/stg.tfstate"
  region  = "eu-west-1"
  encrypt = true
}
```

---

## Step 2 — Configure variables

```bash
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with your values:

```hcl
aws_region            = "eu-west-1"
environment           = "stg"
project_id            = "crop-disease-mlops"
model_bucket_name     = "crop-disease-models-mlflow-stg"   # base name, account_id appended
ecr_repo_name         = "crop-disease-inference"
lambda_function_name  = "crop-disease-predict"
run_id                = "your_mlflow_run_id"                # from training step
mlflow_tracking_uri   = "http://your-ec2-dns:5000"
```

> `terraform.tfvars` is gitignored — never commit it.

---

## Step 3 — Terraform init

```bash
cd infrastructure
terraform init
```

---

## Step 4 — Import existing S3 bucket

The model bucket already exists (created before Terraform). Import it so
Terraform manages it without recreating it:

```bash
terraform import module.s3_bucket.aws_s3_bucket.main \
    crop-disease-models-mlflow-stg-478544568263
```

> If you get "Resource already managed by Terraform", remove it from state first:
> ```bash
> terraform state rm module.s3_bucket.aws_s3_bucket.main
> terraform import module.s3_bucket.aws_s3_bucket.main crop-disease-models-mlflow-stg-478544568263
> ```

---

## Step 5 — Plan and apply

```bash
terraform plan
terraform apply
```

The apply will:
1. Update the existing S3 bucket (add tags, versioning, encryption)
2. Create the ECR repository and push the Docker image (~5 minutes)
3. Create the Lambda function, API Gateway, and IAM roles

Outputs after apply:
```
api_endpoint    = "https://xxx.execute-api.eu-west-1.amazonaws.com/"
ecr_repo        = "crop-disease-inference_crop-disease-mlops"
ecr_repo_url    = "478544568263.dkr.ecr.eu-west-1.amazonaws.com/..."
lambda_function = "crop-disease-predict_crop-disease-mlops"
model_bucket    = "crop-disease-models-mlflow-stg-478544568263"
```

---

## Step 6 — Test the deployed API

```bash
# health check
curl https://YOUR_API_ID.execute-api.eu-west-1.amazonaws.com/health

# prediction (first request ~15-30s cold start)
curl -X POST https://YOUR_API_ID.execute-api.eu-west-1.amazonaws.com/predict \
    -F "file=@data/processed/test/Tomato___healthy/000004.jpg"
```

Expected responses:
```json
// health
{"status":"healthy","model":"efficientnet_b0","num_classes":38,"run_id":"..."}

// predict
{"class_name":"Tomato___healthy","confidence":0.97,"top_k":[...],"run_id":"..."}
```

---

## Updating the model after retraining

After a new training run, update the RUN_ID in `terraform.tfvars` and apply:

```bash
# update terraform.tfvars
run_id = "new_run_id_from_mlflow"

terraform apply   # only updates Lambda env vars, no rebuild
```

Or update directly via AWS CLI (faster):
```bash
aws lambda update-function-configuration \
    --function-name crop-disease-predict_crop-disease-mlops \
    --environment "Variables={RUN_ID=new_run_id,...}" \
    --region eu-west-1
```

---

## Updating the Docker image

After code changes to `api/`:

```bash
# rebuild and push to ECR
aws ecr get-login-password --region eu-west-1 | \
  docker login --username AWS --password-stdin \
  478544568263.dkr.ecr.eu-west-1.amazonaws.com

docker build -t 478544568263.dkr.ecr.eu-west-1.amazonaws.com/crop-disease-inference_crop-disease-mlops:latest \
  -f api/Dockerfile .

docker push \
  478544568263.dkr.ecr.eu-west-1.amazonaws.com/crop-disease-inference_crop-disease-mlops:latest

# update Lambda to use new image
aws lambda update-function-code \
    --function-name crop-disease-predict_crop-disease-mlops \
    --image-uri 478544568263.dkr.ecr.eu-west-1.amazonaws.com/crop-disease-inference_crop-disease-mlops:latest \
    --region eu-west-1
```

---

## Debugging

**Check Lambda logs:**
```bash
aws logs tail /aws/lambda/crop-disease-predict_crop-disease-mlops \
    --follow --region eu-west-1
```

**Check Lambda environment variables:**
```bash
aws lambda get-function-configuration \
    --function-name crop-disease-predict_crop-disease-mlops \
    --region eu-west-1 \
    --query 'Environment.Variables'
```

**Check current Terraform state:**
```bash
terraform state list
terraform show
```

---

## Known issues and solutions

| Issue | Cause | Fix |
|---|---|---|
| `model.onnx` loads locally but fails on Lambda | ONNX exported with external data format — `.data` file is 0 bytes in S3 | Re-export with `opset_version=13` in Colab for single-file ONNX |
| `AccessDenied` on S3 ListObjects | Wrong bucket name in `MODEL_ARTIFACT_BUCKET` env var | Check bucket name matches exactly in `terraform.tfvars` |
| `RepositoryAlreadyExistsException` | ECR repo exists but not in Terraform state | `terraform import module.ecr.aws_ecr_repository.main <repo-name>` |
| Lambda cold start timeout | Model download from S3 on first invocation | Expected 15-30s — subsequent requests ~1-2s |
| `Reserved keys` error on Lambda env vars | `AWS_DEFAULT_REGION` is reserved by Lambda | Remove it from Lambda env vars in Terraform |

---

## Cost management

Stop billing when not in use:

```bash
# destroy Lambda and API Gateway (keeps S3 and ECR)
terraform destroy \
  -target=module.lambda \
  -target=module.ecr.null_resource.ecr_image_bootstrap

# full destroy (WARNING: deletes S3 bucket and all models if force_destroy=true)
terraform destroy
```

Approximate costs while running:
| Resource | Cost |
|---|---|
| Lambda | ~$0.20 per 1M requests + $0.0000166/GB-s |
| API Gateway | ~$1.00 per 1M requests |
| ECR storage (~600 MB) | ~$0.06/month |
| S3 storage | ~$0.023/GB/month |

Lambda and API Gateway are **pay per use** — no charge when idle.
