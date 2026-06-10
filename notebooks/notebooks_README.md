# Training Notebook — Setup Guide

This guide documents the steps to run `train_colab.ipynb` end to end.
The notebook fine-tunes EfficientNet-B0 on PlantVillage (38 classes) and
registers the trained model in MLflow.

---

## Prerequisites

- AWS account with CLI configured (`aws configure`)
- Processed dataset at `data/processed/` (run `src/data/preprocess.py` first)
- Google account for Colab

---

## Step 1 — Upload processed data to S3

Run from the project root:

```bash
aws s3 sync data/processed/train \
    s3://${MODEL_BUCKET}$/data/processed/train --region eu-west-1

aws s3 sync data/processed/val \
    s3://${MODEL_BUCKET}$/data/processed/val --region eu-west-1

aws s3 sync data/processed/test \
    s3://${MODEL_BUCKET}$/data/processed/test --region eu-west-1

aws s3 cp data/processed/metadata.json \
    s3://${MODEL_BUCKET}$/data/processed/metadata.json --region eu-west-1
```

> Only needs to be done once. Re-run only if the processed data changes.

---

## Step 2 — Create AWS infrastructure

Follow the DataTalksClub MLflow on AWS guide:
https://github.com/DataTalksClub/mlops-zoomcamp/blob/main/02-experiment-tracking/mlflow_on_aws.md

### Differences from the guide

**PostgreSQL RDS:**
- Template → **Sandbox** (instead of Free tier)
- Availability and durability → **Single-AZ DB instance deployment**
- Once the instance is created: RDS console → your DB →
  **Connectivity & security** → **Connected compute resources** →
  **Set up EC2 connection** (this automatically configures the security
  group rules between RDS and EC2 — no manual inbound rule editing needed)

**S3 bucket:** this is not the S3 bucket created in Step 1 — create a new bucket for the
MLflow artifact store.

---

## Step 3 — Connect to EC2 and install dependencies

```bash
# fix key permissions (required by SSH)
chmod 400 "mlflow-key-pair-mlops.pem"

# connect
ssh -i "mlflow-key-pair-mlops.pem" ec2-user@your-ec2-dns
```

Once inside the EC2 instance:

```bash
# update system packages
sudo yum update

# install pip (Amazon Linux 2023 ships without it)
sudo dnf install python3-pip -y

# install Python dependencies
pip3 install mlflow boto3 psycopg2-binary

# configure AWS credentials
# (needed so MLflow can write artifacts to S3)
aws configure
```

---

## Step 4 — Start the MLflow tracking server

Still inside the EC2 instance:

```bash
mlflow server \
    -h 0.0.0.0 \
    -p 5000 \
    --backend-store-uri postgresql://MASTER_USERNAME:MASTER_PASSWORD@RDS_ENDPOINT:5432/INITIAL_DB_NAME \
    --default-artifact-root s3://MODEL_BUCKET
```

Replace the placeholders:

| Placeholder | Where to find it |
|---|---|
| `MASTER_USERNAME` | RDS console → your DB → Configuration |
| `MASTER_PASSWORD` | What you set when creating the RDS instance |
| `RDS_ENDPOINT` | RDS console → your DB → Connectivity & security → Endpoint |
| `INITIAL_DB_NAME` | RDS console → your DB → Configuration → DB name |

Verify the server is up by opening in your browser:
```
http://your-ec2-dns:5000
```

> Make sure port 5000 is open in the EC2 security group (inbound TCP 5000
> from your IP or 0.0.0.0/0 for testing).

---

## Step 5 — Train in Google Colab

1. Open `train_colab.ipynb` in Google Colab
2. Set runtime to GPU: **Runtime → Change runtime type → T4 GPU**
3. Edit the configuration cell:
   ```python
   MLFLOW_TRACKING_URI = "http://your-ec2-dns:5000"
   DATA_SOURCE = "s3"
   S3_BUCKET = "MODEL_BUCKET"
   ```
4. Run all cells

At the end of the run the notebook prints:
```
Model registered as: crop-disease-efficientnet-b0
Run ID: <your_run_id>
```

Copy the `RUN_ID` — you will need it for the inference service and
Terraform deployment.

---

## Step 6 — Promote model to Production

In the last notebook cell (or directly in the MLflow UI):

1. Go to `http://your-ec2-dns:5000` → **Models** → `crop-disease-efficientnet-b0`
2. Click the latest version → **Stage** → **Staging**
3. After validating metrics → **Stage** → **Production**

Then store the RUN_ID for the deployment pipeline:
```bash
aws ssm put-parameter \
    --name "/crop-disease-mlops/staging/run_id" \
    --value "<your_run_id>" \
    --type String \
    --overwrite \
    --region eu-west-1
```

---

## Cost management

All three AWS resources incur charges while running. Stop them when not in use:

```bash
# stop the MLflow server (Ctrl+C in the EC2 terminal, then exit)
exit
```

In the AWS console:
- **EC2** → Instances → select instance → **Instance state → Stop**
  (not Terminate — Stop preserves your setup for next time)
- **RDS** → Databases → select DB → **Actions → Stop temporarily**

Restart both before the next training session and re-run Step 4.

---

## Approximate costs per training session

| Resource | Cost |
|---|---|
| EC2 t2.micro (MLflow server) | ~$0.012/hour |
| RDS db.t3.micro | ~$0.018/hour |
| S3 storage (~500 MB) | ~$0.01/month |
| Colab T4 GPU | Free (free tier) |

A typical training session (2-3 hours) costs < $0.20.
