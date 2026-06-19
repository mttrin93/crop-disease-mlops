#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Setup script for the CPU training EC2 instance (Amazon Linux 2023).
#
# Run once after launching a fresh EC2 instance.
# Uses the same instance type as the MLflow EC2 (t3.medium minimum).
#
# Prerequisites:
#   - EC2 instance launched with Amazon Linux 2023
#   - IAM role attached with S3 read/write + SSM read permissions
#     (no aws configure needed — credentials come from IAM instance profile)
#
# Usage:
#   ssh -i your-key.pem ec2-user@YOUR_EC2_DNS
#   bash setup_training_ec2.sh
# ──────────────────────────────────────────────────────────────────────────────

set -e

echo "=== Step 1: System update ==="
sudo yum update

echo "=== Step 2: Install pip and git ==="
sudo dnf install python3-pip -y
sudo dnf install git -y

echo "=== Step 3: Install uv ==="
pip3 install uv

echo "=== Step 4: Clone repository ==="
git clone https://github.com/mttrin93/crop-disease-mlops.git
cd crop-disease-mlops

echo "=== Step 5: Install dependencies ==="
# installs main dependencies + torch + torchvision from training group
uv sync --group training

# onnx is needed by train.py for model export but missing from pyproject.toml
pip3 install onnx

echo "=== Step 6: Verify PyTorch ==="
source .venv/bin/activate
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

echo "=== Step 7: Configure AWS region ==="
# credentials come from EC2 IAM instance profile — no aws configure needed
aws configure set region eu-west-1

echo "=== Setup complete ==="
echo ""
echo "The EC2 instance is ready for training."
echo "The venv activates automatically on login — add to ~/.bashrc:"
echo "  echo 'source /home/ec2-user/crop-disease-mlops/.venv/bin/activate' >> ~/.bashrc"
echo "  source ~/.bashrc"
echo ""
echo "Test with:"
echo "  cd crop-disease-mlops"
echo "  python3 src/model/train.py --help"
