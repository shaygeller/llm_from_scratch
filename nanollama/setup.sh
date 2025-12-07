#!/bin/bash
# NanoLlama-1B Environment Setup Script

set -e  # Exit on error

echo "========================================="
echo "NanoLlama-1B Environment Setup"
echo "========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create conda environment (optional)
read -p "Create new conda environment 'nanollama'? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating conda environment..."
    conda create -n nanollama python=3.11 -y
    echo "✓ Environment created"
    echo "  Activate with: conda activate nanollama"
    echo ""
fi

echo "Installing PyTorch with CUDA 12.4..."
pip install torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "Installing core dependencies..."
pip install -r requirements.txt

echo "Installing Flash Attention 2..."
pip install flash-attn --no-build-isolation

echo ""
echo "Verifying installation..."
python3 << EOF
import torch
import transformers
import datasets
import flash_attn

print(f"✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA version: {torch.version.cuda}")
    print(f"✓ GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"✓ Transformers: {transformers.__version__}")
print(f"✓ Datasets: {datasets.__version__}")
print(f"✓ Flash Attention: Available")
EOF

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Login to HuggingFace (for Llama-3 tokenizer):"
echo "   huggingface-cli login"
echo ""
echo "2. Login to Weights & Biases (for logging):"
echo "   wandb login"
echo ""
echo "3. Download datasets:"
echo "   python scripts/download_datasets.py --all"
echo ""
