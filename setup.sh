#!/bin/bash

# Name of the new environment
ENV_NAME=vlm_arm
PYTHON_VERSION=3.9

echo "🔧 Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

echo "✅ Activating environment..."
conda activate $ENV_NAME || source $(conda info --base)/etc/profile.d/conda.sh && conda activate $ENV_NAME

echo "🐍 Confirming architecture..."
ARCH=$(python -c "import platform; print(platform.machine())")
if [[ "$ARCH" != "arm64" ]]; then
  echo "❌ Not running on ARM Python. Please ensure you're using Miniforge or an ARM-native conda base."
  exit 1
fi

echo "📦 Installing TensorFlow (macOS ARM)..."
pip install tensorflow-macos==2.12.0

echo "🔥 Installing PyTorch + TorchVision (CPU-only)..."
pip install torch==1.13.1 torchvision==0.14.1

echo "📦 Installing THINGSvision v2.4.1 from GitHub..."
pip install git+https://github.com/ViCCo-Group/THINGSvision.git@v2.4.1#egg=thingsvision

echo "📦 Installing remaining dependencies..."
pip install black einops flake8 isort matplotlib ml_collections numpy pandas protobuf pyflakes pytorch_lightning==1.8.6 scipy seaborn tqdm tueplots

echo "✅ All packages installed."

echo "🔍 Final checks:"
python -c "import tensorflow as tf; print('✅ TensorFlow:', tf.__version__)"
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import thingsvision; print('✅ THINGSvision works')"

echo "🎉 Setup complete. You're ready to roll!"
