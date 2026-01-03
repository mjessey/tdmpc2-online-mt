#!/bin/bash
set -e

echo "====================================================="
echo " Installing TD-MPC2 Python 3.10 Environment"
echo " Using local tmp directory for pip builds + cache"
echo "====================================================="

ENV_NAME="tdmpc2-py310"

# ---------------------------------------------------------
# 0. Setup local tmp directory for pip
# ---------------------------------------------------------
if [ ! -d "./tmp" ]; then
  echo ">>> Creating local tmp directory at ./tmp"
  mkdir -p ./tmp
fi

# Force pip + Python build to use ./tmp
export TMPDIR=$(pwd)/tmp
export PIP_CACHE_DIR=$(pwd)/tmp/pip-cache
export PIP_BUILD=$(pwd)/tmp/pip-build
export PIP_NO_BUILD_ISOLATION=no
mkdir -p "$PIP_CACHE_DIR" "$PIP_BUILD"

echo ">>> TMPDIR set to: $TMPDIR"
echo ">>> PIP_CACHE_DIR set to: $PIP_CACHE_DIR"
echo ">>> PIP_BUILD set to: $PIP_BUILD"

# ---------------------------------------------------------
# 1. Create Conda Environment
# ---------------------------------------------------------
echo ""
echo ">>> Creating conda environment: $ENV_NAME"
conda create -y -n $ENV_NAME python=3.10 pip=24.0
source ~/.bashrc
conda activate $ENV_NAME

# ---------------------------------------------------------
# 2. Install critical system libraries from conda-forge
# ---------------------------------------------------------
echo ""
echo ">>> Installing GLEW, GLib, GLFW, FFmpeg"
conda install -y -c conda-forge glew=2.2.0 glib=2.78.4 glfw ffmpeg

# ---------------------------------------------------------
# 3. Install PyTorch + CUDA 11.8
# ---------------------------------------------------------
echo ""
echo ">>> Installing PyTorch 2.3.1 (CUDA 11.8)"
pip install torch==2.3.1 torchvision==0.18.1 \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  --cache-dir "$PIP_CACHE_DIR"

# ---------------------------------------------------------
# 4. Install TorchRL + Tensordict matching PyTorch version
# ---------------------------------------------------------
echo ""
echo ">>> Installing TorchRL + Tensordict"
pip install torchrl==0.6.0 tensordict==0.6.0 \
  --cache-dir "$PIP_CACHE_DIR"

# ---------------------------------------------------------
# 5. Install MuJoCo + dm-control
# ---------------------------------------------------------
echo ""
echo ">>> Installing MuJoCo + dm-control"
pip install mujoco==3.1.2 dm-control==1.0.16 \
  --cache-dir "$PIP_CACHE_DIR"

# ---------------------------------------------------------
# 6. Install RL ecosystem + utilities
# ---------------------------------------------------------
echo ""
echo ">>> Installing remaining RL packages"
pip install \
  gymnasium==0.29.1 \
  imageio==2.34.1 \
  imageio-ffmpeg==0.4.9 \
  moviepy==1.0.3 \
  h5py==3.11.0 \
  numpy==1.26.4 \
  kornia==0.7.2 \
  termcolor==2.4.0 \
  tqdm==4.66.4 \
  pandas==2.0.3 \
  wandb==0.17.4 \
  --cache-dir "$PIP_CACHE_DIR"

echo ""
echo ">>> Installing Hydra + Submitit"
pip install \
  hydra-core==1.3.2 \
  hydra-submitit-launcher==1.2.0 \
  submitit==1.5.1 \
  omegaconf==2.3.0 \
  --cache-dir "$PIP_CACHE_DIR"

# ---------------------------------------------------------
# 7. Configure MuJoCo for headless EGL rendering
# ---------------------------------------------------------
echo ""
echo ">>> Configuring MuJoCo for headless rendering"
export MUJOCO_GL=egl
echo "export MUJOCO_GL=egl" >>~/.bashrc

# ---------------------------------------------------------
# 8. Verify installation
# ---------------------------------------------------------
echo ""
echo ">>> Running verification"
python - <<EOF
import torch
import mujoco
import dm_control
import tensordict
import torchrl
print("✓ PyTorch:", torch.__version__)
print("✓ MuJoCo:", mujoco.__version__)
print("✓ dm-control imported successfully")
print("✓ TorchRL:", torchrl.__version__)
print("✓ TensorDict:", tensordict.__version__)
print("Environment OK!")
EOF

echo ""
echo "====================================================="
echo " TD-MPC2 environment installation complete!"
echo " Activate with: conda activate $ENV_NAME"
echo "====================================================="
