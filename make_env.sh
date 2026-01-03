#!/bin/bash
set -e

echo "====================================================="
echo " Installing TD-MPC2 Python 3.10 Environment"
echo "====================================================="

ENV_NAME="tdmpc2-py310"

# ---------------------------------------------------------
# 1. Create Conda Environment
# ---------------------------------------------------------
echo ""
echo ">>> Creating conda environment: $ENV_NAME"
conda create -y -n $ENV_NAME python=3.10 pip=24.0
source ~/.bashrc
conda activate $ENV_NAME

# ---------------------------------------------------------
# 2. Install GLFW, GLEW, GLib, FFmpeg from conda-forge
# (needed for MuJoCo + dm-control)
# ---------------------------------------------------------
echo ""
echo ">>> Installing system libraries (OpenGL, video, etc.)"
conda install -y -c conda-forge glew=2.2.0 glib=2.78.4 glfw ffmpeg

# ---------------------------------------------------------
# 3. Install PyTorch (stable) + CUDA 11.8
# ---------------------------------------------------------
echo ""
echo ">>> Installing PyTorch 2.3.1 + CUDA 11.8"
pip install torch==2.3.1 torchvision==0.18.1 --extra-index-url https://download.pytorch.org/whl/cu118

# ---------------------------------------------------------
# 4. Install TorchRL + Tensordict (versions compatible with PyTorch 2.3)
# ---------------------------------------------------------
echo ""
echo ">>> Installing TorchRL + Tensordict"
pip install torchrl==0.6.0 tensordict==0.6.0

# ---------------------------------------------------------
# 5. Install MuJoCo + dm-control
# ---------------------------------------------------------
echo ""
echo ">>> Installing MuJoCo + dm-control"
pip install mujoco==3.1.2 dm-control==1.0.16

# ---------------------------------------------------------
# 6. Install core RL / ML packages
# ---------------------------------------------------------
echo ""
echo ">>> Installing RL, video, and utilities"
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
  wandb==0.17.4

# ---------------------------------------------------------
# 7. Install Hydra + Submitit for cluster launches
# ---------------------------------------------------------
echo ""
echo ">>> Installing Hydra + Submitit"
pip install \
  hydra-core==1.3.2 \
  hydra-submitit-launcher==1.2.0 \
  submitit==1.5.1 \
  omegaconf==2.3.0

# ---------------------------------------------------------
# 8. Configure MuJoCo for headless EGL rendering
# ---------------------------------------------------------
echo ""
echo ">>> Setting up MuJoCo headless EGL configuration"

export MUJOCO_GL=egl

mkdir -p ~/.mujoco
echo "MUJOCO_GL=egl" >>~/.bashrc

# ---------------------------------------------------------
# 9. Verification tests
# ---------------------------------------------------------
echo ""
echo ">>> Running environment verification"
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
print("✓ Tensordict:", tensordict.__version__)
print("All checks passed!")
EOF

echo ""
echo "====================================================="
echo " TD-MPC2 environment installation complete!"
echo " Activate with:   conda activate $ENV_NAME"
echo "====================================================="
