#!/bin/bash
set -e

echo "====================================================="
echo " Installing TD-MPC2 + SAI-RL compatible environment"
echo " Python 3.10 | MuJoCo latest | PyTorch 2.3.1 CUDA 11.8"
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

export TMPDIR=$(pwd)/tmp
export PIP_CACHE_DIR=$(pwd)/tmp/pip-cache
export PIP_BUILD=$(pwd)/tmp/pip-build
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
# 2. Install OpenGL / video libs (cluster compatible)
# ---------------------------------------------------------
echo ""
echo ">>> Installing GLEW, GLib, GLFW, FFmpeg"
conda install -y -c conda-forge glew=2.2.0 glib=2.78.4 glfw ffmpeg

# ---------------------------------------------------------
# 3. Install PyTorch 2.3.1 + CUDA 11.8 (NO torchvision)
# ---------------------------------------------------------
echo ""
echo ">>> Installing PyTorch 2.3.1 (CUDA 11.8)"
pip install torch==2.3.1 --extra-index-url https://download.pytorch.org/whl/cu118 \
  --cache-dir "$PIP_CACHE_DIR"

# ---------------------------------------------------------
# 4. Install TorchRL + Tensordict matching PyTorch
# ---------------------------------------------------------
echo ""
echo ">>> Installing TorchRL + Tensordict"
pip install torchrl==0.6.0 tensordict==0.6.0 \
  --cache-dir "$PIP_CACHE_DIR"

# ---------------------------------------------------------
# 5. Install MuJoCo (latest) + dm-control
# ---------------------------------------------------------
echo ""
echo ">>> Installing latest MuJoCo (compatible with sai-rl)"
pip install mujoco==3.1.6 --cache-dir "$PIP_CACHE_DIR"

# dm-control 1.0.16 supports MuJoCo 3.x
echo ">>> Installing dm-control"
pip install dm-control==1.0.16 --cache-dir "$PIP_CACHE_DIR"

# ---------------------------------------------------------
# 6. Install core RL ecosystem + utilities
# ---------------------------------------------------------
echo ""
echo ">>> Installing RL utilities"
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
  requests \
  rich \
  --cache-dir "$PIP_CACHE_DIR"

# ---------------------------------------------------------
# 7. Hydra + Submitit
# ---------------------------------------------------------
echo ""
echo ">>> Installing Hydra + Submitit"
pip install \
  hydra-core==1.3.2 \
  hydra-submitit-launcher==1.2.0 \
  submitit==1.5.1 \
  omegaconf==2.3.0 \
  --cache-dir "$PIP_CACHE_DIR"

# ---------------------------------------------------------
# 8. Install sai-rl + sai-mujoco (NO deps to avoid conflicts)
# ---------------------------------------------------------
echo ""
echo ">>> Installing sai-rl and sai-mujoco (no-deps)"
pip install sai-rl --no-deps --cache-dir "$PIP_CACHE_DIR"
pip install sai_mujoco --no-deps --cache-dir "$PIP_CACHE_DIR"

# ---------------------------------------------------------
# 9. Configure MuJoCo for **OSMesa** (your cluster)
# ---------------------------------------------------------
echo ""
echo ">>> Configuring MuJoCo for OSMesa headless rendering"

export LD_LIBRARY_PATH=/opt/ohpc/pub/spack/v0.21.1/opt/spack/linux-rocky8-x86_64_v3/gcc-8.5.0/mesa-23.0.3-qy7u3gufkr4l7gp3mly5ep7a3552ndcy/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/ohpc/pub/spack/v0.21.1/opt/spack/linux-rocky8-x86_64_v3/gcc-8.5.0/osmesa-11.2.0-2mtp6ylfa43bbq3gpknt73advouseugf/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/ohpc/pub/spack/v0.21.1/opt/spack/linux-rocky8-x86_64_v3/gcc-8.5.0/mesa-glu-9.0.2-nybbwnu5oyz5ses2a24kmgllscyjmdot/lib:$LD_LIBRARY_PATH

export MUJOCO_GL=osmesa
echo "export MUJOCO_GL=osmesa" >>~/.bashrc

# ---------------------------------------------------------
# 10. Verify installation
# ---------------------------------------------------------
echo ""
echo ">>> Running verification"
python - <<EOF
import torch
import mujoco
import dm_control
import tensordict
import torchrl
import numpy
import pandas
print("✓ PyTorch:", torch.__version__)
print("✓ MuJoCo:", mujoco.__version__)
print("✓ dm-control imported successfully")
print("✓ TorchRL:", torchrl.__version__)
print("✓ TensorDict:", tensordict.__version__)
print("✓ NumPy:", numpy.__version__)
print("✓ Pandas:", pandas.__version__)
print("Environment OK!")
EOF

echo ""
echo "====================================================="
echo " TD-MPC2 environment installation complete!"
echo " Activate with: conda activate $ENV_NAME"
echo "====================================================="
