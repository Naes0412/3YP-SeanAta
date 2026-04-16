# Run in a Colab cell
!pip install trimesh
!pip install git+https://github.com/openai/CLIP.git
!pip install fvcore iopath

# PyTorch3D - use pre-built wheels for Colab's CUDA version
import torch
pyt_version = torch.__version__.split("+")[0].replace(".", "")
cuda_version = torch.version.cuda.replace(".", "")
!pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu{cuda_version}_pyt{pyt_version}/download.html