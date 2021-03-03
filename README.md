# Installation

1. Install [Python 3.6](https://www.python.org/downloads/release/python-368/).
1. In terminal change current working directory to the root of this repository.
1. (Optional) Initialize virtual environment and activate it according to the
   [tutorial](https://docs.python.org/3/library/venv.html).
1. Run `python -m pip install -U pip setuptools wheel`. This will update pip, setuptools and wheel packages.
1. Install necessary drivers and software to allow Tensorflow and PyTorch use GPU.
    - [GPU guide for Tensorflow](https://www.tensorflow.org/install/gpu#software_requirements).
    - [GPUs with CUDA](https://developer.nvidia.com/cuda-gpus). *This is not a complete list.*
    - [CUDA executables and documentation](https://developer.nvidia.com/cuda-downloads).
    - Read [docs](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
      to find out which drivers you need to support necessary CUDA version.
    - After installing CUDA and drivers download and install [cuDNN](https://developer.nvidia.com/cudnn).
    - Install [Tensorflow](https://www.tensorflow.org/install/pip)
      and [PyTorch](https://pytorch.org/).
1. Run `pip install -r requirements.txt`. This will install all necessary packages for the project.

# dirt_road_DL

[Dataset](https://drive.google.com/file/d/1yEBQB7d8UI_tdMXAYe-jBWdQDU4o0zgU/view) for training.

[Video](https://drive.google.com/file/d/1HFXTIv1-z0vwpbT61OgiWgK1Slsru_00/view) for testing.
