#!/bin/bash

echo $INTEL_MKL_VERSION
echo $TORCH_CUDA_ARCH_LIST
echo $TORCH_VERSION
echo $TORCH_OUTPUT_NAME
echo $CP_PYTHON_VERSION

function running_in_docker {
  awk -F/ '$2 == "docker"' /proc/self/cgroup | read
}

if ! running_in_docker; then
    echo "ERROR: This script should only be run from within a docker container, try \`make build\` instead."
    exit 1
fi

# go to image root directory
cd /root

# clone pytorch source code
git clone https://github.com/pytorch/pytorch.git
cd pytorch
git fetch --all --tags --prune
git checkout tags/v${TORCH_VERSION}
git submodule update --init --recursive

# build pytorch
BUILD_TEST=OFF \
  CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1" \
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/lib/intel64_lin/ \
  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
  USE_STATIC_NCCL=ON \
  BLAS=MKL \
  USE_MKLDNN=1 \
  USE_CUDNN=1 \
  USE_NCCL=1 \
  python3 setup.py bdist_wheel


ls -l /root/pytorch/dist
ls -l /root

# copy MKL libraries into torch wheel
cd /root/pytorch/dist
unzip ${TORCH_OUTPUT_NAME}-${CP_PYTHON_VERSION}-${CP_PYTHON_VERSION}m-linux_x86_64.whl
cp /opt/intel/lib/intel64_lin/*.so /root/pytorch/dist/torch/lib/
zip -r /root/torch-${TORCH_VERSION}-${CP_PYTHON_VERSION}-${CP_PYTHON_VERSION}m-linux_x86_64.whl torch/ caffe2/ ${TORCH_OUTPUT_NAME}.dist-info/

# copy pytorch wheel into build folder (for persistance outside the docker container)
cp /root/torch-${TORCH_VERSION}-${CP_PYTHON_VERSION}-${CP_PYTHON_VERSION}m-linux_x86_64.whl /WayveCode/torch-${TORCH_VERSION}-${CP_PYTHON_VERSION}-${CP_PYTHON_VERSION}m-linux_x86_64-${INTEL_MKL_VERSION}.whl

# Build torchvision
cd /root
git clone https://github.com/pytorch/vision.git
cd vision
git fetch --all --tags --prune
git checkout tags/v${TORCHVISION_VERSION}
pip3 install /root/pytorch/dist/${TORCH_OUTPUT_NAME}-${CP_PYTHON_VERSION}-${CP_PYTHON_VERSION}m-linux_x86_64.whl
BUILD_TEST=OFF \
  CFLAGS="-D_GLIBCXX_USE_CXX11_ABI=1" \
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/lib/intel64_lin/ \
  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
  USE_STATIC_NCCL=ON \
  BLAS=MKL \
  USE_MKLDNN=1 \
  USE_CUDNN=1 \
  USE_NCCL=1 \
  python3 setup.py bdist_wheel

# copy torchvision wheel into build folder (for persistance outside the docker container)
cp /root/vision/dist/*.whl /WayveCode/torchvision-${TORCHVISION_VERSION}-${CP_PYTHON_VERSION}-${CP_PYTHON_VERSION}m-linux_x86_64.torch${TORCH_VERSION}-${INTEL_MKL_VERSION}.whl

cp /WayveCode/3rdparty/pytorch/${PYTHON_VERSION}/CMakeLists.txt /root/vision
mkdir build
mkdir install
cd build
cmake -DCMAKE_PREFIX_PATH=/root/pytorch -DCMAKE_INSTALL_PREFIX:PATH=../install -DTORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} ..
make -j12
make install
cd /root/vision/install
TORCHVISION_OUTPUT_NAME=torchvision-cpp-${TORCHVISION_VERSION}-${CP_PYTHON_VERSION}-${CP_PYTHON_VERSION}m-linux_x86_64.torch${TORCH_VERSION}-${INTEL_MKL_VERSION}.zip
zip -r $TORCHVISION_OUTPUT_NAME include/ lib/

# copy torchvision wheel into build folder (for persistance outside the docker container)
cp /root/vision/install/$TORCHVISION_OUTPUT_NAME /WayveCode/$TORCHVISION_OUTPUT_NAME
