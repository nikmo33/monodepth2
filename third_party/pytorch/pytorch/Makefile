
WAYVE_DIR := $(abspath ../..)
INTEL_MKL_VERSION=intel-mkl-2020.1-102
TORCH_CUDA_ARCH_LIST="3.7 6.0 6.1 7.0 7.5"
TORCH_VERSION=1.4.1
TORCH_OUTPUT_NAME=torch-1.4.0a0+7404463
TORCHVISION_VERSION=0.5.0

build_base_image:
	docker build -t nvidia_cuda_base_image --build-arg INTEL_MKL_VERSION=${INTEL_MKL_VERSION} -f Dockerfile.base .

_build_python_image: build_base_image
	docker build -t python-${PYTHON_VERSION}-nvidia-cuda -f Dockerfile.python${PYTHON_VERSION} .

# Set variables for target
build_base_python3.6: PYTHON_VERSION = 3.6
build_base_python3.7: PYTHON_VERSION = 3.7

build_base_python3.6: _build_python_image
build_base_python3.7: _build_python_image

build_base_python:
	$(MAKE) build_base_python3.6
	$(MAKE) build_base_python3.7

_build: build_base_python
	docker run \
		-it \
		--rm \
		--mount type=bind,source="$(WAYVE_DIR)",target=/WayveCode \
		--entrypoint=/WayveCode/3rdparty/pytorch/build.bash \
		-e INTEL_MKL_VERSION=${INTEL_MKL_VERSION} \
		-e CP_PYTHON_VERSION=${CP_PYTHON_VERSION} \
		-e PYTHON_VERSION=${PYTHON_VERSION} \
		-e TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
		-e TORCH_VERSION=${TORCH_VERSION} \
		-e TORCH_OUTPUT_NAME=${TORCH_OUTPUT_NAME} \
		-e TORCHVISION_VERSION=${TORCHVISION_VERSION} \
		python-${PYTHON_VERSION}-nvidia-cuda

build-torch-3.6: CP_PYTHON_VERSION = cp36
build-torch-3.6: PYTHON_VERSION = 3.6
build-torch-3.6: _build

build-torch-3.7: CP_PYTHON_VERSION = cp37
build-torch-3.7: PYTHON_VERSION = 3.7
build-torch-3.7: _build

build-torch:
	$(MAKE) build-torch-3.6
	$(MAKE) build-torch-3.7