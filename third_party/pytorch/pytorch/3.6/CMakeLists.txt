cmake_minimum_required(VERSION 2.8)
project(torchvision)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED PATH /root/pytorch)

file(GLOB HEADERS torchvision/csrc/vision.h)
file(GLOB MODELS_HEADERS torchvision/csrc/models/*.h)
file(GLOB MODELS_SOURCES torchvision/csrc/models/*.h torchvision/csrc/models/*.cpp)

file(GLOB OPS_HEADERS torchvision/csrc/*.h)
file(GLOB OPS_SOURCES torchvision/csrc/*.h torchvision/csrc/*.cpp)

file(GLOB CPU_HEADERS torchvision/csrc/cpu/*.h)
file(GLOB CPU_SOURCES torchvision/csrc/cpu/*.h torchvision/csrc/cpu/*.cpp)

file(GLOB CUDA_HEADERS torchvision/csrc/cuda/*.h)
file(GLOB CUDA_SOURCES torchvision/csrc/cuda/*.h torchvision/csrc/cuda/*.cu)

set(CUDA_NVCC_FLAGS "-D__CUDA_NO_HALF_OPERATORS__")
add_definitions(-DWITH_CUDA)
include_directories(/usr/include/python3.6m/)

cuda_add_library (${PROJECT_NAME} SHARED ${MODELS_SOURCES} ${OPS_SOURCES} ${CPU_SOURCES} ${CUDA_SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC "${TORCH_LIBRARIES}")

install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME})
install(FILES ${MODELS_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/models)
install(FILES ${CPU_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/cpu)
install(FILES ${CUDA_HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/cuda)
