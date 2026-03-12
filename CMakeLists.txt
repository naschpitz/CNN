cmake_minimum_required(VERSION 3.14)

project(CNN LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Core Concurrent)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core Concurrent)

# Include OpenCLWrapper submodule as a subdirectory if not already included
if(NOT TARGET OpenCLWrapper)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/OpenCLWrapper ${CMAKE_CURRENT_BINARY_DIR}/OpenCLWrapper)
endif()

# Include ANN submodule as a subdirectory if not already included
if(NOT TARGET ANN)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/extern/ANN ${CMAKE_CURRENT_BINARY_DIR}/ANN)
endif()

add_library(CNN STATIC
  CNN_Core.hpp CNN_Core.cpp
  CNN_Worker.hpp CNN_Worker.cpp
  CNN_CoreCPU.hpp CNN_CoreCPU.cpp
  CNN_CoreCPUWorker.hpp CNN_CoreCPUWorker.cpp
  CNN_CoreGPU.hpp CNN_CoreGPU.cpp
  CNN_CoreGPUWorker.hpp CNN_CoreGPUWorker.cpp
  CNN_GPUBufferManager.hpp CNN_GPUBufferManager.cpp
  CNN_GPUKernelBuilder.hpp CNN_GPUKernelBuilder.cpp
  opencl/CNN_Defines.hpp.cl
  opencl/CNN_Propagate.cpp.cl
  opencl/CNN_Backpropagate.cpp.cl
  opencl/CNN_Update.cpp.cl
  opencl/CNN_Bridge.cpp.cl
  opencl/CNN_Normalization.cpp.cl
  opencl/CNN_GlobalAvgPool.cpp.cl
  CNN_Types.hpp
  CNN_Device.hpp CNN_Device.cpp
  CNN_Mode.hpp CNN_Mode.cpp
  CNN_SlidingStrategy.hpp CNN_SlidingStrategy.cpp
  CNN_PoolType.hpp CNN_PoolType.cpp
  CNN_LayersConfig.hpp CNN_LayersConfig.cpp
  CNN_CoreConfig.hpp
  CNN_Optimizer.hpp CNN_Optimizer.cpp
  CNN_TrainingConfig.hpp
  CNN_Parameters.hpp
  CNN_Conv2DParameters.hpp
  CNN_NormParameters.hpp
  CNN_Sample.hpp
  CNN_ProgressCallback.hpp
  CNN_TrainingProgress.hpp
  CNN_TrainingMetadata.hpp
  CNN_PredictMetadata.hpp
  CNN_TestResult.hpp
  CNN_Conv2D.hpp CNN_Conv2D.cpp
  CNN_ReLU.hpp CNN_ReLU.cpp
  CNN_Pool.hpp CNN_Pool.cpp
  CNN_Flatten.hpp CNN_Flatten.cpp
  CNN_GlobalAvgPool.hpp CNN_GlobalAvgPool.cpp
  CNN_Normalization.hpp CNN_Normalization.cpp
)

# PUBLIC: Consumers of CNN need access to CNN headers
target_include_directories(CNN PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# PUBLIC: Consumers of CNN will also need ANN and OpenCLWrapper
# PRIVATE: Qt Concurrent is only used internally
target_link_libraries(CNN
  PUBLIC ANN
  PUBLIC OpenCLWrapper
  PRIVATE Qt${QT_VERSION_MAJOR}::Concurrent
)

target_compile_definitions(CNN PRIVATE CNN_LIBRARY)




# Test executable
add_executable(test_cnn
  tests/test_main.cpp
  tests/test_conv2d.cpp
  tests/test_layers.cpp
  tests/test_layers2.cpp
  tests/test_integration_basic.cpp
  tests/test_integration_basic2.cpp
  tests/test_integration_basic3.cpp
  tests/test_integration_costfunc.cpp
  tests/test_integration_exact.cpp
  tests/test_integration_batchnorm.cpp
  tests/test_gpu_basic.cpp
  tests/test_gpu_basic2.cpp
  tests/test_gpu_basic3.cpp
  tests/test_gpu_basic4.cpp
  tests/test_gpu_multigpu.cpp
  tests/test_gpu_multigpu2.cpp
  tests/test_gpu_exact.cpp
  tests/test_gpu_exact_bn.cpp
  tests/test_gpu_batchnorm.cpp
  tests/test_gpu_batchnorm2.cpp
)
target_link_libraries(test_cnn PRIVATE CNN)
