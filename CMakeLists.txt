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
  CNN_CoreCPU.hpp CNN_CoreCPU.cpp
  CNN_Types.hpp
  CNN_Device.hpp CNN_Device.cpp
  CNN_Mode.hpp CNN_Mode.cpp
  CNN_SlidingStrategy.hpp CNN_SlidingStrategy.cpp
  CNN_PoolType.hpp CNN_PoolType.cpp
  CNN_LayersConfig.hpp CNN_LayersConfig.cpp
  CNN_CoreConfig.hpp
  CNN_TrainingConfig.hpp
  CNN_Parameters.hpp
  CNN_Sample.hpp
  CNN_TrainingProgress.hpp
  CNN_TrainingMetadata.hpp
  CNN_PredictMetadata.hpp
  CNN_TestResult.hpp
  CNN_Utils.hpp CNN_Utils.cpp
  CNN_Conv2D.hpp CNN_Conv2D.cpp
  CNN_ReLU.hpp CNN_ReLU.cpp
  CNN_Pool.hpp CNN_Pool.cpp
  CNN_Flatten.hpp CNN_Flatten.cpp
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

# nlohmann JSON library (header-only)
set(LIB_NLOHMANN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs/nlohmann")
target_include_directories(CNN PUBLIC ${LIB_NLOHMANN_DIR})


# Test executable
add_executable(test_cnn tests/test_cnn.cpp)
target_link_libraries(test_cnn PRIVATE CNN)
