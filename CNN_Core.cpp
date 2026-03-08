#include "CNN_Core.hpp"
#include "CNN_CoreCPU.hpp"
#include "CNN_CoreGPU.hpp"

#include <ANN_Utils.hpp>

#include <stdexcept>

using namespace CNN;

//===================================================================================================================//

template <typename T>
std::unique_ptr<Core<T>> Core<T>::makeCore(const CoreConfig<T>& config)
{
  switch (config.deviceType) {
  case DeviceType::CPU:
    return std::make_unique<CoreCPU<T>>(config);
  case DeviceType::GPU:
    return std::make_unique<CoreGPU<T>>(config);
  default:
    throw std::runtime_error("Unknown device type");
  }
}

//===================================================================================================================//

template <typename T>
Core<T>::Core(const CoreConfig<T>& coreConfig) : coreConfig(coreConfig)
{
  this->sanityCheck(coreConfig);

  this->deviceType = coreConfig.deviceType;
  this->modeType = coreConfig.modeType;
  this->numThreads = coreConfig.numThreads;
  this->numGPUs = coreConfig.numGPUs;
  this->inputShape = coreConfig.inputShape;
  this->layersConfig = coreConfig.layersConfig;
  this->trainingConfig = coreConfig.trainingConfig;
  this->testConfig = coreConfig.testConfig;
  this->parameters = coreConfig.parameters;
  this->progressReports = coreConfig.progressReports;
  this->logLevel = coreConfig.logLevel;
  this->seed = coreConfig.seed;
}

//===================================================================================================================//

template <typename T>
void Core<T>::sanityCheck(const CoreConfig<T>& coreConfig)
{
  // Validate input shape
  if (coreConfig.inputShape.c == 0 || coreConfig.inputShape.h == 0 || coreConfig.inputShape.w == 0) {
    throw std::runtime_error("Invalid input shape: all dimensions must be > 0");
  }

  // Validate CNN layers shape compatibility
  coreConfig.layersConfig.validateShapes(coreConfig.inputShape);

  // Validate dense layers exist
  if (coreConfig.layersConfig.denseLayers.empty()) {
    throw std::runtime_error("At least one dense layer is required");
  }

  // Validate training config if in training mode
  if (coreConfig.modeType == ModeType::TRAIN) {
    if (coreConfig.trainingConfig.numEpochs == 0) {
      throw std::runtime_error("Number of epochs must be > 0 for training mode");
    }

    if (coreConfig.trainingConfig.learningRate <= 0) {
      throw std::runtime_error("Learning rate must be > 0 for training mode");
    }
  }
}

//===================================================================================================================//

template <typename T>
void Core<T>::trainingStart(ulong numSamples)
{
  this->trainingStartTime = std::chrono::system_clock::now();
  this->trainingMetadata.startTime = ANN::Utils<T>::formatISO8601();
  this->trainingMetadata.numSamples = numSamples;
}

//===================================================================================================================//

template <typename T>
TrainingMetadata<T> Core<T>::trainingEnd()
{
  auto endTime = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = endTime - this->trainingStartTime;

  this->trainingMetadata.endTime = ANN::Utils<T>::formatISO8601();
  this->trainingMetadata.durationSeconds = elapsed.count();
  this->trainingMetadata.durationFormatted = ANN::Utils<T>::formatDuration(elapsed.count());

  return this->trainingMetadata;
}

//===================================================================================================================//

template <typename T>
void Core<T>::predictStart()
{
  this->predictStartTime = std::chrono::system_clock::now();
}

//===================================================================================================================//

template <typename T>
PredictMetadata<T> Core<T>::predictEnd()
{
  auto endTime = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = endTime - this->predictStartTime;

  this->predictMetadata.startTime = ANN::Utils<T>::formatISO8601();
  this->predictMetadata.endTime = ANN::Utils<T>::formatISO8601();
  this->predictMetadata.durationSeconds = elapsed.count();
  this->predictMetadata.durationFormatted = ANN::Utils<T>::formatDuration(elapsed.count());

  return this->predictMetadata;
}

//===================================================================================================================//

// Explicit template instantiations
template class CNN::Core<int>;
template class CNN::Core<double>;
template class CNN::Core<float>;
