#include "CNN_CoreGPUWorker.hpp"
#include "CNN_SlidingStrategy.hpp"
#include "CNN_Conv2D.hpp"
#include "CNN_ReLU.hpp"
#include "CNN_Pool.hpp"
#include "CNN_Flatten.hpp"

#include <ANN_CoreGPUWorker.hpp>
#include <OCLW_Core.hpp>

#include <cmath>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>

using namespace CNN;

//===================================================================================================================//
//-- Constructors --//
//===================================================================================================================//

template <typename T>
CoreGPUWorker<T>::CoreGPUWorker(const CoreConfig<T>& config)
  : coreConfig(config),
    parameters(config.parameters),
    logLevel(config.logLevel)
{
  this->costFunctionConfig = config.costFunctionConfig;

  this->ownedCore = std::make_unique<OpenCLWrapper::Core>(false);
  this->core = this->ownedCore.get();
  this->core->setVerbose(this->logLevel >= CNN::LogLevel::DEBUG);

  // Initialize conv parameters (He initialization if not loaded)
  Worker<T>::initializeConvParams(config.layersConfig, config.inputShape, this->parameters);

  // Initialize batch norm parameters if not loaded
  Worker<T>::initializeInstanceNormParams(config.layersConfig, config.inputShape, this->parameters);

  // Create buffer manager
  this->bufferManager =
    std::make_unique<GPUBufferManager<T>>(this->core, this->coreConfig, this->parameters, this->logLevel);

  // Compute buffer offsets for all layers
  this->bufferManager->computeLayerOffsets();

  // Load OpenCL sources (defines first, then kernels)
  this->bufferManager->loadSources(false);

  // Build ANN GPU worker on the shared core (loads ANN sources + allocates ANN buffers)
  this->bufferManager->buildANNWorker();

  // Allocate CNN GPU buffers and write initial parameters
  this->bufferManager->allocateBuffers();

  // Create kernel builder
  this->kernelBuilder =
    std::make_unique<GPUKernelBuilder<T>>(this->core, this->coreConfig, *this->bufferManager, this->logLevel);
}

//===================================================================================================================//

template <typename T>
CoreGPUWorker<T>::CoreGPUWorker(const CoreConfig<T>& config, OpenCLWrapper::Core& sharedCore)
  : coreConfig(config),
    parameters(config.parameters),
    logLevel(config.logLevel),
    core(&sharedCore)
{
  this->costFunctionConfig = config.costFunctionConfig;

  // Initialize conv parameters (He initialization if not loaded)
  Worker<T>::initializeConvParams(config.layersConfig, config.inputShape, this->parameters);

  // Initialize batch norm parameters if not loaded
  Worker<T>::initializeInstanceNormParams(config.layersConfig, config.inputShape, this->parameters);

  // Create buffer manager
  this->bufferManager =
    std::make_unique<GPUBufferManager<T>>(this->core, this->coreConfig, this->parameters, this->logLevel);

  // Compute buffer offsets for all layers
  this->bufferManager->computeLayerOffsets();

  // Create kernel builder
  this->kernelBuilder =
    std::make_unique<GPUKernelBuilder<T>>(this->core, this->coreConfig, *this->bufferManager, this->logLevel);

  // Caller must call loadSources(), buildANNWorker(), allocateBuffers() manually
}

//===================================================================================================================//
//-- Predict --//
//===================================================================================================================//

template <typename T>
Output<T> CoreGPUWorker<T>::predict(const Input<T>& input)
{
  // Set up predict kernels if needed (CNN propagate → bridge → ANN propagate)
  if (!this->kernelBuilder->predictKernelsSetup) {
    this->kernelBuilder->setupPredictKernels();
  }

  // Write input to cnn_actvs at offset 0
  std::vector<T> inputVec(input.data.begin(), input.data.end());
  this->core->template writeBuffer<T>("cnn_actvs", inputVec, 0);

  // Single run: CNN propagate → copy_cnn_to_ann → ANN propagate
  this->core->run();

  // Read ANN output
  ANN::Output<T> annOutput = this->bufferManager->annGPUWorker->bufferManager->readOutput();

  return Output<T>(annOutput.begin(), annOutput.end());
}

//===================================================================================================================//
//-- Training --//
//===================================================================================================================//

template <typename T>
T CoreGPUWorker<T>::trainSubset(const Samples<T>& batchSamples, ulong totalSamples, ulong epoch, ulong totalEpochs,
                                const TrainingCallback<T>& callback)
{
  ulong numSamplesInSubset = batchSamples.size();

  T subsetLoss = static_cast<T>(0);

  // Reset CNN and ANN accumulators
  this->bufferManager->resetAccumulators();

  // Set up training kernels once (full propagate + backpropagate + accumulate pipeline)
  if (!this->kernelBuilder->trainingKernelsSetup) {
    this->kernelBuilder->setupTrainingKernels();
  }

  // Zero the GPU loss accumulator once per subset
  T zeroVal = static_cast<T>(0);
  this->core->template fillBuffer<T>("accum_loss", zeroVal, 1);

  for (ulong s = 0; s < numSamplesInSubset; s++) {
    const Sample<T>& sample = batchSamples[s];

    // Write CNN input to GPU
    std::vector<T> inputVec(sample.input.data.begin(), sample.input.data.end());
    this->core->template writeBuffer<T>("cnn_actvs", inputVec, 0);

    // Write ANN expected output to GPU (for loss computation in backpropagation)
    std::vector<T> expectedVec(sample.output.begin(), sample.output.end());
    this->core->template writeBuffer<T>("outputs", expectedVec, 0);

    // Generate and upload dropout mask for ANN dense layers (different mask per sample)
    if (this->bufferManager->annGPUWorker->bufferManager->hasDropout)
      this->bufferManager->annGPUWorker->bufferManager->generateAndUploadDropoutMask();

    // Single run: CNN propagate → bridge → ANN propagate → ANN backpropagate → reverse bridge → CNN backpropagate → accumulate → loss
    this->core->run();

    // Report progress (no per-sample loss — accumulated on GPU)
    if (callback) {
      TrainingProgress<T> progress;
      progress.currentEpoch = epoch;
      progress.totalEpochs = totalEpochs;
      progress.currentSample = s + 1;
      progress.totalSamples = totalSamples;
      progress.sampleLoss = static_cast<T>(0);
      progress.epochLoss = static_cast<T>(0);
      callback(progress);
    }
  }

  // Read accumulated loss from GPU once per subset
  std::vector<T> lossVec(1);
  this->core->template readBuffer<T>("accum_loss", lossVec, 0);
  subsetLoss = lossVec[0];

  return subsetLoss;
}

//===================================================================================================================//
//-- Testing --//
//===================================================================================================================//

template <typename T>
std::pair<T, ulong> CoreGPUWorker<T>::testSubset(const Samples<T>& samples, ulong startIdx, ulong endIdx)
{
  T subsetLoss = static_cast<T>(0);
  ulong subsetCorrect = 0;

  for (ulong s = startIdx; s < endIdx; s++) {
    Output<T> predicted = this->predict(samples[s].input);
    subsetLoss += this->calculateLoss(predicted, samples[s].output);

    // Accuracy: compare argmax of predicted vs expected
    auto predIdx = std::distance(predicted.begin(), std::max_element(predicted.begin(), predicted.end()));
    auto expIdx =
      std::distance(samples[s].output.begin(), std::max_element(samples[s].output.begin(), samples[s].output.end()));

    if (predIdx == expIdx)
      subsetCorrect++;
  }

  return {subsetLoss, subsetCorrect};
}

//===================================================================================================================//
//-- Step-by-step: backpropagate a single sample (for external orchestration) --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::backpropagateSample(const Input<T>& input, const Output<T>& expected)
{
  // Set up full training pipeline if needed
  if (!this->kernelBuilder->trainingKernelsSetup) {
    this->kernelBuilder->setupTrainingKernels();
  }

  // Write CNN input to GPU
  std::vector<T> inputVec(input.data.begin(), input.data.end());
  this->core->template writeBuffer<T>("cnn_actvs", inputVec, 0);

  // Write ANN expected output to GPU
  std::vector<T> expectedVec(expected.begin(), expected.end());
  this->core->template writeBuffer<T>("outputs", expectedVec, 0);

  // Generate and upload dropout mask for ANN dense layers (different mask per sample)
  if (this->bufferManager->annGPUWorker->bufferManager->hasDropout)
    this->bufferManager->annGPUWorker->bufferManager->generateAndUploadDropoutMask();

  // Single run: full propagate + backpropagate + accumulate
  this->core->run();
}

//===================================================================================================================//
//-- Step-by-step: accumulate (no-op — accumulation is baked into training pipeline) --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::accumulate()
{
  // No-op: accumulation is part of the training kernel pipeline
}

//===================================================================================================================//
//-- Weight update --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::update(ulong numSamples)
{
  this->kernelBuilder->setupUpdateKernels(numSamples);
  this->core->run();
  this->kernelBuilder->invalidateAllKernelFlags();
}

//===================================================================================================================//
//-- Kernel save/restore --//
//===================================================================================================================//

template <typename T>
std::vector<std::vector<OpenCLWrapper::Kernel>> CoreGPUWorker<T>::saveKernels()
{
  return this->core->saveKernels();
}

template <typename T>
void CoreGPUWorker<T>::restoreKernels(const std::vector<std::vector<OpenCLWrapper::Kernel>>& kernels)
{
  this->core->restoreKernels(kernels);
}

template <typename T>
void CoreGPUWorker<T>::setTrainingKernelsReady(bool ready)
{
  this->kernelBuilder->trainingKernelsSetup = ready;
  this->kernelBuilder->updateKernelsSetup = false;
  this->kernelBuilder->predictKernelsSetup = false;
}

//===================================================================================================================//
//-- Loss calculation --//
//===================================================================================================================//

// Note: calculateLoss is now inherited from Worker<T>.

//===================================================================================================================//
//-- Explicit template instantiations --//
//===================================================================================================================//

template class CNN::CoreGPUWorker<int>;
template class CNN::CoreGPUWorker<double>;
template class CNN::CoreGPUWorker<float>;
