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

  // Initialize normalization parameters if not loaded
  Worker<T>::initializeNormParams(config.layersConfig, config.inputShape, this->parameters);

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
  this->bufferManager->allocateBuffers(config.trainingConfig.batchSize);

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

  // Initialize normalization parameters if not loaded
  Worker<T>::initializeNormParams(config.layersConfig, config.inputShape, this->parameters);

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
  ulong batchSize = batchSamples.size();
  ulong sampleStride = this->bufferManager->totalActvSize;
  bool useCache = (batchSize == this->cachedBatchSize);

  // Reset CNN and ANN accumulators
  this->bufferManager->resetAccumulators();

  // Zero the GPU loss accumulator
  T zeroVal = static_cast<T>(0);
  this->core->template fillBuffer<T>("accum_loss", zeroVal, 1);

  // Phase 1: Write all N inputs to GPU at sample-specific offsets
  for (ulong s = 0; s < batchSize; s++) {
    ulong writeOffset = s * sampleStride;
    std::vector<T> inputVec(batchSamples[s].input.data.begin(), batchSamples[s].input.data.end());
    this->core->template writeBuffer<T>("cnn_actvs", inputVec, writeOffset);
  }

  // Phase 2: CNN forward for all N samples (layer-by-layer, batch-aware)
  if (useCache) {
    this->core->restoreKernels(this->cachedForwardKernels);
  } else {
    this->kernelBuilder->setupBatchForwardKernels();
    this->cachedForwardKernels = this->core->saveKernels();
    this->cachedANNKernels.resize(batchSize);
  }

  this->core->run();

  // Phase 3: Per-sample ANN forward + backward + reverse bridge + accumulate + loss
  for (ulong s = 0; s < batchSize; s++) {
    // Write ANN expected output for this sample
    std::vector<T> expectedVec(batchSamples[s].output.begin(), batchSamples[s].output.end());
    this->core->template writeBuffer<T>("outputs", expectedVec, 0);

    // Generate and upload dropout mask for ANN dense layers (different mask per sample)
    if (this->bufferManager->annGPUWorker->bufferManager->hasDropout)
      this->bufferManager->annGPUWorker->bufferManager->generateAndUploadDropoutMask();

    // Set up and run per-sample ANN pipeline
    if (useCache) {
      this->core->restoreKernels(this->cachedANNKernels[s]);
    } else {
      this->kernelBuilder->setupPerSampleANNKernels(s);
      this->cachedANNKernels[s] = this->core->saveKernels();
    }

    this->core->run();

    // Report progress
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

  // Phase 4: CNN backward for all N samples + accumulate
  if (useCache) {
    this->core->restoreKernels(this->cachedBackwardKernels);
  } else {
    this->kernelBuilder->setupBatchBackwardKernels();
    this->cachedBackwardKernels = this->core->saveKernels();
    this->cachedBatchSize = batchSize;
  }

  this->core->run();

  // Read accumulated loss from GPU
  std::vector<T> lossVec(1);
  this->core->template readBuffer<T>("accum_loss", lossVec, 0);

  return lossVec[0];
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
  // Process a single sample as a batch of size 1
  Samples<T> singleSample = {{Sample<T>{input, expected}}};
  this->trainSubset(singleSample, 1, 1, 1, nullptr);
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
void CoreGPUWorker<T>::invalidateKernelCache()
{
  this->cachedBatchSize = 0;
  this->cachedForwardKernels.clear();
  this->cachedANNKernels.clear();
  this->cachedBackwardKernels.clear();
  this->kernelBuilder->invalidateAllKernelFlags();
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
