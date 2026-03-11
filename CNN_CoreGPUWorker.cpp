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
//-- Batch-norm-aware training --//
//===================================================================================================================//

template <typename T>
T CoreGPUWorker<T>::trainBatchNormSubset(const Samples<T>& batchSamples, ulong totalSamples, ulong epoch,
                                         ulong totalEpochs, const TrainingCallback<T>& callback)
{
  ulong N = batchSamples.size();
  ulong sampleStride = this->bufferManager->totalActvSize;

  // Allocate batch buffers if not already done
  this->bufferManager->allocateBatchBuffers(N);

  // Tell kernel builder to skip BN running stats in update() (we handle them here)
  this->kernelBuilder->skipBNRunningStatsInUpdate = true;

  // Reset accumulators
  this->bufferManager->resetAccumulators();

  // Zero the GPU loss accumulator
  T zeroVal = static_cast<T>(0);
  this->core->template fillBuffer<T>("accum_loss", zeroVal, 1);

  // Write all sample inputs to batch activation buffer
  for (ulong n = 0; n < N; n++) {
    const auto& input = batchSamples[n].input;
    std::vector<T> inputVec(input.data.begin(), input.data.end());
    this->core->template writeBuffer<T>("cnn_batch_actvs", inputVec, n * sampleStride);
  }

  // Identify BN layer positions to determine segments
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  ulong numLayers = cnnLayers.size();

  std::vector<ulong> bnLayerIndices;
  std::vector<ulong> bnNormIndices; // normIdx for each BN layer (index into normInfos)
  ulong normCounter = 0;

  for (ulong i = 0; i < numLayers; i++) {
    if (cnnLayers[i].type == LayerType::BATCHNORM) {
      bnLayerIndices.push_back(i);
      bnNormIndices.push_back(normCounter);
      normCounter++;
    } else if (cnnLayers[i].type == LayerType::INSTANCENORM) {
      normCounter++;
    }
  }

  // ---- FORWARD PASS ----
  // Process segments between BN layers
  ulong segStart = 0;

  for (ulong bnIdx = 0; bnIdx < bnLayerIndices.size(); bnIdx++) {
    ulong bnLayer = bnLayerIndices[bnIdx];

    // Process layers [segStart, bnLayer) for each sample
    if (segStart < bnLayer) {
      this->core->clearKernels();

      for (ulong n = 0; n < N; n++) {
        this->kernelBuilder->addBatchPropagateKernelsForSample(n, segStart, bnLayer);
      }

      this->core->run();
    }

    // Batch-wide BN forward (cross-sample reduction)
    this->core->clearKernels();
    this->kernelBuilder->addBatchNormForwardKernels(bnLayer, N);
    this->core->run();

    segStart = bnLayer + 1;
  }

  // Process remaining layers after last BN layer
  if (segStart < numLayers) {
    this->core->clearKernels();

    for (ulong n = 0; n < N; n++) {
      this->kernelBuilder->addBatchPropagateKernelsForSample(n, segStart, numLayers);
    }

    this->core->run();
  }

  // ---- ANN FORWARD + BACKWARD (per sample) ----
  for (ulong n = 0; n < N; n++) {
    // Copy CNN output from batch buffer to ANN input
    this->core->clearKernels();
    this->kernelBuilder->addBatchCopyBridgeKernels(n);
    this->bufferManager->annGPUWorker->kernelBuilder->addPropagateKernels();
    this->core->run();

    // Write expected output
    std::vector<T> expectedVec(batchSamples[n].output.begin(), batchSamples[n].output.end());
    this->core->template writeBuffer<T>("outputs", expectedVec, 0);

    // Generate dropout mask
    if (this->bufferManager->annGPUWorker->bufferManager->hasDropout)
      this->bufferManager->annGPUWorker->bufferManager->generateAndUploadDropoutMask();

    // ANN backpropagate
    this->core->clearKernels();
    this->bufferManager->annGPUWorker->kernelBuilder->addBackpropagateKernels(true);

    // Reverse bridge: copy ANN input gradients to batch gradient buffer
    this->kernelBuilder->addBatchReverseBridgeKernels(n);

    // ANN accumulate
    this->bufferManager->annGPUWorker->kernelBuilder->addAccumulateKernels();

    // Loss
    ulong outputActvOffset = this->bufferManager->annGPUWorker->bufferManager->getOutputActvOffset();
    ulong numOutputNeurons = this->bufferManager->annGPUWorker->bufferManager->getNumOutputNeurons();
    this->core->addKernel("calculate_sample_loss_s" + std::to_string(n), "calculate_sample_loss", 1, 0);
    this->core->template addArgument<T>("calculate_sample_loss_s" + std::to_string(n), "actvs");
    this->core->template addArgument<T>("calculate_sample_loss_s" + std::to_string(n), "outputs");
    this->core->template addArgument<T>("calculate_sample_loss_s" + std::to_string(n), "lossWeights");
    this->core->template addArgument<T>("calculate_sample_loss_s" + std::to_string(n), "accum_loss");
    this->core->template addArgument<ulong>("calculate_sample_loss_s" + std::to_string(n), outputActvOffset);
    this->core->template addArgument<ulong>("calculate_sample_loss_s" + std::to_string(n), numOutputNeurons);
    this->core->template addArgument<ulong>("calculate_sample_loss_s" + std::to_string(n),
                                            static_cast<ulong>(this->coreConfig.costFunctionConfig.type));

    this->core->run();

    // Report progress
    if (callback) {
      TrainingProgress<T> progress;
      progress.currentEpoch = epoch;
      progress.totalEpochs = totalEpochs;
      progress.currentSample = n + 1;
      progress.totalSamples = totalSamples;
      progress.sampleLoss = static_cast<T>(0);
      progress.epochLoss = static_cast<T>(0);
      callback(progress);
    }
  }

  // ---- CNN BACKWARD PASS ----
  // Process segments between BN layers in reverse.
  // Only accumulate conv/bias gradients in segments that contain conv layers.
  ulong segEnd = numLayers;

  // Helper: check if a segment [start, end) contains any conv layers
  auto segmentHasConv = [&cnnLayers](ulong start, ulong end) {
    for (ulong i = start; i < end; i++) {
      if (cnnLayers[i].type == LayerType::CONV)
        return true;
    }

    return false;
  };

  for (long bnIdx = static_cast<long>(bnLayerIndices.size()) - 1; bnIdx >= 0; bnIdx--) {
    ulong bnLayer = bnLayerIndices[static_cast<ulong>(bnIdx)];

    // Process layers (bnLayer+1, segEnd] for each sample
    if (bnLayer + 1 < segEnd) {
      bool hasConv = segmentHasConv(bnLayer + 1, segEnd);

      this->core->clearKernels();

      for (ulong n = 0; n < N; n++) {
        this->kernelBuilder->addBatchBackpropagateKernelsForSample(n, bnLayer + 1, segEnd);

        if (hasConv) {
          this->kernelBuilder->addBatchCNNAccumulateKernelsForSample(n, bnLayer + 1, segEnd);
        }
      }

      this->core->run();
    }

    // Batch-wide BN backward (cross-sample reduction)
    this->core->clearKernels();
    this->kernelBuilder->addBatchNormBackwardKernels(bnLayer, N);

    // Accumulate BN dGamma/dBeta for this specific BN layer only
    ulong normIdx = bnNormIndices[static_cast<ulong>(bnIdx)];
    ulong normParamOffset = this->bufferManager->normInfos[normIdx].paramOffset;
    ulong numChannels = this->bufferManager->normInfos[normIdx].numChannels;

    this->core->addKernel("bn_accum_dGamma", "accumulate_gradients", numChannels, 0);
    this->core->template addArgument<T>("bn_accum_dGamma", "cnn_accum_norm_dGamma");
    this->core->template addArgument<T>("bn_accum_dGamma", "cnn_norm_dGamma");
    this->core->template addArgument<ulong>("bn_accum_dGamma", normParamOffset);
    this->core->template addArgument<ulong>("bn_accum_dGamma", numChannels);

    this->core->addKernel("bn_accum_dBeta", "accumulate_gradients", numChannels, 0);
    this->core->template addArgument<T>("bn_accum_dBeta", "cnn_accum_norm_dBeta");
    this->core->template addArgument<T>("bn_accum_dBeta", "cnn_norm_dBeta");
    this->core->template addArgument<ulong>("bn_accum_dBeta", normParamOffset);
    this->core->template addArgument<ulong>("bn_accum_dBeta", numChannels);

    this->core->run();

    segEnd = bnLayer;
  }

  // Process remaining layers before first BN layer
  if (segEnd > 0) {
    this->core->clearKernels();

    for (ulong n = 0; n < N; n++) {
      this->kernelBuilder->addBatchBackpropagateKernelsForSample(n, 0, segEnd);
      this->kernelBuilder->addBatchCNNAccumulateKernelsForSample(n, 0, segEnd);
    }

    this->core->run();
  }

  // Update running stats for BN layers using batch-wide mean/var
  this->core->clearKernels();
  this->kernelBuilder->addBatchNormRunningStatsUpdate(N);
  this->core->run();

  // Read accumulated loss
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
