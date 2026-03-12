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
  // Use the configured batch size so batch norm training has room for all samples
  ulong batchSize = std::max(static_cast<ulong>(1), static_cast<ulong>(config.trainingConfig.batchSize));
  this->bufferManager->allocateBuffers(batchSize);

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
  ulong N = batchSamples.size();
  ulong sampleStride = this->bufferManager->totalActvSize;

  // Detect whether any BN layers exist
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  ulong numLayers = cnnLayers.size();

  bool hasBatchNorm = false;

  for (const auto& layer : cnnLayers) {
    if (layer.type == LayerType::BATCHNORM) {
      hasBatchNorm = true;
      break;
    }
  }

  // Tell kernel builder whether to skip BN running stats in update()
  this->kernelBuilder->skipBNRunningStatsInUpdate = hasBatchNorm;

  // Reset accumulators
  this->bufferManager->resetAccumulators();

  // Zero the GPU loss accumulator
  T zeroVal = static_cast<T>(0);
  this->core->template fillBuffer<T>("accum_loss", zeroVal, 1);

  if (!hasBatchNorm) {
    // ---- FAST PATH: no BN layers — pre-built kernel set, one run() per sample ----
    if (!this->kernelBuilder->trainingKernelsSetup) {
      this->kernelBuilder->setupTrainingKernels();
    }

    for (ulong s = 0; s < N; s++) {
      const Sample<T>& sample = batchSamples[s];

      std::vector<T> inputVec(sample.input.data.begin(), sample.input.data.end());
      this->core->template writeBuffer<T>("cnn_actvs", inputVec, 0);

      std::vector<T> expectedVec(sample.output.begin(), sample.output.end());
      this->core->template writeBuffer<T>("outputs", expectedVec, 0);

      if (this->bufferManager->annGPUWorker->bufferManager->hasDropout)
        this->bufferManager->annGPUWorker->bufferManager->generateAndUploadDropoutMask();

      this->core->run();

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
  } else {
    // ---- BATCH NORM PATH: segment-based processing with cross-sample statistics ----

    // Write all sample inputs to the unified activation buffer at sample offsets
    for (ulong n = 0; n < N; n++) {
      const auto& input = batchSamples[n].input;
      std::vector<T> inputVec(input.data.begin(), input.data.end());
      this->core->template writeBuffer<T>("cnn_actvs", inputVec, n * sampleStride);
    }

    // Identify BN layer positions to determine segments
    std::vector<ulong> bnLayerIndices;
    std::vector<ulong> bnNormIndices;
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
    ulong segStart = 0;

    for (ulong bnIdx = 0; bnIdx < bnLayerIndices.size(); bnIdx++) {
      ulong bnLayer = bnLayerIndices[bnIdx];

      if (segStart < bnLayer) {
        this->core->clearKernels();

        for (ulong n = 0; n < N; n++) {
          this->kernelBuilder->addPropagateKernels(n, segStart, bnLayer, true);
        }

        this->core->run();
      }

      this->core->clearKernels();
      this->kernelBuilder->addBatchNormForwardKernels(bnLayer, N);
      this->core->run();

      segStart = bnLayer + 1;
    }

    if (segStart < numLayers) {
      this->core->clearKernels();

      for (ulong n = 0; n < N; n++) {
        this->kernelBuilder->addPropagateKernels(n, segStart, numLayers, true);
      }

      this->core->run();
    }

    // ---- ANN FORWARD + BACKWARD (per sample) ----
    for (ulong n = 0; n < N; n++) {
      this->core->clearKernels();
      this->kernelBuilder->addCopyBridgeKernels(n);
      this->bufferManager->annGPUWorker->kernelBuilder->addPropagateKernels();
      this->core->run();

      std::vector<T> expectedVec(batchSamples[n].output.begin(), batchSamples[n].output.end());
      this->core->template writeBuffer<T>("outputs", expectedVec, 0);

      if (this->bufferManager->annGPUWorker->bufferManager->hasDropout)
        this->bufferManager->annGPUWorker->bufferManager->generateAndUploadDropoutMask();

      this->core->clearKernels();
      this->bufferManager->annGPUWorker->kernelBuilder->addBackpropagateKernels(true);
      this->kernelBuilder->addReverseBridgeKernels(n);
      this->bufferManager->annGPUWorker->kernelBuilder->addAccumulateKernels();

      // Loss
      ulong outputActvOffset = this->bufferManager->annGPUWorker->bufferManager->getOutputActvOffset();
      ulong numOutputNeurons = this->bufferManager->annGPUWorker->bufferManager->getNumOutputNeurons();
      std::string lossId = "calculate_sample_loss_s" + std::to_string(n);
      this->core->addKernel(lossId, "calculate_sample_loss", 1, 0);
      this->core->template addArgument<T>(lossId, "actvs");
      this->core->template addArgument<T>(lossId, "outputs");
      this->core->template addArgument<T>(lossId, "lossWeights");
      this->core->template addArgument<T>(lossId, "accum_loss");
      this->core->template addArgument<ulong>(lossId, outputActvOffset);
      this->core->template addArgument<ulong>(lossId, numOutputNeurons);
      this->core->template addArgument<ulong>(lossId, static_cast<ulong>(this->coreConfig.costFunctionConfig.type));

      this->core->run();

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
    ulong segEnd = numLayers;

    for (long bnIdx = static_cast<long>(bnLayerIndices.size()) - 1; bnIdx >= 0; bnIdx--) {
      ulong bnLayer = bnLayerIndices[static_cast<ulong>(bnIdx)];

      if (bnLayer + 1 < segEnd) {
        this->core->clearKernels();

        for (ulong n = 0; n < N; n++) {
          this->kernelBuilder->addBackpropagateKernels(n, bnLayer + 1, segEnd);
          this->kernelBuilder->addCNNAccumulateKernels(n, bnLayer + 1, segEnd);
        }

        this->core->run();
      }

      // Batch-wide BN backward
      this->core->clearKernels();
      this->kernelBuilder->addBatchNormBackwardKernels(bnLayer, N);

      // Accumulate BN dGamma/dBeta
      ulong normIdx = bnNormIndices[static_cast<ulong>(bnIdx)];
      ulong normParamOffset = this->bufferManager->normInfos[normIdx].paramOffset;
      ulong numChannels = this->bufferManager->normInfos[normIdx].numChannels;

      std::string dgId = "bn_accum_dGamma_" + std::to_string(bnIdx);
      this->core->addKernel(dgId, "accumulate_gradients", numChannels, 0);
      this->core->template addArgument<T>(dgId, "cnn_accum_norm_dGamma");
      this->core->template addArgument<T>(dgId, "cnn_norm_dGamma");
      this->core->template addArgument<ulong>(dgId, normParamOffset);
      this->core->template addArgument<ulong>(dgId, numChannels);

      std::string dbId = "bn_accum_dBeta_" + std::to_string(bnIdx);
      this->core->addKernel(dbId, "accumulate_gradients", numChannels, 0);
      this->core->template addArgument<T>(dbId, "cnn_accum_norm_dBeta");
      this->core->template addArgument<T>(dbId, "cnn_norm_dBeta");
      this->core->template addArgument<ulong>(dbId, normParamOffset);
      this->core->template addArgument<ulong>(dbId, numChannels);

      this->core->run();

      segEnd = bnLayer;
    }

    if (segEnd > 0) {
      this->core->clearKernels();

      for (ulong n = 0; n < N; n++) {
        this->kernelBuilder->addBackpropagateKernels(n, 0, segEnd);
        this->kernelBuilder->addCNNAccumulateKernels(n, 0, segEnd);
      }

      this->core->run();
    }

    // Update running stats for BN layers using batch-wide mean/var
    this->core->clearKernels();
    this->kernelBuilder->addBatchNormRunningStatsUpdate(N);
    this->core->run();
  }

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
