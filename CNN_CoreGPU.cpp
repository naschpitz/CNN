#include "CNN_CoreGPU.hpp"
#include "CNN_CoreGPUWorkerConfig.hpp"
#include "CNN_TrainingMonitor.hpp"
#include "CNN_Worker.hpp"

#include <OCLW_Core.hpp>
#include <QtConcurrent>

#include <algorithm>
#include <atomic>
#include <iostream>
#include <numeric>
#include <random>

using namespace CNN;

//===================================================================================================================//
//-- Constructor --//
//===================================================================================================================//

template <typename T>
CoreGPU<T>::CoreGPU(const CoreConfig<T>& coreConfig) : Core<T>(coreConfig)
{
  // Initialize OpenCL before querying device information
  OpenCLWrapper::Core::initialize(this->logLevel >= CNN::LogLevel::DEBUG);

  // Determine number of GPUs to use
  int requestedGPUs = coreConfig.numGPUs;
  size_t availableGPUs = OpenCLWrapper::Core::getDevicesUsage().size();

  if (requestedGPUs == 0) {
    this->numGPUs = availableGPUs;
  } else {
    this->numGPUs = std::min(static_cast<size_t>(requestedGPUs), availableGPUs);
  }

  this->initializeWorkers();
}

//===================================================================================================================//
//-- Predict --//
//===================================================================================================================//

template <typename T>
PredictResults<T> CoreGPU<T>::predict(ulong numSamples, const InputProvider<T>& provider)
{
  // Pull batches from the provider; for each batch, fan out across the
  // worker GPUs. Bounded host memory regardless of total dataset size —
  // only one batch lives in host RAM at a time.
  PredictResults<T> results;
  results.reserve(numSamples);

  ulong batchSize = std::max<ulong>(this->numGPUs, this->testConfig.batchSize);
  ulong numBatches = (numSamples + batchSize - 1) / batchSize;
  std::atomic<ulong> completedInputs{0};

  struct GPUWorkItem {
      size_t gpuIdx;
      ulong startIdx;
      ulong endIdx;
  };

  for (ulong batchIndex = 0; batchIndex < numBatches; batchIndex++) {
    Inputs<T> batch = provider(batchSize, batchIndex);
    ulong batchN = batch.size();

    if (batchN == 0)
      break;

    // Split this batch across GPUs.
    ulong perGPU = batchN / this->numGPUs;
    ulong remainder = batchN % this->numGPUs;

    QVector<GPUWorkItem> workItems;

    for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
      ulong startIdx = gpuIdx * perGPU + std::min(gpuIdx, remainder);
      ulong endIdx = startIdx + perGPU + (gpuIdx < remainder ? 1 : 0);

      if (startIdx < endIdx)
        workItems.append({gpuIdx, startIdx, endIdx});
    }

    std::vector<PredictResults<T>> gpuResults(this->numGPUs);

    QtConcurrent::blockingMap(
      workItems, [this, &batch, &gpuResults, &completedInputs, numSamples](const GPUWorkItem& item) {
        ProgressCallback callback;

        if (this->progressCallback) {
          callback = [this, &completedInputs, numSamples](ulong /*current*/, ulong /*total*/) {
            ulong completed = ++completedInputs;
            this->progressCallback(completed, numSamples);
          };
        }

        gpuResults[item.gpuIdx] =
          this->gpuWorkers[item.gpuIdx]->predictSubset(batch, item.startIdx, item.endIdx, callback);
      });

    // Append per-GPU results in input order.
    for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
      for (auto& r : gpuResults[gpuIdx])
        results.push_back(std::move(r));
    }
  }

  return results;
}

//===================================================================================================================//
//-- Training --//
//===================================================================================================================//

template <typename T>
void CoreGPU<T>::train(ulong numSamples, const SampleProvider<T>& sampleProvider)
{
  this->trainingStart(numSamples);

  ulong numEpochs = this->trainingConfig.numEpochs;

  // Adjust batch size to be divisible by numGPUs (round down, minimum = numGPUs)
  ulong batchSize = this->trainingConfig.batchSize;
  batchSize = std::max(this->numGPUs, (batchSize / this->numGPUs) * this->numGPUs);

  if (this->logLevel >= CNN::LogLevel::INFO) {
    std::cout << "Starting GPU training: " << numSamples << " samples, " << numEpochs << " epochs, " << this->numGPUs
              << " GPU" << (this->numGPUs > 1 ? "s" : "") << "\n";
  }

  struct GPUWorkItem {
      size_t gpuIdx;
      ulong startIdx;
      ulong endIdx;
  };

  // Per-GPU cumulative sample counters for progress tracking across mini-batches
  std::vector<ulong> gpuCumulativeSamples(this->numGPUs, 0);

  // Sample index indirection for shuffling
  std::vector<ulong> sampleIndices(numSamples);
  std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
  // Reproducible when trainingConfig.shuffleSeed != 0; non-deterministic otherwise.
  std::mt19937 rng(this->trainingConfig.shuffleSeed != 0 ? this->trainingConfig.shuffleSeed : std::random_device{}());

  // Saved training kernels per GPU — populated after first trainSubset
  std::vector<std::vector<std::vector<OpenCLWrapper::Kernel>>> savedKernels(this->numGPUs);
  bool kernelsSaved = false;

  // Create training monitor if monitoring is enabled
  const MonitoringConfig& monitoringConfig = this->trainingConfig.monitoringConfig;
  std::unique_ptr<TrainingMonitor<T>> monitor;

  if (monitoringConfig.enabled) {
    monitor = std::make_unique<TrainingMonitor<T>>(monitoringConfig);
  }

  for (ulong e = 0; e < numEpochs && !this->stopRequested.load(); e++) {
    T epochLoss = 0;

    // Shuffle sample order for this epoch
    if (this->trainingConfig.shuffleSamples) {
      std::shuffle(sampleIndices.begin(), sampleIndices.end(), rng);
    }

    // Reset cumulative counters at the start of each epoch
    std::fill(gpuCumulativeSamples.begin(), gpuCumulativeSamples.end(), 0);

    // Process samples in mini-batches
    ulong batchIndex = 0;

    for (ulong batchStart = 0; batchStart < numSamples; batchStart += batchSize, batchIndex++) {
      ulong batchEnd = std::min(batchStart + batchSize, numSamples);
      ulong currentBatchSize = batchEnd - batchStart;

      // Fetch batch samples via provider
      Samples<T> batchSamples = sampleProvider(sampleIndices, batchSize, batchIndex);

      // Distribute the batch across GPUs
      ulong samplesPerGPU = currentBatchSize / this->numGPUs;
      ulong remainder = currentBatchSize % this->numGPUs;

      QVector<GPUWorkItem> workItems;

      for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
        ulong startIdx = gpuIdx * samplesPerGPU + std::min(gpuIdx, remainder);
        ulong endIdx = startIdx + samplesPerGPU + (gpuIdx < remainder ? 1 : 0);
        workItems.append({gpuIdx, startIdx, endIdx});
      }

      std::vector<T> gpuLosses(this->numGPUs, 0);

      QtConcurrent::blockingMap(workItems, [this, &batchSamples, &gpuLosses, e, numEpochs, numSamples,
                                            &gpuCumulativeSamples](const GPUWorkItem& item) {
        // Build the per-GPU sub-batch
        Samples<T> gpuSamples(batchSamples.begin() + item.startIdx, batchSamples.begin() + item.endIdx);

        // Create per-batch callback that translates indices to cumulative per-GPU counts
        TrainingCallback<T> callback;

        if (this->trainingCallback) {
          ulong offset = gpuCumulativeSamples[item.gpuIdx];
          size_t gpuIdx = item.gpuIdx;
          callback = [this, offset, gpuIdx, numSamples](const TrainingProgress<T>& progress) {
            TrainingProgress<T> gpuProgress = progress;
            gpuProgress.currentSample = offset + progress.currentSample;
            gpuProgress.totalSamples = numSamples;
            gpuProgress.gpuIndex = static_cast<int>(gpuIdx);
            gpuProgress.totalGPUs = static_cast<int>(this->numGPUs);
            this->trainingCallback(gpuProgress);
          };
        }

        gpuLosses[item.gpuIdx] =
          this->gpuWorkers[item.gpuIdx]->trainSubset(gpuSamples, numSamples, e + 1, numEpochs, callback);
      });

      // Save training kernels after first batch
      if (!kernelsSaved) {
        for (size_t i = 0; i < this->numGPUs; i++) {
          savedKernels[i] = this->gpuWorkers[i]->saveKernels();
        }

        kernelsSaved = true;
      }

      // Update cumulative counters after batch completes
      for (const auto& item : workItems) {
        gpuCumulativeSamples[item.gpuIdx] += (item.endIdx - item.startIdx);
      }

      for (size_t i = 0; i < this->numGPUs; i++) {
        epochLoss += gpuLosses[i];
      }

      // Merge CNN and ANN gradients across workers, then unified update
      this->mergeCNNGradients();
      this->mergeANNGradients();

      // Update weights after each mini-batch (parallel across GPUs)
      {
        QVector<size_t> gpuIndices;

        for (size_t i = 0; i < this->numGPUs; i++)
          gpuIndices.append(i);
        QtConcurrent::blockingMap(
          gpuIndices, [this, currentBatchSize](size_t gpuIdx) { this->gpuWorkers[gpuIdx]->update(currentBatchSize); });
      }

      // Restore training kernels after update
      if (kernelsSaved) {
        QVector<size_t> gpuIndices;

        for (size_t i = 0; i < this->numGPUs; i++)
          gpuIndices.append(i);
        QtConcurrent::blockingMap(gpuIndices, [this, &savedKernels](size_t gpuIdx) {
          this->gpuWorkers[gpuIdx]->restoreKernels(savedKernels[gpuIdx]);
          this->gpuWorkers[gpuIdx]->setTrainingKernelsReady(true);
        });
      }
    }

    // Sync parameters from GPU so getParameters() returns current values (e.g., for checkpoint saves)
    this->gpuWorkers[0]->bufferManager->syncParametersFromGPU();
    this->parameters = this->gpuWorkers[0]->getParameters();

    T avgEpochLoss = epochLoss / static_cast<T>(numSamples);
    this->trainingMetadata.finalLoss = avgEpochLoss;

    // Check training monitor
    bool shouldStop = false;

    if (monitor) {
      shouldStop = monitor->checkEpoch(e + 1, avgEpochLoss);
    }

    if (this->trainingCallback) {
      TrainingProgress<T> progress;
      progress.currentEpoch = e + 1;
      progress.totalEpochs = numEpochs;
      progress.currentSample = numSamples;
      progress.totalSamples = numSamples;
      progress.sampleLoss = 0;
      progress.epochLoss = avgEpochLoss;
      progress.gpuIndex = -1;
      progress.totalGPUs = static_cast<int>(this->numGPUs);

      if (monitor) {
        progress.isNewBest = monitor->isNewBest();

        if (progress.isNewBest) {
          this->trainingMetadata.lastEpoch = e + 1;
        }

        if (shouldStop) {
          progress.stoppedEarly = true;
        }
      }

      this->trainingCallback(progress);
    }

    if (shouldStop) {
      break;
    }
  }

  // Populate monitoring metadata
  if (monitor) {
    this->trainingMetadata.stopReason = monitor->stopReason();
    this->trainingMetadata.bestEpoch = monitor->bestEpoch();
    this->trainingMetadata.bestLoss = monitor->bestLoss();
  }

  this->trainingEnd();
}

//===================================================================================================================//
//-- Testing --//
//===================================================================================================================//

template <typename T>
TestResult<T> CoreGPU<T>::test(ulong numSamples, const SampleProvider<T>& sampleProvider)
{
  // Sequential index array (no shuffling for test)
  std::vector<ulong> sampleIndices(numSamples);

  for (ulong i = 0; i < numSamples; i++) {
    sampleIndices[i] = i;
  }

  ulong batchSize = this->testConfig.batchSize;
  ulong numBatches = (numSamples + batchSize - 1) / batchSize;

  T totalLoss = static_cast<T>(0);
  ulong totalCorrect = 0;

  for (ulong b = 0; b < numBatches; b++) {
    Samples<T> batch = sampleProvider(sampleIndices, batchSize, b);

    // Distribute batch across GPUs
    ulong batchLen = batch.size();
    ulong samplesPerGPU = batchLen / this->numGPUs;
    ulong remainder = batchLen % this->numGPUs;

    struct GPUWorkItem {
        size_t gpuIdx;
        ulong startIdx;
        ulong endIdx;
    };

    QVector<GPUWorkItem> workItems;

    for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
      ulong startIdx = gpuIdx * samplesPerGPU + std::min(gpuIdx, remainder);
      ulong endIdx = startIdx + samplesPerGPU + (gpuIdx < remainder ? 1 : 0);

      if (startIdx < endIdx)
        workItems.append({gpuIdx, startIdx, endIdx});
    }

    std::vector<std::pair<T, ulong>> gpuResults(this->numGPUs, {0, 0});

    QtConcurrent::blockingMap(workItems, [this, &batch, &gpuResults](const GPUWorkItem& item) {
      gpuResults[item.gpuIdx] = this->gpuWorkers[item.gpuIdx]->testSubset(batch, item.startIdx, item.endIdx);
    });

    for (size_t i = 0; i < this->numGPUs; i++) {
      totalLoss += gpuResults[i].first;
      totalCorrect += gpuResults[i].second;
    }

    if (this->progressCallback) {
      ulong samplesProcessed = std::min((b + 1) * batchSize, numSamples);
      this->progressCallback(samplesProcessed, numSamples);
    }
  }

  TestResult<T> result;
  result.numSamples = numSamples;
  result.totalLoss = totalLoss;
  result.numCorrect = totalCorrect;
  result.averageLoss = (numSamples > 0) ? totalLoss / static_cast<T>(numSamples) : static_cast<T>(0);
  result.accuracy = (numSamples > 0) ? static_cast<T>(totalCorrect) / static_cast<T>(numSamples) * static_cast<T>(100)
                                     : static_cast<T>(0);

  return result;
}

//===================================================================================================================//
//-- Parameter sync --//
//===================================================================================================================//

template <typename T>
void CoreGPU<T>::syncParametersToGPU()
{
  // Propagate core's parameters to each worker's buffer manager, then upload to GPU.
  // Workers hold their own copy of parameters (via CoreGPUWorkerConfig), so setParameters()
  // on the Core doesn't automatically reach them.
  for (auto& worker : this->gpuWorkers) {
    worker->bufferManager->setParameters(this->parameters);
    worker->bufferManager->syncParametersToGPU();
  }
}

//===================================================================================================================//
//-- Worker initialization --//
//===================================================================================================================//

template <typename T>
void CoreGPU<T>::initializeWorkers()
{
  CoreGPUWorkerConfig<T> workerConfig(this->coreConfig);

  // Only allocate batch-sized buffers when BatchNorm layers are present.
  // BatchNorm needs cross-sample statistics, so all samples must be in GPU memory.
  // InstanceNorm-only models use the fast single-sample path (batchSize=1).
  bool hasBatchNorm = false;

  for (const auto& layer : this->coreConfig.layersConfig.cnnLayers) {
    if (layer.type == LayerType::BATCHNORM) {
      hasBatchNorm = true;
      break;
    }
  }

  if (hasBatchNorm) {
    // Each GPU only receives batchSize/numGPUs samples.
    // Integer ceiling division: ⌈batchSize / numGPUs⌉.
    // Ensures the buffer fits the GPU that gets the extra sample when the batch doesn't divide evenly.
    ulong fullBatchSize = this->coreConfig.trainingConfig.batchSize;
    workerConfig.batchSize = (fullBatchSize + this->numGPUs - 1) / this->numGPUs;
  }

  for (size_t i = 0; i < this->numGPUs; i++) {
    auto worker = std::make_unique<CoreGPUWorker<T>>(workerConfig);
    this->gpuWorkers.push_back(std::move(worker));
  }
}

//===================================================================================================================//
//-- Multi-GPU gradient merging: CNN --//
//===================================================================================================================//

template <typename T>
void CoreGPU<T>::mergeCNNGradients()
{
  if (this->numGPUs <= 1)
    return;

  // Read gradients from all GPUs in parallel
  std::vector<std::vector<T>> allFilters(this->numGPUs);
  std::vector<std::vector<T>> allBiases(this->numGPUs);
  std::vector<std::vector<T>> allBNGamma(this->numGPUs);
  std::vector<std::vector<T>> allBNBeta(this->numGPUs);

  QVector<size_t> gpuIndices;

  for (size_t i = 0; i < this->numGPUs; i++)
    gpuIndices.append(i);

  QtConcurrent::blockingMap(gpuIndices, [this, &allFilters, &allBiases, &allBNGamma, &allBNBeta](size_t gpuIdx) {
    this->gpuWorkers[gpuIdx]->bufferManager->readAccumulatedGradients(allFilters[gpuIdx], allBiases[gpuIdx]);
    this->gpuWorkers[gpuIdx]->bufferManager->readBNAccumulatedGradients(allBNGamma[gpuIdx], allBNBeta[gpuIdx]);
  });

  // Sum on CPU
  std::vector<T>& totalFilters = allFilters[0];
  std::vector<T>& totalBiases = allBiases[0];
  std::vector<T>& totalBNGamma = allBNGamma[0];
  std::vector<T>& totalBNBeta = allBNBeta[0];

  for (size_t g = 1; g < this->numGPUs; g++) {
    for (size_t i = 0; i < totalFilters.size(); i++)
      totalFilters[i] += allFilters[g][i];

    for (size_t i = 0; i < totalBiases.size(); i++)
      totalBiases[i] += allBiases[g][i];

    for (size_t i = 0; i < totalBNGamma.size(); i++)
      totalBNGamma[i] += allBNGamma[g][i];

    for (size_t i = 0; i < totalBNBeta.size(); i++)
      totalBNBeta[i] += allBNBeta[g][i];
  }

  // Write merged gradients back to all workers in parallel
  QtConcurrent::blockingMap(gpuIndices,
                            [this, &totalFilters, &totalBiases, &totalBNGamma, &totalBNBeta](size_t gpuIdx) {
                              this->gpuWorkers[gpuIdx]->bufferManager->setAccumulators(totalFilters, totalBiases);
                              this->gpuWorkers[gpuIdx]->bufferManager->setBNAccumulators(totalBNGamma, totalBNBeta);
                            });
}

//===================================================================================================================//
//-- Multi-GPU gradient merging: ANN --//
//===================================================================================================================//

template <typename T>
void CoreGPU<T>::mergeANNGradients()
{
  if (this->numGPUs <= 1)
    return;

  // Read gradients from all GPUs in parallel
  std::vector<ANN::Tensor1D<T>> allWeights(this->numGPUs);
  std::vector<ANN::Tensor1D<T>> allBiases(this->numGPUs);

  QVector<size_t> gpuIndices;

  for (size_t i = 0; i < this->numGPUs; i++)
    gpuIndices.append(i);

  QtConcurrent::blockingMap(gpuIndices, [this, &allWeights, &allBiases](size_t gpuIdx) {
    this->gpuWorkers[gpuIdx]->bufferManager->readANNAccumulatedGradients(allWeights[gpuIdx], allBiases[gpuIdx]);
  });

  // Sum on CPU
  ANN::Tensor1D<T>& totalWeights = allWeights[0];
  ANN::Tensor1D<T>& totalBiases = allBiases[0];

  for (size_t g = 1; g < this->numGPUs; g++) {
    for (size_t i = 0; i < totalWeights.size(); i++)
      totalWeights[i] += allWeights[g][i];

    for (size_t i = 0; i < totalBiases.size(); i++)
      totalBiases[i] += allBiases[g][i];
  }

  // Write merged gradients back to all workers in parallel
  QtConcurrent::blockingMap(gpuIndices, [this, &totalWeights, &totalBiases](size_t gpuIdx) {
    this->gpuWorkers[gpuIdx]->bufferManager->setANNAccumulators(totalWeights, totalBiases);
  });
}

//===================================================================================================================//

// Explicit template instantiations
template class CNN::CoreGPU<int>;
template class CNN::CoreGPU<double>;
template class CNN::CoreGPU<float>;
