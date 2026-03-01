#include "CNN_CoreGPU.hpp"

#include <OCLW_Core.hpp>
#include <QtConcurrent>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>

using namespace CNN;

//===================================================================================================================//
//-- Constructor --//
//===================================================================================================================//

template <typename T>
CoreGPU<T>::CoreGPU(const CoreConfig<T>& coreConfig)
    : Core<T>(coreConfig) {

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
Output<T> CoreGPU<T>::predict(const Input<T>& input) {
  this->predictStart();

  Output<T> output = this->gpuWorkers[0]->predict(input);

  this->predictEnd();

  return output;
}

//===================================================================================================================//
//-- Training --//
//===================================================================================================================//

template <typename T>
void CoreGPU<T>::train(ulong numSamples, const SampleProvider<T>& sampleProvider) {
  this->trainingStart(numSamples);

  ulong numEpochs = this->trainingConfig.numEpochs;

  // Adjust batch size to be divisible by numGPUs (round down, minimum = numGPUs)
  ulong batchSize = this->trainingConfig.batchSize;
  batchSize = std::max(this->numGPUs, (batchSize / this->numGPUs) * this->numGPUs);

  if (this->logLevel >= CNN::LogLevel::INFO) {
    std::cout << "Starting GPU training: " << numSamples << " samples, "
              << numEpochs << " epochs, " << this->numGPUs << " GPU"
              << (this->numGPUs > 1 ? "s" : "") << "\n";
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
  std::mt19937 rng(std::random_device{}());

  for (ulong e = 0; e < numEpochs; e++) {
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

      QtConcurrent::blockingMap(workItems,
          [this, &batchSamples, &gpuLosses, e, numEpochs, numSamples, &gpuCumulativeSamples](const GPUWorkItem& item) {
        // Build the per-GPU sub-batch
        Samples<T> gpuSamples(batchSamples.begin() + item.startIdx,
                               batchSamples.begin() + item.endIdx);

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
        gpuLosses[item.gpuIdx] = this->gpuWorkers[item.gpuIdx]->trainSubset(
            gpuSamples, numSamples, e + 1, numEpochs, callback);
      });

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
        for (size_t i = 0; i < this->numGPUs; i++) gpuIndices.append(i);
        QtConcurrent::blockingMap(gpuIndices, [this, currentBatchSize](size_t gpuIdx) {
          this->gpuWorkers[gpuIdx]->update(currentBatchSize);
        });
      }

    }

    // Sync parameters from GPU so getParameters() returns current values (e.g., for checkpoint saves)
    this->gpuWorkers[0]->syncParametersFromGPU();
    this->parameters = this->gpuWorkers[0]->getParameters();

    T avgEpochLoss = epochLoss / static_cast<T>(numSamples);
    this->trainingMetadata.finalLoss = avgEpochLoss;

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
      this->trainingCallback(progress);
    }
  }

  this->trainingEnd();
}

//===================================================================================================================//
//-- Testing --//
//===================================================================================================================//

template <typename T>
TestResult<T> CoreGPU<T>::test(const Samples<T>& samples) {
  ulong numSamples = samples.size();

  ulong samplesPerGPU = numSamples / this->numGPUs;
  ulong remainder = numSamples % this->numGPUs;

  struct GPUWorkItem {
    size_t gpuIdx;
    ulong startIdx;
    ulong endIdx;
  };

  QVector<GPUWorkItem> workItems;

  for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
    ulong startIdx = gpuIdx * samplesPerGPU + std::min(gpuIdx, remainder);
    ulong endIdx = startIdx + samplesPerGPU + (gpuIdx < remainder ? 1 : 0);
    workItems.append({gpuIdx, startIdx, endIdx});
  }

  std::vector<std::pair<T, ulong>> gpuResults(this->numGPUs, {0, 0});

  QtConcurrent::blockingMap(workItems,
      [this, &samples, &gpuResults](const GPUWorkItem& item) {
    gpuResults[item.gpuIdx] = this->gpuWorkers[item.gpuIdx]->testSubset(
        samples, item.startIdx, item.endIdx);
  });

  T totalLoss = static_cast<T>(0);
  ulong totalCorrect = 0;

  for (size_t i = 0; i < this->numGPUs; i++) {
    totalLoss += gpuResults[i].first;
    totalCorrect += gpuResults[i].second;
  }

  TestResult<T> result;
  result.numSamples = numSamples;
  result.totalLoss = totalLoss;
  result.numCorrect = totalCorrect;

  result.averageLoss = (numSamples > 0)
    ? result.totalLoss / static_cast<T>(numSamples)
    : static_cast<T>(0);

  result.accuracy = (numSamples > 0)
    ? static_cast<T>(totalCorrect) / static_cast<T>(numSamples) * static_cast<T>(100)
    : static_cast<T>(0);

  return result;
}

//===================================================================================================================//
//-- Worker initialization --//
//===================================================================================================================//

template <typename T>
void CoreGPU<T>::initializeWorkers() {
  for (size_t i = 0; i < this->numGPUs; i++) {
    auto worker = std::make_unique<CoreGPUWorker<T>>(this->coreConfig);
    this->gpuWorkers.push_back(std::move(worker));
  }
}

//===================================================================================================================//
//-- Multi-GPU gradient merging: CNN --//
//===================================================================================================================//

template <typename T>
void CoreGPU<T>::mergeCNNGradients() {
  if (this->numGPUs <= 1) return;

  // Read gradients from all GPUs in parallel
  std::vector<std::vector<T>> allFilters(this->numGPUs);
  std::vector<std::vector<T>> allBiases(this->numGPUs);

  QVector<size_t> gpuIndices;
  for (size_t i = 0; i < this->numGPUs; i++) gpuIndices.append(i);

  QtConcurrent::blockingMap(gpuIndices, [this, &allFilters, &allBiases](size_t gpuIdx) {
    this->gpuWorkers[gpuIdx]->readAccumulatedGradients(allFilters[gpuIdx], allBiases[gpuIdx]);
  });

  // Sum on CPU
  std::vector<T>& totalFilters = allFilters[0];
  std::vector<T>& totalBiases = allBiases[0];
  for (size_t g = 1; g < this->numGPUs; g++) {
    for (size_t i = 0; i < totalFilters.size(); i++) totalFilters[i] += allFilters[g][i];
    for (size_t i = 0; i < totalBiases.size(); i++) totalBiases[i] += allBiases[g][i];
  }

  // Write merged gradients back to all workers in parallel
  QtConcurrent::blockingMap(gpuIndices, [this, &totalFilters, &totalBiases](size_t gpuIdx) {
    this->gpuWorkers[gpuIdx]->setAccumulators(totalFilters, totalBiases);
  });
}

//===================================================================================================================//
//-- Multi-GPU gradient merging: ANN --//
//===================================================================================================================//

template <typename T>
void CoreGPU<T>::mergeANNGradients() {
  if (this->numGPUs <= 1) return;

  // Read gradients from all GPUs in parallel
  std::vector<ANN::Tensor1D<T>> allWeights(this->numGPUs);
  std::vector<ANN::Tensor1D<T>> allBiases(this->numGPUs);

  QVector<size_t> gpuIndices;
  for (size_t i = 0; i < this->numGPUs; i++) gpuIndices.append(i);

  QtConcurrent::blockingMap(gpuIndices, [this, &allWeights, &allBiases](size_t gpuIdx) {
    this->gpuWorkers[gpuIdx]->readANNAccumulatedGradients(allWeights[gpuIdx], allBiases[gpuIdx]);
  });

  // Sum on CPU
  ANN::Tensor1D<T>& totalWeights = allWeights[0];
  ANN::Tensor1D<T>& totalBiases = allBiases[0];
  for (size_t g = 1; g < this->numGPUs; g++) {
    for (size_t i = 0; i < totalWeights.size(); i++) totalWeights[i] += allWeights[g][i];
    for (size_t i = 0; i < totalBiases.size(); i++) totalBiases[i] += allBiases[g][i];
  }

  // Write merged gradients back to all workers in parallel
  QtConcurrent::blockingMap(gpuIndices, [this, &totalWeights, &totalBiases](size_t gpuIdx) {
    this->gpuWorkers[gpuIdx]->setANNAccumulators(totalWeights, totalBiases);
  });
}

//===================================================================================================================//

// Explicit template instantiations
template class CNN::CoreGPU<int>;
template class CNN::CoreGPU<double>;
template class CNN::CoreGPU<float>;
