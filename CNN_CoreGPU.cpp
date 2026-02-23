#include "CNN_CoreGPU.hpp"

#include <OCLW_Core.hpp>
#include <QtConcurrent>

#include <iostream>

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
  int requestedGPUs = coreConfig.trainingConfig.numGPUs;
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
void CoreGPU<T>::train(const Samples<T>& samples) {
  this->trainingStart(samples.size());

  ulong numSamples = samples.size();
  ulong numEpochs = this->trainingConfig.numEpochs;

  if (this->logLevel >= CNN::LogLevel::INFO) {
    std::cout << "Starting GPU training: " << numSamples << " samples, "
              << numEpochs << " epochs, " << this->numGPUs << " GPU"
              << (this->numGPUs > 1 ? "s" : "") << "\n";
  }

  auto createGpuCallback = [this](size_t gpuIdx) -> TrainingCallback<T> {
    if (!this->trainingCallback) return nullptr;

    return [this, gpuIdx](const TrainingProgress<T>& progress) {
      TrainingProgress<T> gpuProgress = progress;
      gpuProgress.gpuIndex = static_cast<int>(gpuIdx);
      gpuProgress.totalGPUs = static_cast<int>(this->numGPUs);
      this->trainingCallback(gpuProgress);
    };
  };

  for (ulong e = 0; e < numEpochs; e++) {
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

    std::vector<T> gpuLosses(this->numGPUs, 0);

    QtConcurrent::blockingMap(workItems,
        [this, &samples, &gpuLosses, e, numEpochs, &createGpuCallback](const GPUWorkItem& item) {
      gpuLosses[item.gpuIdx] = this->gpuWorkers[item.gpuIdx]->trainSubset(
          samples, item.startIdx, item.endIdx, e + 1, numEpochs, createGpuCallback(item.gpuIdx));
    });

    T totalLoss = 0;
    for (size_t i = 0; i < this->numGPUs; i++) {
      totalLoss += gpuLosses[i];
    }

    // Merge CNN and ANN gradients across workers, then unified update
    this->mergeCNNGradients();
    this->mergeANNGradients();

    for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
      this->gpuWorkers[gpuIdx]->update(numSamples);
    }

    // Sync parameters from GPU so getParameters() returns current values (e.g., for checkpoint saves)
    this->gpuWorkers[0]->syncParametersFromGPU();
    this->parameters = this->gpuWorkers[0]->getParameters();

    T avgEpochLoss = totalLoss / static_cast<T>(numSamples);
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

  std::vector<T> totalAccumFilters;
  std::vector<T> totalAccumBiases;

  for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
    std::vector<T> workerFilters;
    std::vector<T> workerBiases;

    this->gpuWorkers[gpuIdx]->readAccumulatedGradients(workerFilters, workerBiases);

    if (gpuIdx == 0) {
      totalAccumFilters = workerFilters;
      totalAccumBiases = workerBiases;
    } else {
      for (size_t i = 0; i < workerFilters.size(); i++) {
        totalAccumFilters[i] += workerFilters[i];
      }

      for (size_t i = 0; i < workerBiases.size(); i++) {
        totalAccumBiases[i] += workerBiases[i];
      }
    }
  }

  // Write merged gradients back to all workers
  for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
    this->gpuWorkers[gpuIdx]->setAccumulators(totalAccumFilters, totalAccumBiases);
  }
}

//===================================================================================================================//
//-- Multi-GPU gradient merging: ANN --//
//===================================================================================================================//

template <typename T>
void CoreGPU<T>::mergeANNGradients() {
  if (this->numGPUs <= 1) return;

  ANN::Tensor1D<T> totalAccumWeights;
  ANN::Tensor1D<T> totalAccumBiases;

  for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
    ANN::Tensor1D<T> workerWeights;
    ANN::Tensor1D<T> workerBiases;

    this->gpuWorkers[gpuIdx]->readANNAccumulatedGradients(workerWeights, workerBiases);

    if (gpuIdx == 0) {
      totalAccumWeights = workerWeights;
      totalAccumBiases = workerBiases;
    } else {
      for (size_t i = 0; i < workerWeights.size(); i++) {
        totalAccumWeights[i] += workerWeights[i];
      }

      for (size_t i = 0; i < workerBiases.size(); i++) {
        totalAccumBiases[i] += workerBiases[i];
      }
    }
  }

  // Write merged gradients back to all workers
  for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
    this->gpuWorkers[gpuIdx]->setANNAccumulators(totalAccumWeights, totalAccumBiases);
  }
}

//===================================================================================================================//

// Explicit template instantiations
template class CNN::CoreGPU<int>;
template class CNN::CoreGPU<double>;
template class CNN::CoreGPU<float>;
