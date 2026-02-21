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
  OpenCLWrapper::Core::initialize(this->verbose);

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

  if (this->verbose) {
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

    // Merge CNN gradients across workers and update CNN weights
    this->mergeCNNGradients();

    for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
      this->gpuWorkers[gpuIdx]->updateCNN(numSamples);
    }

    // Update ANN on each worker with its local gradients, then average parameters
    for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
      ulong localN = workItems[static_cast<int>(gpuIdx)].endIdx
                   - workItems[static_cast<int>(gpuIdx)].startIdx;
      this->gpuWorkers[gpuIdx]->updateANN(localN);
    }

    if (this->numGPUs > 1) {
      this->mergeANNParameters();
    }

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

  this->gpuWorkers[0]->syncParametersFromGPU();
  this->parameters = this->gpuWorkers[0]->getParameters();
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

  std::vector<T> gpuLosses(this->numGPUs, 0);

  QtConcurrent::blockingMap(workItems,
      [this, &samples, &gpuLosses](const GPUWorkItem& item) {
    gpuLosses[item.gpuIdx] = this->gpuWorkers[item.gpuIdx]->testSubset(
        samples, item.startIdx, item.endIdx);
  });

  TestResult<T> result;
  result.numSamples = numSamples;
  result.totalLoss = static_cast<T>(0);

  for (size_t i = 0; i < this->numGPUs; i++) {
    result.totalLoss += gpuLosses[i];
  }

  result.averageLoss = (numSamples > 0)
    ? result.totalLoss / static_cast<T>(numSamples)
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
//-- Multi-GPU parameter merging: ANN --//
//===================================================================================================================//

template <typename T>
void CoreGPU<T>::mergeANNParameters() {
  // Read ANN parameters from each worker and average element-wise
  ANN::Parameters<T> avgParams = this->gpuWorkers[0]->getANNParameters();
  T scale = static_cast<T>(1) / static_cast<T>(this->numGPUs);

  // Sum parameters from all workers (starting from worker 1, adding to worker 0's)
  for (size_t gpuIdx = 1; gpuIdx < this->numGPUs; gpuIdx++) {
    ANN::Parameters<T> workerParams = this->gpuWorkers[gpuIdx]->getANNParameters();

    // Sum weights: [layer][neuron][prevNeuron]
    for (size_t l = 0; l < workerParams.weights.size(); l++) {
      for (size_t j = 0; j < workerParams.weights[l].size(); j++) {
        for (size_t k = 0; k < workerParams.weights[l][j].size(); k++) {
          avgParams.weights[l][j][k] += workerParams.weights[l][j][k];
        }
      }
    }

    // Sum biases: [layer][neuron]
    for (size_t l = 0; l < workerParams.biases.size(); l++) {
      for (size_t j = 0; j < workerParams.biases[l].size(); j++) {
        avgParams.biases[l][j] += workerParams.biases[l][j];
      }
    }
  }

  // Scale to average
  for (size_t l = 0; l < avgParams.weights.size(); l++) {
    for (size_t j = 0; j < avgParams.weights[l].size(); j++) {
      for (size_t k = 0; k < avgParams.weights[l][j].size(); k++) {
        avgParams.weights[l][j][k] *= scale;
      }
    }
  }

  for (size_t l = 0; l < avgParams.biases.size(); l++) {
    for (size_t j = 0; j < avgParams.biases[l].size(); j++) {
      avgParams.biases[l][j] *= scale;
    }
  }

  // Broadcast averaged parameters to all workers
  for (size_t gpuIdx = 0; gpuIdx < this->numGPUs; gpuIdx++) {
    this->gpuWorkers[gpuIdx]->setANNParameters(avgParams);
  }
}

//===================================================================================================================//

// Explicit template instantiations
template class CNN::CoreGPU<int>;
template class CNN::CoreGPU<double>;
template class CNN::CoreGPU<float>;
