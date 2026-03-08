#include "CNN_CoreCPU.hpp"

#include <ANN_Core.hpp>

#include <QDebug>
#include <QThreadPool>
#include <QtConcurrent>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

using namespace CNN;

//===================================================================================================================//

template <typename T>
CoreCPU<T>::CoreCPU(const CoreConfig<T>& config) : Core<T>(config)
{
  // Initialize conv parameters if not loaded
  Worker<T>::initializeConvParams(this->layersConfig, this->inputShape, this->parameters);

  // Initialize normalization parameters if not loaded
  Worker<T>::initializeNormParams(this->layersConfig, this->inputShape, this->parameters);

  // Create the step worker (used for predict and single-threaded paths)
  bool allocateTraining = (this->modeType == ModeType::TRAIN);
  this->stepWorker = std::make_unique<CoreCPUWorker<T>>(config, this->layersConfig, this->parameters, allocateTraining);

  // Initialize global CNN gradient accumulators if training
  if (allocateTraining) {
    this->accumDConvFilters.resize(this->parameters.convParams.size());
    this->accumDConvBiases.resize(this->parameters.convParams.size());

    for (ulong i = 0; i < this->parameters.convParams.size(); i++) {
      this->accumDConvFilters[i].resize(this->parameters.convParams[i].filters.size(), static_cast<T>(0));
      this->accumDConvBiases[i].resize(this->parameters.convParams[i].biases.size(), static_cast<T>(0));
    }

    this->accumDNormGamma.resize(this->parameters.normParams.size());
    this->accumDNormBeta.resize(this->parameters.normParams.size());
    this->accumNormMean.resize(this->parameters.normParams.size());
    this->accumNormVar.resize(this->parameters.normParams.size());

    for (ulong i = 0; i < this->parameters.normParams.size(); i++) {
      this->accumDNormGamma[i].resize(this->parameters.normParams[i].numChannels, static_cast<T>(0));
      this->accumDNormBeta[i].resize(this->parameters.normParams[i].numChannels, static_cast<T>(0));
      this->accumNormMean[i].resize(this->parameters.normParams[i].numChannels, static_cast<T>(0));
      this->accumNormVar[i].resize(this->parameters.normParams[i].numChannels, static_cast<T>(0));
    }

    if (this->trainingConfig.optimizer.type == OptimizerType::ADAM) {
      this->allocateAdamState();
    }
  }
}

//===================================================================================================================//

template <typename T>
Output<T> CoreCPU<T>::predict(const Input<T>& input)
{
  return this->stepWorker->predict(input);
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::resetGlobalCNNAccumulators()
{
  for (ulong i = 0; i < this->accumDConvFilters.size(); i++) {
    std::fill(this->accumDConvFilters[i].begin(), this->accumDConvFilters[i].end(), static_cast<T>(0));
    std::fill(this->accumDConvBiases[i].begin(), this->accumDConvBiases[i].end(), static_cast<T>(0));
  }

  for (ulong i = 0; i < this->accumDNormGamma.size(); i++) {
    std::fill(this->accumDNormGamma[i].begin(), this->accumDNormGamma[i].end(), static_cast<T>(0));
    std::fill(this->accumDNormBeta[i].begin(), this->accumDNormBeta[i].end(), static_cast<T>(0));
    std::fill(this->accumNormMean[i].begin(), this->accumNormMean[i].end(), static_cast<T>(0));
    std::fill(this->accumNormVar[i].begin(), this->accumNormVar[i].end(), static_cast<T>(0));
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::mergeWorkerCNNAccumulators(const CoreCPUWorker<T>& worker)
{
  const auto& wFilters = worker.getAccumConvFilters();
  const auto& wBiases = worker.getAccumConvBiases();

  for (ulong i = 0; i < wFilters.size(); i++) {
    for (ulong j = 0; j < wFilters[i].size(); j++)
      this->accumDConvFilters[i][j] += wFilters[i][j];

    for (ulong j = 0; j < wBiases[i].size(); j++)
      this->accumDConvBiases[i][j] += wBiases[i][j];
  }

  const auto& wNormGamma = worker.getAccumNormGamma();
  const auto& wNormBeta = worker.getAccumNormBeta();

  for (ulong i = 0; i < wNormGamma.size(); i++) {
    for (ulong j = 0; j < wNormGamma[i].size(); j++)
      this->accumDNormGamma[i][j] += wNormGamma[i][j];

    for (ulong j = 0; j < wNormBeta[i].size(); j++)
      this->accumDNormBeta[i][j] += wNormBeta[i][j];
  }

  const auto& wNormMean = worker.getAccumNormMean();
  const auto& wNormVar = worker.getAccumNormVar();

  for (ulong i = 0; i < wNormMean.size(); i++) {
    for (ulong j = 0; j < wNormMean[i].size(); j++)
      this->accumNormMean[i][j] += wNormMean[i][j];

    for (ulong j = 0; j < wNormVar[i].size(); j++)
      this->accumNormVar[i][j] += wNormVar[i][j];
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::updateNormRunningStats(ulong numSamples)
{
  T n = static_cast<T>(numSamples);

  ulong normIdx = 0;

  for (const auto& layerConfig : this->layersConfig.cnnLayers) {
    if (layerConfig.type == LayerType::BATCHNORM || layerConfig.type == LayerType::INSTANCENORM) {
      const auto& normConfig = std::get<NormLayerConfig>(layerConfig.config);
      T momentum = static_cast<T>(normConfig.momentum);

      for (ulong j = 0; j < this->parameters.normParams[normIdx].numChannels; j++) {
        T avgMean = this->accumNormMean[normIdx][j] / n;
        T avgVar = this->accumNormVar[normIdx][j] / n;
        this->parameters.normParams[normIdx].runningMean[j] =
          (static_cast<T>(1) - momentum) * this->parameters.normParams[normIdx].runningMean[j] + momentum * avgMean;
        this->parameters.normParams[normIdx].runningVar[j] =
          (static_cast<T>(1) - momentum) * this->parameters.normParams[normIdx].runningVar[j] + momentum * avgVar;
      }

      normIdx++;
    }
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::allocateAdamState()
{
  ulong numConvLayers = this->parameters.convParams.size();

  this->adam_m_filters.resize(numConvLayers);
  this->adam_v_filters.resize(numConvLayers);
  this->adam_m_biases.resize(numConvLayers);
  this->adam_v_biases.resize(numConvLayers);

  for (ulong i = 0; i < numConvLayers; i++) {
    this->adam_m_filters[i].resize(this->parameters.convParams[i].filters.size(), static_cast<T>(0));
    this->adam_v_filters[i].resize(this->parameters.convParams[i].filters.size(), static_cast<T>(0));
    this->adam_m_biases[i].resize(this->parameters.convParams[i].biases.size(), static_cast<T>(0));
    this->adam_v_biases[i].resize(this->parameters.convParams[i].biases.size(), static_cast<T>(0));
  }

  ulong numNormLayers = this->parameters.normParams.size();

  this->adam_m_norm_gamma.resize(numNormLayers);
  this->adam_v_norm_gamma.resize(numNormLayers);
  this->adam_m_norm_beta.resize(numNormLayers);
  this->adam_v_norm_beta.resize(numNormLayers);

  for (ulong i = 0; i < numNormLayers; i++) {
    this->adam_m_norm_gamma[i].resize(this->parameters.normParams[i].numChannels, static_cast<T>(0));
    this->adam_v_norm_gamma[i].resize(this->parameters.normParams[i].numChannels, static_cast<T>(0));
    this->adam_m_norm_beta[i].resize(this->parameters.normParams[i].numChannels, static_cast<T>(0));
    this->adam_v_norm_beta[i].resize(this->parameters.normParams[i].numChannels, static_cast<T>(0));
  }

  this->adam_t = 0;
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::updateCNNParameters(ulong numSamples)
{
  T lr = static_cast<T>(this->trainingConfig.learningRate);
  T n = static_cast<T>(numSamples);

  if (this->trainingConfig.optimizer.type == OptimizerType::ADAM) {
    const auto& opt = this->trainingConfig.optimizer;
    T beta1 = opt.beta1;
    T beta2 = opt.beta2;
    T epsilon = opt.epsilon;

    this->adam_t++;

    T bc1 = static_cast<T>(1) - std::pow(beta1, static_cast<T>(this->adam_t));
    T bc2 = static_cast<T>(1) - std::pow(beta2, static_cast<T>(this->adam_t));

    for (ulong i = 0; i < this->parameters.convParams.size(); i++) {
      for (ulong j = 0; j < this->parameters.convParams[i].filters.size(); j++) {
        T g = this->accumDConvFilters[i][j] / n;
        this->adam_m_filters[i][j] = beta1 * this->adam_m_filters[i][j] + (static_cast<T>(1) - beta1) * g;
        this->adam_v_filters[i][j] = beta2 * this->adam_v_filters[i][j] + (static_cast<T>(1) - beta2) * g * g;
        T m_hat = this->adam_m_filters[i][j] / bc1;
        T v_hat = this->adam_v_filters[i][j] / bc2;
        this->parameters.convParams[i].filters[j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
      }

      for (ulong j = 0; j < this->parameters.convParams[i].biases.size(); j++) {
        T g = this->accumDConvBiases[i][j] / n;
        this->adam_m_biases[i][j] = beta1 * this->adam_m_biases[i][j] + (static_cast<T>(1) - beta1) * g;
        this->adam_v_biases[i][j] = beta2 * this->adam_v_biases[i][j] + (static_cast<T>(1) - beta2) * g * g;
        T m_hat = this->adam_m_biases[i][j] / bc1;
        T v_hat = this->adam_v_biases[i][j] / bc2;
        this->parameters.convParams[i].biases[j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
      }
    }

    for (ulong i = 0; i < this->parameters.normParams.size(); i++) {
      for (ulong j = 0; j < this->parameters.normParams[i].numChannels; j++) {
        T g = this->accumDNormGamma[i][j] / n;
        this->adam_m_norm_gamma[i][j] = beta1 * this->adam_m_norm_gamma[i][j] + (static_cast<T>(1) - beta1) * g;
        this->adam_v_norm_gamma[i][j] = beta2 * this->adam_v_norm_gamma[i][j] + (static_cast<T>(1) - beta2) * g * g;
        T m_hat = this->adam_m_norm_gamma[i][j] / bc1;
        T v_hat = this->adam_v_norm_gamma[i][j] / bc2;
        this->parameters.normParams[i].gamma[j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
      }

      for (ulong j = 0; j < this->parameters.normParams[i].numChannels; j++) {
        T g = this->accumDNormBeta[i][j] / n;
        this->adam_m_norm_beta[i][j] = beta1 * this->adam_m_norm_beta[i][j] + (static_cast<T>(1) - beta1) * g;
        this->adam_v_norm_beta[i][j] = beta2 * this->adam_v_norm_beta[i][j] + (static_cast<T>(1) - beta2) * g * g;
        T m_hat = this->adam_m_norm_beta[i][j] / bc1;
        T v_hat = this->adam_v_norm_beta[i][j] / bc2;
        this->parameters.normParams[i].beta[j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
      }
    }
  } else {
    // SGD
    for (ulong i = 0; i < this->parameters.convParams.size(); i++) {
      for (ulong j = 0; j < this->parameters.convParams[i].filters.size(); j++) {
        this->parameters.convParams[i].filters[j] -= lr * (this->accumDConvFilters[i][j] / n);
      }

      for (ulong j = 0; j < this->parameters.convParams[i].biases.size(); j++) {
        this->parameters.convParams[i].biases[j] -= lr * (this->accumDConvBiases[i][j] / n);
      }
    }

    for (ulong i = 0; i < this->parameters.normParams.size(); i++) {
      for (ulong j = 0; j < this->parameters.normParams[i].numChannels; j++) {
        this->parameters.normParams[i].gamma[j] -= lr * (this->accumDNormGamma[i][j] / n);
      }

      for (ulong j = 0; j < this->parameters.normParams[i].numChannels; j++) {
        this->parameters.normParams[i].beta[j] -= lr * (this->accumDNormBeta[i][j] / n);
      }
    }
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::train(ulong numSamples, const SampleProvider<T>& sampleProvider)
{
  ulong numEpochs = this->trainingConfig.numEpochs;

  if (numSamples == 0)
    throw std::runtime_error("No training samples provided");

  ulong batchSize = this->trainingConfig.batchSize;
  this->trainingStart(numSamples);

  if (this->logLevel >= CNN::LogLevel::INFO)
    qDebug() << "CNN Training:" << numEpochs << "epochs," << numSamples << "samples, batch size" << batchSize;

  // Sample index indirection for shuffling
  std::vector<ulong> sampleIndices(numSamples);
  std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
  std::mt19937 rng(this->seed > 0 ? static_cast<unsigned>(this->seed) : std::random_device{}());

  // Emit initial 0% progress callback
  if (this->trainingCallback) {
    TrainingProgress<T> progress{1, numEpochs, 0, numSamples, 0, 0, -1, 0};
    this->trainingCallback(progress);
  }

  for (ulong e = 0; e < numEpochs; e++) {
    T epochLoss = static_cast<T>(0);
    ulong completedSamples = 0;

    if (this->trainingConfig.shuffleSamples) {
      std::shuffle(sampleIndices.begin(), sampleIndices.end(), rng);
    }

    ulong batchIndex = 0;

    for (ulong batchStart = 0; batchStart < numSamples; batchStart += batchSize, batchIndex++) {
      ulong batchEnd = std::min(batchStart + batchSize, numSamples);
      ulong currentBatchSize = batchEnd - batchStart;

      Samples<T> batchSamples = sampleProvider(sampleIndices, batchSize, batchIndex);

      // Batch processing: single worker processes entire batch layer-by-layer.
      // This handles all layer types uniformly — BatchNorm uses batch-wide stats,
      // while Conv/ReLU/Pool/InstanceNorm process each sample independently.
      this->stepWorker->resetAccumulators();
      this->stepWorker->resetAccumLoss();

      std::vector<std::pair<Input<T>, Output<T>>> batchPairs(currentBatchSize);

      for (ulong s = 0; s < currentBatchSize; s++)
        batchPairs[s] = {batchSamples[s].input, batchSamples[s].output};

      T batchLoss = this->stepWorker->processBatch(batchPairs);

      completedSamples += currentBatchSize;

      if (this->trainingCallback) {
        TrainingProgress<T> progress;
        progress.currentEpoch = e + 1;
        progress.totalEpochs = numEpochs;
        progress.currentSample = completedSamples;
        progress.totalSamples = numSamples;
        progress.sampleLoss = batchLoss / static_cast<T>(currentBatchSize);
        progress.epochLoss = 0;
        this->trainingCallback(progress);
      }

      // Update ANN
      this->stepWorker->getANNCore()->update(currentBatchSize);

      // Merge CNN accumulators and update parameters
      this->resetGlobalCNNAccumulators();
      this->mergeWorkerCNNAccumulators(*this->stepWorker);
      epochLoss += this->stepWorker->getAccumLoss();
      this->updateCNNParameters(currentBatchSize);
      this->updateNormRunningStats(currentBatchSize);
    }

    // Sync ANN parameters for checkpoint saves
    this->parameters.denseParams = this->stepWorker->getANNCore()->getParameters();

    T avgLoss = epochLoss / static_cast<T>(numSamples);
    this->trainingMetadata.finalLoss = avgLoss;

    if (this->logLevel >= CNN::LogLevel::INFO)
      qDebug() << "Epoch " << (e + 1) << "/" << numEpochs << " - Loss: " << avgLoss;

    if (this->trainingCallback) {
      TrainingProgress<T> progress;
      progress.currentEpoch = e + 1;
      progress.totalEpochs = numEpochs;
      progress.currentSample = numSamples;
      progress.totalSamples = numSamples;
      progress.sampleLoss = 0;
      progress.epochLoss = avgLoss;
      this->trainingCallback(progress);
    }
  }

  this->trainingEnd();
}

//===================================================================================================================//

template <typename T>
TestResult<T> CoreCPU<T>::test(ulong numSamples, const SampleProvider<T>& sampleProvider)
{
  // When seed > 0, force single-threaded for deterministic results.
  int numThreads = this->numThreads;

  if (this->seed > 0)
    numThreads = 1;
  else if (numThreads <= 0)
    numThreads = QThreadPool::globalInstance()->maxThreadCount();

  // Create per-thread workers (forward pass only — no training buffers)
  std::vector<std::unique_ptr<CoreCPUWorker<T>>> workers;

  for (int i = 0; i < numThreads; i++)
    workers.push_back(
      std::make_unique<CoreCPUWorker<T>>(this->coreConfig, this->layersConfig, this->parameters, false));

  if (this->logLevel >= CNN::LogLevel::INFO)
    qDebug() << "CNN Test:" << numSamples << "samples," << numThreads << "threads";

  // Sequential index array (no shuffling for test)
  std::vector<ulong> sampleIndices(numSamples);

  for (ulong i = 0; i < numSamples; i++)
    sampleIndices[i] = i;

  ulong batchSize = this->testConfig.batchSize;
  ulong numBatches = (numSamples + batchSize - 1) / batchSize;

  T totalLoss = static_cast<T>(0);
  ulong totalCorrect = 0;
  std::atomic<ulong> completedSamples{0};
  QMutex callbackMutex;

  // Emit initial 0% progress callback
  if (this->testCallback) {
    TestProgress<T> progress;
    progress.currentSample = 0;
    progress.totalSamples = numSamples;
    this->testCallback(progress);
  }

  for (ulong b = 0; b < numBatches; b++) {
    Samples<T> batch = sampleProvider(sampleIndices, batchSize, b);
    ulong currentBatchSize = batch.size();

    // Per-worker loss and correct counters
    std::vector<T> workerLosses(numThreads, static_cast<T>(0));
    std::vector<ulong> workerCorrects(numThreads, 0);

    // Per-worker sample ranges
    std::vector<ulong> workerSampleCounts(numThreads);

    for (int i = 0; i < numThreads; i++)
      workerSampleCounts[i] = currentBatchSize / static_cast<ulong>(numThreads) +
                              (static_cast<ulong>(i) < currentBatchSize % static_cast<ulong>(numThreads) ? 1 : 0);

    QVector<int> workerIndices(numThreads);

    for (int i = 0; i < numThreads; i++)
      workerIndices[i] = i;

    QtConcurrent::blockingMap(workerIndices, [&](int workerIdx) {
      CoreCPUWorker<T>& worker = *workers[workerIdx];

      ulong workerLocalStart = 0;

      for (int i = 0; i < workerIdx; i++)
        workerLocalStart += workerSampleCounts[i];
      ulong workerLocalEnd = workerLocalStart + workerSampleCounts[workerIdx];

      for (ulong s = workerLocalStart; s < workerLocalEnd; s++) {
        Output<T> predicted = worker.predict(batch[s].input);
        workerLosses[workerIdx] += worker.calculateLoss(predicted, batch[s].output);

        auto predIdx = std::distance(predicted.begin(), std::max_element(predicted.begin(), predicted.end()));
        auto expIdx =
          std::distance(batch[s].output.begin(), std::max_element(batch[s].output.begin(), batch[s].output.end()));

        if (predIdx == expIdx)
          workerCorrects[workerIdx]++;

        ulong completed = ++completedSamples;

        if (this->testCallback) {
          QMutexLocker locker(&callbackMutex);
          TestProgress<T> progress;
          progress.currentSample = completed;
          progress.totalSamples = numSamples;
          this->testCallback(progress);
        }
      }
    });

    for (int i = 0; i < numThreads; i++) {
      totalLoss += workerLosses[i];
      totalCorrect += workerCorrects[i];
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

// Explicit template instantiations
template class CNN::CoreCPU<int>;
template class CNN::CoreCPU<double>;
template class CNN::CoreCPU<float>;