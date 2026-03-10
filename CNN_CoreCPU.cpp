#include "CNN_CoreCPU.hpp"

#include <ANN_Core.hpp>

#include <QDebug>
#include <QMutex>
#include <QThreadPool>
#include <QtConcurrent>

#include <algorithm>
#include <atomic>
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

  // Initialize batch norm parameters if not loaded
  Worker<T>::initializeBatchNormParams(this->layersConfig, this->inputShape, this->parameters);

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

    this->accumDBNGamma.resize(this->parameters.bnParams.size());
    this->accumDBNBeta.resize(this->parameters.bnParams.size());
    this->accumBNMean.resize(this->parameters.bnParams.size());
    this->accumBNVar.resize(this->parameters.bnParams.size());

    for (ulong i = 0; i < this->parameters.bnParams.size(); i++) {
      this->accumDBNGamma[i].resize(this->parameters.bnParams[i].numChannels, static_cast<T>(0));
      this->accumDBNBeta[i].resize(this->parameters.bnParams[i].numChannels, static_cast<T>(0));
      this->accumBNMean[i].resize(this->parameters.bnParams[i].numChannels, static_cast<T>(0));
      this->accumBNVar[i].resize(this->parameters.bnParams[i].numChannels, static_cast<T>(0));
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

  for (ulong i = 0; i < this->accumDBNGamma.size(); i++) {
    std::fill(this->accumDBNGamma[i].begin(), this->accumDBNGamma[i].end(), static_cast<T>(0));
    std::fill(this->accumDBNBeta[i].begin(), this->accumDBNBeta[i].end(), static_cast<T>(0));
    std::fill(this->accumBNMean[i].begin(), this->accumBNMean[i].end(), static_cast<T>(0));
    std::fill(this->accumBNVar[i].begin(), this->accumBNVar[i].end(), static_cast<T>(0));
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

  const auto& wBNGamma = worker.getAccumBNGamma();
  const auto& wBNBeta = worker.getAccumBNBeta();

  for (ulong i = 0; i < wBNGamma.size(); i++) {
    for (ulong j = 0; j < wBNGamma[i].size(); j++)
      this->accumDBNGamma[i][j] += wBNGamma[i][j];

    for (ulong j = 0; j < wBNBeta[i].size(); j++)
      this->accumDBNBeta[i][j] += wBNBeta[i][j];
  }

  const auto& wBNMean = worker.getAccumBNMean();
  const auto& wBNVar = worker.getAccumBNVar();

  for (ulong i = 0; i < wBNMean.size(); i++) {
    for (ulong j = 0; j < wBNMean[i].size(); j++)
      this->accumBNMean[i][j] += wBNMean[i][j];

    for (ulong j = 0; j < wBNVar[i].size(); j++)
      this->accumBNVar[i][j] += wBNVar[i][j];
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::updateBNRunningStats(ulong numSamples)
{
  T momentum = static_cast<T>(0.1); // Default momentum

  // Get momentum from the first BN layer config
  for (const auto& layerConfig : this->layersConfig.cnnLayers) {
    if (layerConfig.type == LayerType::BATCHNORM) {
      const auto& bn = std::get<BatchNormLayerConfig>(layerConfig.config);
      momentum = static_cast<T>(bn.momentum);
      break;
    }
  }

  T n = static_cast<T>(numSamples);

  for (ulong i = 0; i < this->parameters.bnParams.size(); i++) {
    for (ulong j = 0; j < this->parameters.bnParams[i].numChannels; j++) {
      T avgMean = this->accumBNMean[i][j] / n;
      T avgVar = this->accumBNVar[i][j] / n;
      this->parameters.bnParams[i].runningMean[j] =
        (static_cast<T>(1) - momentum) * this->parameters.bnParams[i].runningMean[j] + momentum * avgMean;
      this->parameters.bnParams[i].runningVar[j] =
        (static_cast<T>(1) - momentum) * this->parameters.bnParams[i].runningVar[j] + momentum * avgVar;
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

  ulong numBNLayers = this->parameters.bnParams.size();

  this->adam_m_bn_gamma.resize(numBNLayers);
  this->adam_v_bn_gamma.resize(numBNLayers);
  this->adam_m_bn_beta.resize(numBNLayers);
  this->adam_v_bn_beta.resize(numBNLayers);

  for (ulong i = 0; i < numBNLayers; i++) {
    this->adam_m_bn_gamma[i].resize(this->parameters.bnParams[i].numChannels, static_cast<T>(0));
    this->adam_v_bn_gamma[i].resize(this->parameters.bnParams[i].numChannels, static_cast<T>(0));
    this->adam_m_bn_beta[i].resize(this->parameters.bnParams[i].numChannels, static_cast<T>(0));
    this->adam_v_bn_beta[i].resize(this->parameters.bnParams[i].numChannels, static_cast<T>(0));
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

    for (ulong i = 0; i < this->parameters.bnParams.size(); i++) {
      for (ulong j = 0; j < this->parameters.bnParams[i].numChannels; j++) {
        T g = this->accumDBNGamma[i][j] / n;
        this->adam_m_bn_gamma[i][j] = beta1 * this->adam_m_bn_gamma[i][j] + (static_cast<T>(1) - beta1) * g;
        this->adam_v_bn_gamma[i][j] = beta2 * this->adam_v_bn_gamma[i][j] + (static_cast<T>(1) - beta2) * g * g;
        T m_hat = this->adam_m_bn_gamma[i][j] / bc1;
        T v_hat = this->adam_v_bn_gamma[i][j] / bc2;
        this->parameters.bnParams[i].gamma[j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
      }

      for (ulong j = 0; j < this->parameters.bnParams[i].numChannels; j++) {
        T g = this->accumDBNBeta[i][j] / n;
        this->adam_m_bn_beta[i][j] = beta1 * this->adam_m_bn_beta[i][j] + (static_cast<T>(1) - beta1) * g;
        this->adam_v_bn_beta[i][j] = beta2 * this->adam_v_bn_beta[i][j] + (static_cast<T>(1) - beta2) * g * g;
        T m_hat = this->adam_m_bn_beta[i][j] / bc1;
        T v_hat = this->adam_v_bn_beta[i][j] / bc2;
        this->parameters.bnParams[i].beta[j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
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

    for (ulong i = 0; i < this->parameters.bnParams.size(); i++) {
      for (ulong j = 0; j < this->parameters.bnParams[i].numChannels; j++) {
        this->parameters.bnParams[i].gamma[j] -= lr * (this->accumDBNGamma[i][j] / n);
      }

      for (ulong j = 0; j < this->parameters.bnParams[i].numChannels; j++) {
        this->parameters.bnParams[i].beta[j] -= lr * (this->accumDBNBeta[i][j] / n);
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

  int numThreads = this->numThreads;

  if (numThreads <= 0)
    numThreads = QThreadPool::globalInstance()->maxThreadCount();

  ulong batchSize = this->trainingConfig.batchSize;
  this->trainingStart(numSamples);

  if (this->logLevel >= CNN::LogLevel::INFO)
    qDebug() << "CNN Training:" << numEpochs << "epochs," << numSamples << "samples," << numThreads
             << "threads, batch size" << batchSize;

  // Create per-thread CoreCPUWorkers (each owns its own ANN core)
  std::vector<std::unique_ptr<CoreCPUWorker<T>>> workers;

  for (int i = 0; i < numThreads; i++) {
    workers.push_back(std::make_unique<CoreCPUWorker<T>>(this->coreConfig, this->layersConfig, this->parameters, true));
  }

  QMutex callbackMutex;

  // Sample index indirection for shuffling
  std::vector<ulong> sampleIndices(numSamples);
  std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
  std::mt19937 rng(std::random_device{}());

  for (ulong e = 0; e < numEpochs; e++) {
    T epochLoss = static_cast<T>(0);
    std::atomic<ulong> completedSamples{0};

    if (this->trainingConfig.shuffleSamples) {
      std::shuffle(sampleIndices.begin(), sampleIndices.end(), rng);
    }

    ulong batchIndex = 0;

    for (ulong batchStart = 0; batchStart < numSamples; batchStart += batchSize, batchIndex++) {
      ulong batchEnd = std::min(batchStart + batchSize, numSamples);
      ulong currentBatchSize = batchEnd - batchStart;

      Samples<T> batchSamples = sampleProvider(sampleIndices, batchSize, batchIndex);

      // Per-worker sample counts (extras distributed to first workers)
      std::vector<ulong> workerSampleCounts(numThreads);

      for (int i = 0; i < numThreads; i++)
        workerSampleCounts[i] = currentBatchSize / static_cast<ulong>(numThreads) +
                                (static_cast<ulong>(i) < currentBatchSize % static_cast<ulong>(numThreads) ? 1 : 0);

      // Sync worker ANN cores with main parameters and reset all accumulators
      ANN::Parameters<T> mainANNParams = this->stepWorker->getANNCore()->getParameters();

      for (int i = 0; i < numThreads; i++) {
        workers[i]->getANNCore()->setParameters(mainANNParams);
        workers[i]->resetAccumulators();
        workers[i]->resetAccumLoss();
      }

      // Each worker processes its chunk of the batch end-to-end (fully parallel)
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
          const Sample<T>& sample = batchSamples[s];
          T sampleLoss = worker.processSample(sample.input, sample.output);

          ulong completed = ++completedSamples;

          if (this->trainingCallback) {
            QMutexLocker locker(&callbackMutex);
            TrainingProgress<T> progress;
            progress.currentEpoch = e + 1;
            progress.totalEpochs = numEpochs;
            progress.currentSample = completed;
            progress.totalSamples = numSamples;
            progress.sampleLoss = sampleLoss;
            progress.epochLoss = 0;
            this->trainingCallback(progress);
          }
        }
      });

      // Merge: update each worker's ANN, then weighted-average their parameters
      for (int i = 0; i < numThreads; i++)

        if (workerSampleCounts[i] > 0)
          workers[i]->getANNCore()->update(workerSampleCounts[i]);

      ANN::Parameters<T> mergedParams;
      const ANN::Parameters<T>& ref = workers[0]->getANNCore()->getParameters();
      mergedParams.weights.resize(ref.weights.size());

      for (ulong l = 0; l < ref.weights.size(); l++) {
        mergedParams.weights[l].resize(ref.weights[l].size());

        for (ulong j = 0; j < ref.weights[l].size(); j++)
          mergedParams.weights[l][j].assign(ref.weights[l][j].size(), static_cast<T>(0));
      }

      mergedParams.biases.resize(ref.biases.size());

      for (ulong l = 0; l < ref.biases.size(); l++)
        mergedParams.biases[l].assign(ref.biases[l].size(), static_cast<T>(0));

      for (int i = 0; i < numThreads; i++) {
        if (workerSampleCounts[i] == 0)
          continue;
        T w = static_cast<T>(workerSampleCounts[i]) / static_cast<T>(currentBatchSize);
        const ANN::Parameters<T>& wp = workers[i]->getANNCore()->getParameters();

        for (ulong l = 0; l < wp.weights.size(); l++)

          for (ulong j = 0; j < wp.weights[l].size(); j++)

            for (ulong k = 0; k < wp.weights[l][j].size(); k++)
              mergedParams.weights[l][j][k] += wp.weights[l][j][k] * w;

        for (ulong l = 0; l < wp.biases.size(); l++)

          for (ulong j = 0; j < wp.biases[l].size(); j++)
            mergedParams.biases[l][j] += wp.biases[l][j] * w;
      }

      this->stepWorker->getANNCore()->setParameters(mergedParams);

      // Merge worker CNN accumulators and update CNN parameters
      this->resetGlobalCNNAccumulators();

      for (int i = 0; i < numThreads; i++) {
        this->mergeWorkerCNNAccumulators(*workers[i]);
        epochLoss += workers[i]->getAccumLoss();
      }

      this->updateCNNParameters(currentBatchSize);
      this->updateBNRunningStats(currentBatchSize);
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
  int numThreads = this->numThreads;

  if (numThreads <= 0)
    numThreads = QThreadPool::globalInstance()->maxThreadCount();

  if (this->logLevel >= CNN::LogLevel::INFO)
    qDebug() << "CNN Test:" << numSamples << "samples," << numThreads << "threads";

  // Create per-thread workers (predict-only, no training buffers)
  std::vector<std::unique_ptr<CoreCPUWorker<T>>> workers;

  for (int i = 0; i < numThreads; i++) {
    workers.push_back(
      std::make_unique<CoreCPUWorker<T>>(this->coreConfig, this->layersConfig, this->parameters, false));
  }

  // Sequential index array (no shuffling for test)
  std::vector<ulong> sampleIndices(numSamples);
  std::iota(sampleIndices.begin(), sampleIndices.end(), 0);

  ulong batchSize = this->testConfig.batchSize;
  ulong numBatches = (numSamples + batchSize - 1) / batchSize;

  T totalLoss = static_cast<T>(0);
  ulong totalCorrect = 0;

  for (ulong b = 0; b < numBatches; b++) {
    Samples<T> batch = sampleProvider(sampleIndices, batchSize, b);
    ulong currentBatchSize = batch.size();

    // Distribute samples across workers
    std::vector<ulong> workerSampleCounts(numThreads);

    for (int i = 0; i < numThreads; i++)
      workerSampleCounts[i] = currentBatchSize / static_cast<ulong>(numThreads) +
                              (static_cast<ulong>(i) < currentBatchSize % static_cast<ulong>(numThreads) ? 1 : 0);

    // Per-worker partial results
    std::vector<T> workerLoss(numThreads, static_cast<T>(0));
    std::vector<ulong> workerCorrect(numThreads, 0);

    // Each worker predicts its chunk in parallel
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
        const Sample<T>& sample = batch[s];
        Output<T> predicted = worker.predict(sample.input);
        workerLoss[workerIdx] += worker.calculateLoss(predicted, sample.output);

        auto predIdx = std::distance(predicted.begin(), std::max_element(predicted.begin(), predicted.end()));
        auto expIdx =
          std::distance(sample.output.begin(), std::max_element(sample.output.begin(), sample.output.end()));

        if (predIdx == expIdx)
          workerCorrect[workerIdx]++;
      }
    });

    // Aggregate worker results
    for (int i = 0; i < numThreads; i++) {
      totalLoss += workerLoss[i];
      totalCorrect += workerCorrect[i];
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