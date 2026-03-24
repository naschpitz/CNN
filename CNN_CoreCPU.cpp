#include "CNN_CoreCPU.hpp"
#include "CNN_Conv2D.hpp"
#include "CNN_ReLU.hpp"
#include "CNN_Pool.hpp"
#include "CNN_Flatten.hpp"
#include "CNN_GlobalAvgPool.hpp"
#include "CNN_Normalization.hpp"

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
  Worker<T>::initializeNormParams(this->layersConfig, this->inputShape, this->parameters);

  // Check if any layer is BATCHNORM
  for (const auto& layerConfig : this->layersConfig.cnnLayers) {
    if (layerConfig.type == LayerType::BATCHNORM) {
      this->hasBatchNorm = true;
      break;
    }
  }

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

    this->accumDBNGamma.resize(this->parameters.normParams.size());
    this->accumDBNBeta.resize(this->parameters.normParams.size());
    this->accumNormMean.resize(this->parameters.normParams.size());
    this->accumNormVar.resize(this->parameters.normParams.size());

    for (ulong i = 0; i < this->parameters.normParams.size(); i++) {
      this->accumDBNGamma[i].resize(this->parameters.normParams[i].numChannels, static_cast<T>(0));
      this->accumDBNBeta[i].resize(this->parameters.normParams[i].numChannels, static_cast<T>(0));
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
Outputs<T> CoreCPU<T>::predict(const Inputs<T>& inputs)
{
  int numThreads = this->numThreads;

  if (numThreads <= 0)
    numThreads = QThreadPool::globalInstance()->maxThreadCount();

  // Build a config snapshot with current trained parameters so fresh workers get the right weights
  CoreConfig<T> predictConfig = this->coreConfig;
  predictConfig.parameters = this->parameters;

  // Create per-thread workers (predict-only, no training buffers)
  std::vector<std::unique_ptr<CoreCPUWorker<T>>> workers;

  for (int i = 0; i < numThreads; i++) {
    workers.push_back(std::make_unique<CoreCPUWorker<T>>(predictConfig, this->layersConfig, this->parameters, false));
  }

  ulong numInputs = inputs.size();
  Outputs<T> outputs(numInputs);

  // Distribute inputs across workers
  std::vector<ulong> workerInputCounts(numThreads);

  for (int i = 0; i < numThreads; i++)
    workerInputCounts[i] = numInputs / static_cast<ulong>(numThreads) +
                           (static_cast<ulong>(i) < numInputs % static_cast<ulong>(numThreads) ? 1 : 0);

  // Each worker predicts its chunk in parallel
  QVector<int> workerIndices(numThreads);

  for (int i = 0; i < numThreads; i++)
    workerIndices[i] = i;

  QtConcurrent::blockingMap(workerIndices, [&](int workerIdx) {
    CoreCPUWorker<T>& worker = *workers[workerIdx];

    ulong workerLocalStart = 0;

    for (int i = 0; i < workerIdx; i++)
      workerLocalStart += workerInputCounts[i];
    ulong workerLocalEnd = workerLocalStart + workerInputCounts[workerIdx];

    for (ulong s = workerLocalStart; s < workerLocalEnd; s++)
      outputs[s] = worker.predict(inputs[s]);
  });

  return outputs;
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
      this->accumNormMean[i][j] += wBNMean[i][j];

    for (ulong j = 0; j < wBNVar[i].size(); j++)
      this->accumNormVar[i][j] += wBNVar[i][j];
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::updateNormRunningStats(ulong numSamples)
{
  T momentum = static_cast<T>(0.1); // Default momentum

  // Get momentum from the first norm layer config
  for (const auto& layerConfig : this->layersConfig.cnnLayers) {
    if (layerConfig.type == LayerType::INSTANCENORM || layerConfig.type == LayerType::BATCHNORM) {
      const auto& bn = std::get<NormLayerConfig>(layerConfig.config);
      momentum = static_cast<T>(bn.momentum);
      break;
    }
  }

  T n = static_cast<T>(numSamples);

  for (ulong i = 0; i < this->parameters.normParams.size(); i++) {
    for (ulong j = 0; j < this->parameters.normParams[i].numChannels; j++) {
      T avgMean = this->accumNormMean[i][j] / n;
      T avgVar = this->accumNormVar[i][j] / n;
      this->parameters.normParams[i].runningMean[j] =
        (static_cast<T>(1) - momentum) * this->parameters.normParams[i].runningMean[j] + momentum * avgMean;
      this->parameters.normParams[i].runningVar[j] =
        (static_cast<T>(1) - momentum) * this->parameters.normParams[i].runningVar[j] + momentum * avgVar;
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

  ulong numBNLayers = this->parameters.normParams.size();

  this->adam_m_norm_gamma.resize(numBNLayers);
  this->adam_v_norm_gamma.resize(numBNLayers);
  this->adam_m_norm_beta.resize(numBNLayers);
  this->adam_v_norm_beta.resize(numBNLayers);

  for (ulong i = 0; i < numBNLayers; i++) {
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
        T g = this->accumDBNGamma[i][j] / n;
        this->adam_m_norm_gamma[i][j] = beta1 * this->adam_m_norm_gamma[i][j] + (static_cast<T>(1) - beta1) * g;
        this->adam_v_norm_gamma[i][j] = beta2 * this->adam_v_norm_gamma[i][j] + (static_cast<T>(1) - beta2) * g * g;
        T m_hat = this->adam_m_norm_gamma[i][j] / bc1;
        T v_hat = this->adam_v_norm_gamma[i][j] / bc2;
        this->parameters.normParams[i].gamma[j] -= lr * m_hat / (std::sqrt(v_hat) + epsilon);
      }

      for (ulong j = 0; j < this->parameters.normParams[i].numChannels; j++) {
        T g = this->accumDBNBeta[i][j] / n;
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
        this->parameters.normParams[i].gamma[j] -= lr * (this->accumDBNGamma[i][j] / n);
      }

      for (ulong j = 0; j < this->parameters.normParams[i].numChannels; j++) {
        this->parameters.normParams[i].beta[j] -= lr * (this->accumDBNBeta[i][j] / n);
      }
    }
  }
}

//===================================================================================================================//

template <typename T>
void CoreCPU<T>::train(ulong numSamples, const SampleProvider<T>& sampleProvider)
{
  if (numSamples == 0)
    throw std::runtime_error("No training samples provided");

  // Dispatch to batch-norm-aware training loop if any BATCHNORM layers exist
  if (this->hasBatchNorm) {
    this->trainBatchNorm(numSamples, sampleProvider);
    return;
  }

  ulong numEpochs = this->trainingConfig.numEpochs;

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

//===================================================================================================================//
//-- BatchNorm-aware training (layer-by-layer orchestration) --//
//===================================================================================================================//

template <typename T>
void CoreCPU<T>::trainBatchNorm(ulong numSamples, const SampleProvider<T>& sampleProvider)
{
  ulong numEpochs = this->trainingConfig.numEpochs;
  ulong batchSize = this->trainingConfig.batchSize;

  int numThreads = this->numThreads;

  if (numThreads <= 0)
    numThreads = QThreadPool::globalInstance()->maxThreadCount();

  this->trainingStart(numSamples);

  if (this->logLevel >= CNN::LogLevel::INFO)
    qDebug() << "CNN Training (BatchNorm):" << numEpochs << "epochs," << numSamples << "samples," << numThreads
             << "threads, batch size" << batchSize;

  // Create per-thread ANN workers (each owns its own ANN core for thread safety)
  std::vector<std::unique_ptr<CoreCPUWorker<T>>> workers;

  for (int i = 0; i < numThreads; i++) {
    workers.push_back(std::make_unique<CoreCPUWorker<T>>(this->coreConfig, this->layersConfig, this->parameters, true));
  }

  QMutex callbackMutex;

  // Sample index indirection for shuffling
  std::vector<ulong> sampleIndices(numSamples);
  std::iota(sampleIndices.begin(), sampleIndices.end(), 0);
  std::mt19937 rng(std::random_device{}());

  // Precompute CNN output shape
  Shape3D cnnOutputShape = this->layersConfig.validateShapes(this->inputShape);
  ulong flattenSize = cnnOutputShape.size();

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
      ulong N = batchSamples.size();

      // ---- FORWARD PASS (layer by layer) ----
      // Per-sample activations: currentActvs[n] is the current activation for sample n
      std::vector<Tensor3D<T>> currentActvs(N);

      for (ulong n = 0; n < N; n++) {
        currentActvs[n] = batchSamples[n].input;
      }

      // Per-sample intermediates for backprop: intermediates[n][layerIdx]
      std::vector<std::vector<Tensor3D<T>>> intermediates(N);

      for (ulong n = 0; n < N; n++) {
        intermediates[n].resize(this->layersConfig.cnnLayers.size());
      }

      // Per-sample pool max indices: poolMaxIndices[n][poolLayerIdx]
      std::vector<std::vector<std::vector<ulong>>> poolMaxIndices(N);

      // Per-norm-layer stats and xNorm (unified for both InstanceNorm and BatchNorm)
      ulong numNormLayers = this->parameters.normParams.size();
      std::vector<std::vector<Tensor3D<T>>> normXNormalized(numNormLayers);
      std::vector<std::vector<T>> normStatsMean(numNormLayers);
      std::vector<std::vector<T>> normStatsVar(numNormLayers);

      ulong convIdx = 0;
      ulong poolIdx = 0;
      ulong normIdx = 0;

      for (ulong layerIdx = 0; layerIdx < this->layersConfig.cnnLayers.size(); layerIdx++) {
        const CNNLayerConfig& layerConfig = this->layersConfig.cnnLayers[layerIdx];

        // Save intermediates for all samples
        for (ulong n = 0; n < N; n++) {
          intermediates[n][layerIdx] = currentActvs[n];
        }

        switch (layerConfig.type) {
        case LayerType::CONV: {
          const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);
          ulong ci = convIdx;

          for (ulong n = 0; n < N; n++) {
            currentActvs[n] = Conv2D<T>::propagate(currentActvs[n], conv, this->parameters.convParams[ci]);
          }

          convIdx++;
          break;
        }

        case LayerType::RELU: {
          for (ulong n = 0; n < N; n++) {
            currentActvs[n] = ReLU<T>::propagate(currentActvs[n]);
          }

          break;
        }

        case LayerType::POOL: {
          const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);

          for (ulong n = 0; n < N; n++) {
            poolMaxIndices[n].push_back({});
            currentActvs[n] = Pool<T>::propagate(currentActvs[n], pool, poolMaxIndices[n].back());
          }

          poolIdx++;
          break;
        }

        case LayerType::INSTANCENORM:
        case LayerType::BATCHNORM: {
          const auto& normConfig = std::get<NormLayerConfig>(layerConfig.config);
          ulong ni = normIdx;

          std::vector<Tensor3D<T>*> batchPtrs(N);

          for (ulong n = 0; n < N; n++)
            batchPtrs[n] = &currentActvs[n];

          Normalization<T>::propagate(batchPtrs, currentActvs[0].shape, this->parameters.normParams[ni], normConfig,
                                      layerConfig.type, true, &normXNormalized[ni], &normStatsMean[ni],
                                      &normStatsVar[ni]);

          normIdx++;
          break;
        }

        case LayerType::GLOBALAVGPOOL: {
          for (ulong n = 0; n < N; n++)
            GlobalAvgPool<T>::propagate(currentActvs[n], currentActvs[n].shape);
          break;
        }

        case LayerType::FLATTEN: {
          break;
        }
        }
      }

      // ---- ANN FORWARD + LOSS (parallel per sample) ----
      // Sync worker ANN cores with main parameters
      ANN::Parameters<T> mainANNParams = this->stepWorker->getANNCore()->getParameters();

      for (int i = 0; i < numThreads; i++) {
        workers[i]->getANNCore()->setParameters(mainANNParams);
        workers[i]->getANNCore()->resetAccumulators();
      }

      // Per-sample ANN outputs and losses
      std::vector<Output<T>> predictions(N);
      std::vector<T> sampleLosses(N);
      std::vector<Tensor1D<T>> dFlatInputs(N);

      // Distribute samples across workers for ANN forward + backward
      std::vector<ulong> workerSampleCounts(numThreads);

      for (int i = 0; i < numThreads; i++)
        workerSampleCounts[i] =
          N / static_cast<ulong>(numThreads) + (static_cast<ulong>(i) < N % static_cast<ulong>(numThreads) ? 1 : 0);

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
          // Flatten CNN output
          Tensor1D<T> flatInput = Flatten<T>::propagate(currentActvs[s]);

          // ANN forward
          ANN::Input<T> annInput(flatInput.begin(), flatInput.end());
          ANN::Output<T> annOutput = worker.getANNCore()->predict(annInput);
          predictions[s] = Output<T>(annOutput.begin(), annOutput.end());

          // Loss
          sampleLosses[s] = worker.calculateLoss(predictions[s], batchSamples[s].output);

          // ANN backward + accumulate
          ANN::Output<T> annExpected(batchSamples[s].output.begin(), batchSamples[s].output.end());
          ANN::Tensor1D<T> dFlatANN = worker.getANNCore()->backpropagate(annExpected);
          worker.getANNCore()->accumulate();

          dFlatInputs[s] = Tensor1D<T>(dFlatANN.begin(), dFlatANN.end());

          ulong completed = ++completedSamples;

          if (this->trainingCallback) {
            QMutexLocker locker(&callbackMutex);
            TrainingProgress<T> progress;
            progress.currentEpoch = e + 1;
            progress.totalEpochs = numEpochs;
            progress.currentSample = completed;
            progress.totalSamples = numSamples;
            progress.sampleLoss = sampleLosses[s];
            progress.epochLoss = 0;
            this->trainingCallback(progress);
          }
        }
      });

      // Sum losses
      T batchLoss = static_cast<T>(0);

      for (ulong n = 0; n < N; n++) {
        batchLoss += sampleLosses[n];
      }

      epochLoss += batchLoss;

      // ---- CNN BACKWARD PASS (layer by layer, reverse) ----
      // Unflatten ANN input gradients to CNN output gradients
      std::vector<Tensor3D<T>> dCurrents(N);

      for (ulong n = 0; n < N; n++) {
        dCurrents[n] = Flatten<T>::backpropagate(dFlatInputs[n], cnnOutputShape);
      }

      // Per-layer gradient accumulators
      ulong numConvLayers = this->parameters.convParams.size();
      std::vector<std::vector<T>> dConvFilters(numConvLayers);
      std::vector<std::vector<T>> dConvBiases(numConvLayers);

      for (ulong i = 0; i < numConvLayers; i++) {
        dConvFilters[i].assign(this->parameters.convParams[i].filters.size(), static_cast<T>(0));
        dConvBiases[i].assign(this->parameters.convParams[i].biases.size(), static_cast<T>(0));
      }

      std::vector<std::vector<T>> dBNGamma(numNormLayers);
      std::vector<std::vector<T>> dBNBeta(numNormLayers);

      for (ulong i = 0; i < numNormLayers; i++) {
        dBNGamma[i].assign(this->parameters.normParams[i].numChannels, static_cast<T>(0));
        dBNBeta[i].assign(this->parameters.normParams[i].numChannels, static_cast<T>(0));
      }

      convIdx = numConvLayers;
      poolIdx = poolMaxIndices.empty() ? 0 : poolMaxIndices[0].size();
      normIdx = numNormLayers;

      for (long layerIdx = static_cast<long>(this->layersConfig.cnnLayers.size()) - 1; layerIdx >= 0; layerIdx--) {
        const CNNLayerConfig& layerConfig = this->layersConfig.cnnLayers[static_cast<ulong>(layerIdx)];

        switch (layerConfig.type) {
        case LayerType::CONV: {
          convIdx--;
          const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);

          for (ulong n = 0; n < N; n++) {
            const Tensor3D<T>& layerInput = intermediates[n][static_cast<ulong>(layerIdx)];
            std::vector<T> sampleDFilters;
            std::vector<T> sampleDBiases;
            dCurrents[n] = Conv2D<T>::backpropagate(
              dCurrents[n], layerInput, conv, this->parameters.convParams[convIdx], sampleDFilters, sampleDBiases);

            for (ulong j = 0; j < sampleDFilters.size(); j++)
              dConvFilters[convIdx][j] += sampleDFilters[j];

            for (ulong j = 0; j < sampleDBiases.size(); j++)
              dConvBiases[convIdx][j] += sampleDBiases[j];
          }

          break;
        }

        case LayerType::RELU: {
          for (ulong n = 0; n < N; n++) {
            const Tensor3D<T>& layerInput = intermediates[n][static_cast<ulong>(layerIdx)];
            dCurrents[n] = ReLU<T>::backpropagate(dCurrents[n], layerInput);
          }

          break;
        }

        case LayerType::POOL: {
          poolIdx--;
          const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);

          for (ulong n = 0; n < N; n++) {
            const Tensor3D<T>& layerInput = intermediates[n][static_cast<ulong>(layerIdx)];
            dCurrents[n] = Pool<T>::backpropagate(dCurrents[n], layerInput.shape, pool, poolMaxIndices[n][poolIdx]);
          }

          break;
        }

        case LayerType::INSTANCENORM:
        case LayerType::BATCHNORM: {
          normIdx--;
          const auto& normConfig = std::get<NormLayerConfig>(layerConfig.config);

          std::vector<Tensor3D<T>*> dBatchPtrs(N);

          for (ulong n = 0; n < N; n++)
            dBatchPtrs[n] = &dCurrents[n];

          std::vector<T> layerDGamma;
          std::vector<T> layerDBeta;
          Shape3D layerShape = intermediates[0][static_cast<ulong>(layerIdx)].shape;
          Normalization<T>::backpropagate(dBatchPtrs, layerShape, this->parameters.normParams[normIdx], normConfig,
                                          layerConfig.type, normStatsMean[normIdx], normStatsVar[normIdx],
                                          normXNormalized[normIdx], layerDGamma, layerDBeta);

          for (ulong j = 0; j < layerDGamma.size(); j++)
            dBNGamma[normIdx][j] += layerDGamma[j];

          for (ulong j = 0; j < layerDBeta.size(); j++)
            dBNBeta[normIdx][j] += layerDBeta[j];

          break;
        }

        case LayerType::GLOBALAVGPOOL: {
          for (ulong n = 0; n < N; n++) {
            const Tensor3D<T>& layerInput = intermediates[n][static_cast<ulong>(layerIdx)];
            GlobalAvgPool<T>::backpropagate(dCurrents[n], layerInput.shape);
          }

          break;
        }

        case LayerType::FLATTEN: {
          break;
        }
        }
      }

      // ---- MERGE + UPDATE ----
      // Merge ANN parameters across workers (weighted average)
      for (int i = 0; i < numThreads; i++) {
        if (workerSampleCounts[i] > 0)
          workers[i]->getANNCore()->update(workerSampleCounts[i]);
      }

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

      // Update CNN parameters using accumulated gradients
      this->resetGlobalCNNAccumulators();

      for (ulong i = 0; i < numConvLayers; i++) {
        for (ulong j = 0; j < dConvFilters[i].size(); j++)
          this->accumDConvFilters[i][j] = dConvFilters[i][j];

        for (ulong j = 0; j < dConvBiases[i].size(); j++)
          this->accumDConvBiases[i][j] = dConvBiases[i][j];
      }

      for (ulong i = 0; i < numNormLayers; i++) {
        for (ulong j = 0; j < dBNGamma[i].size(); j++)
          this->accumDBNGamma[i][j] = dBNGamma[i][j];

        for (ulong j = 0; j < dBNBeta[i].size(); j++)
          this->accumDBNBeta[i][j] = dBNBeta[i][j];
      }

      this->updateCNNParameters(currentBatchSize);
      // Note: BatchNorm running stats are updated during propagate() directly
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

// Explicit template instantiations
template class CNN::CoreCPU<int>;
template class CNN::CoreCPU<double>;
template class CNN::CoreCPU<float>;