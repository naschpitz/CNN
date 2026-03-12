#include "CNN_GPUBufferManager.hpp"
#include "CNN_SlidingStrategy.hpp"
#include "CNN_Conv2D.hpp"
#include "CNN_ReLU.hpp"
#include "CNN_Pool.hpp"
#include "CNN_Flatten.hpp"
#include "CNN_Worker.hpp"

#include <iostream>

using namespace CNN;

//===================================================================================================================//

template <typename T>
GPUBufferManager<T>::GPUBufferManager(OpenCLWrapper::Core* core, const CoreConfig<T>& coreConfig,
                                      Parameters<T>& parameters, LogLevel logLevel)
  : core(core),
    coreConfig(coreConfig),
    parameters(parameters),
    logLevel(logLevel)
{
}

//===================================================================================================================//
//-- Compute layer offsets --//
//===================================================================================================================//

template <typename T>
void GPUBufferManager<T>::computeLayerOffsets()
{
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  Shape3D currentShape = this->coreConfig.inputShape;

  // First entry: input tensor
  this->layerInfos.clear();
  this->layerInfos.push_back({0, currentShape.size()});
  ulong offset = currentShape.size();

  ulong convIdx = 0;
  ulong poolIdx = 0;
  this->totalFilterSize = 0;
  this->totalBiasSize = 0;
  this->totalPoolIndexSize = 0;
  this->totalNormParamSize = 0;
  this->convInfos.clear();
  this->poolInfos.clear();
  this->normInfos.clear();

  for (const auto& layerConfig : cnnLayers) {
    Shape3D outShape = currentShape;

    switch (layerConfig.type) {
    case LayerType::CONV: {
      const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
      ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
      outShape = {conv.numFilters, outH, outW};

      ConvInfo ci;
      ci.filterOffset = this->totalFilterSize;
      ci.biasOffset = this->totalBiasSize;
      ci.numFilterElems = conv.numFilters * currentShape.c * conv.filterH * conv.filterW;
      ci.numBiases = conv.numFilters;
      this->convInfos.push_back(ci);

      this->totalFilterSize += ci.numFilterElems;
      this->totalBiasSize += ci.numBiases;
      convIdx++;
      break;
    }

    case LayerType::RELU: {
      // ReLU doesn't change shape
      break;
    }

    case LayerType::POOL: {
      const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);
      ulong outH = (currentShape.h - pool.poolH) / pool.strideY + 1;
      ulong outW = (currentShape.w - pool.poolW) / pool.strideX + 1;
      outShape = {currentShape.c, outH, outW};

      PoolInfo pi;
      pi.indexOffset = this->totalPoolIndexSize;

      if (pool.poolType == PoolTypeEnum::MAX) {
        pi.indexSize = outShape.size();
        this->totalPoolIndexSize += pi.indexSize;
      } else {
        pi.indexSize = 0; // Avg pool doesn't need index storage
      }

      this->poolInfos.push_back(pi);
      poolIdx++;
      break;
    }

    case LayerType::GLOBALAVGPOOL: {
      // Global average pooling reduces spatial dims to 1x1
      outShape = {currentShape.c, 1, 1};
      break;
    }

    case LayerType::INSTANCENORM: {
      // Instance norm doesn't change shape
      InstanceNormInfo bi;
      bi.paramOffset = this->totalNormParamSize;
      bi.numChannels = currentShape.c;
      this->normInfos.push_back(bi);
      this->totalNormParamSize += bi.numChannels;
      break;
    }

    case LayerType::BATCHNORM: {
      // Batch norm doesn't change shape
      InstanceNormInfo bi;
      bi.paramOffset = this->totalNormParamSize;
      bi.numChannels = currentShape.c;
      this->normInfos.push_back(bi);
      this->totalNormParamSize += bi.numChannels;
      break;
    }

    case LayerType::FLATTEN: {
      // Flatten shares the same buffer as its input — no GPU kernel copies data
      this->layerInfos.push_back({this->layerInfos.back().actvOffset, outShape.size()});
      currentShape = outShape;
      continue; // Skip the normal push_back and offset advancement below
    }
    }

    this->layerInfos.push_back({offset, outShape.size()});
    offset += outShape.size();
    currentShape = outShape;
  }

  this->totalActvSize = offset;
}

//===================================================================================================================//
//-- Load OpenCL sources --//
//===================================================================================================================//

template <typename T>
void GPUBufferManager<T>::loadSources(bool skipDefines)
{
  if (this->logLevel >= CNN::LogLevel::INFO)
    std::cout << "Loading CNN OpenCL kernels...\n";

  // Resolve .cl file paths relative to the source file's directory (via __FILE__)
  std::string srcFile = __FILE__;
  std::string srcDir = srcFile.substr(0, srcFile.find_last_of("/\\") + 1);

  if (!skipDefines) {
    this->core->addSourceFile(srcDir + "opencl/CNN_Defines.hpp.cl");
  }

  this->core->addSourceFile(srcDir + "opencl/CNN_Propagate.cpp.cl");
  this->core->addSourceFile(srcDir + "opencl/CNN_Backpropagate.cpp.cl");
  this->core->addSourceFile(srcDir + "opencl/CNN_Update.cpp.cl");
  this->core->addSourceFile(srcDir + "opencl/CNN_Bridge.cpp.cl");
  this->core->addSourceFile(srcDir + "opencl/CNN_Normalization.cpp.cl");
  this->core->addSourceFile(srcDir + "opencl/CNN_GlobalAvgPool.cpp.cl");

  if (this->logLevel >= CNN::LogLevel::INFO)
    std::cout << "CNN OpenCL kernels loaded.\n";
}

//===================================================================================================================//
//-- Allocate GPU buffers --//
//===================================================================================================================//

template <typename T>
void GPUBufferManager<T>::allocateBuffers(ulong batchSize)
{
  this->batchSize = batchSize;
  ulong batchActvSize = this->totalActvSize * batchSize;

  // Activation and gradient buffers — sized for the full batch
  this->core->template allocateBuffer<T>("cnn_actvs", batchActvSize);
  this->core->template allocateBuffer<T>("cnn_grads", batchActvSize);

  // Filter and bias parameter buffers
  if (this->totalFilterSize > 0) {
    this->core->template allocateBuffer<T>("cnn_filters", this->totalFilterSize);
    this->core->template allocateBuffer<T>("cnn_dFilters", this->totalFilterSize);
    this->core->template allocateBuffer<T>("cnn_accum_dFilters", this->totalFilterSize);
  }

  if (this->totalBiasSize > 0) {
    this->core->template allocateBuffer<T>("cnn_biases", this->totalBiasSize);
    this->core->template allocateBuffer<T>("cnn_dBiases", this->totalBiasSize);
    this->core->template allocateBuffer<T>("cnn_accum_dBiases", this->totalBiasSize);
  }

  // Pool index buffer — sized for the full batch
  if (this->totalPoolIndexSize > 0) {
    this->core->template allocateBuffer<ulong>("cnn_pool_indices", this->totalPoolIndexSize * batchSize);
  }

  // Batch norm buffers
  if (this->totalNormParamSize > 0) {
    this->core->template allocateBuffer<T>("cnn_norm_gamma", this->totalNormParamSize);
    this->core->template allocateBuffer<T>("cnn_norm_beta", this->totalNormParamSize);
    this->core->template allocateBuffer<T>("cnn_norm_running_mean", this->totalNormParamSize);
    this->core->template allocateBuffer<T>("cnn_norm_running_var", this->totalNormParamSize);
    this->core->template allocateBuffer<T>("cnn_norm_dGamma", this->totalNormParamSize);
    this->core->template allocateBuffer<T>("cnn_norm_dBeta", this->totalNormParamSize);
    this->core->template allocateBuffer<T>("cnn_accum_norm_dGamma", this->totalNormParamSize);
    this->core->template allocateBuffer<T>("cnn_accum_norm_dBeta", this->totalNormParamSize);
    // Per-sample batch mean/var for backprop
    this->core->template allocateBuffer<T>("cnn_norm_batch_mean", this->totalNormParamSize);
    this->core->template allocateBuffer<T>("cnn_norm_batch_var", this->totalNormParamSize);
    // Accumulators for batch mean/var across samples (for running stats update)
    this->core->template allocateBuffer<T>("cnn_accum_norm_batch_mean", this->totalNormParamSize);
    this->core->template allocateBuffer<T>("cnn_accum_norm_batch_var", this->totalNormParamSize);
    // Normalized values for backprop — sized for the full batch
    this->core->template allocateBuffer<T>("cnn_norm_xnorm", batchActvSize);
  }

  // Write initial filter/bias values to GPU
  if (this->totalFilterSize > 0) {
    std::vector<T> flatFilters(this->totalFilterSize);

    for (ulong i = 0; i < this->convInfos.size(); i++) {
      const auto& ci = this->convInfos[i];
      const auto& cp = this->parameters.convParams[i];

      for (ulong j = 0; j < ci.numFilterElems; j++) {
        flatFilters[ci.filterOffset + j] = cp.filters[j];
      }
    }

    this->core->template writeBuffer<T>("cnn_filters", flatFilters, 0);
  }

  if (this->totalBiasSize > 0) {
    std::vector<T> flatBiases(this->totalBiasSize);

    for (ulong i = 0; i < this->convInfos.size(); i++) {
      const auto& ci = this->convInfos[i];
      const auto& cp = this->parameters.convParams[i];

      for (ulong j = 0; j < ci.numBiases; j++) {
        flatBiases[ci.biasOffset + j] = cp.biases[j];
      }
    }

    this->core->template writeBuffer<T>("cnn_biases", flatBiases, 0);
  }

  // Write initial batch norm parameters to GPU
  if (this->totalNormParamSize > 0) {
    std::vector<T> flatGamma(this->totalNormParamSize);
    std::vector<T> flatBeta(this->totalNormParamSize);
    std::vector<T> flatRunningMean(this->totalNormParamSize);
    std::vector<T> flatRunningVar(this->totalNormParamSize);

    for (ulong i = 0; i < this->normInfos.size(); i++) {
      const auto& bi = this->normInfos[i];
      const auto& bp = this->parameters.normParams[i];

      for (ulong j = 0; j < bi.numChannels; j++) {
        flatGamma[bi.paramOffset + j] = bp.gamma[j];
        flatBeta[bi.paramOffset + j] = bp.beta[j];
        flatRunningMean[bi.paramOffset + j] = bp.runningMean[j];
        flatRunningVar[bi.paramOffset + j] = bp.runningVar[j];
      }
    }

    this->core->template writeBuffer<T>("cnn_norm_gamma", flatGamma, 0);
    this->core->template writeBuffer<T>("cnn_norm_beta", flatBeta, 0);
    this->core->template writeBuffer<T>("cnn_norm_running_mean", flatRunningMean, 0);
    this->core->template writeBuffer<T>("cnn_norm_running_var", flatRunningVar, 0);
  }

  // Adam optimizer buffers
  if (this->coreConfig.trainingConfig.optimizer.type == OptimizerType::ADAM) {
    T zero = static_cast<T>(0);

    if (this->totalFilterSize > 0) {
      this->core->template allocateBuffer<T>("cnn_adam_m_filters", this->totalFilterSize);
      this->core->template allocateBuffer<T>("cnn_adam_v_filters", this->totalFilterSize);
      this->core->template fillBuffer<T>("cnn_adam_m_filters", zero, this->totalFilterSize);
      this->core->template fillBuffer<T>("cnn_adam_v_filters", zero, this->totalFilterSize);
    }

    if (this->totalBiasSize > 0) {
      this->core->template allocateBuffer<T>("cnn_adam_m_biases", this->totalBiasSize);
      this->core->template allocateBuffer<T>("cnn_adam_v_biases", this->totalBiasSize);
      this->core->template fillBuffer<T>("cnn_adam_m_biases", zero, this->totalBiasSize);
      this->core->template fillBuffer<T>("cnn_adam_v_biases", zero, this->totalBiasSize);
    }

    if (this->totalNormParamSize > 0) {
      this->core->template allocateBuffer<T>("cnn_adam_m_norm_gamma", this->totalNormParamSize);
      this->core->template allocateBuffer<T>("cnn_adam_v_norm_gamma", this->totalNormParamSize);
      this->core->template allocateBuffer<T>("cnn_adam_m_norm_beta", this->totalNormParamSize);
      this->core->template allocateBuffer<T>("cnn_adam_v_norm_beta", this->totalNormParamSize);
      this->core->template fillBuffer<T>("cnn_adam_m_norm_gamma", zero, this->totalNormParamSize);
      this->core->template fillBuffer<T>("cnn_adam_v_norm_gamma", zero, this->totalNormParamSize);
      this->core->template fillBuffer<T>("cnn_adam_m_norm_beta", zero, this->totalNormParamSize);
      this->core->template fillBuffer<T>("cnn_adam_v_norm_beta", zero, this->totalNormParamSize);
    }
  }

  if (this->logLevel >= CNN::LogLevel::INFO)
    std::cout << "CNN GPU buffers allocated (batchSize=" << batchSize << ").\n";
}

//===================================================================================================================//
//-- Build ANN GPU worker --//
//===================================================================================================================//

template <typename T>
void GPUBufferManager<T>::buildANNWorker()
{
  // Compute flatten size from cnn output shape
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  Shape3D currentShape = this->coreConfig.inputShape;

  for (const auto& layerConfig : cnnLayers) {
    switch (layerConfig.type) {
    case LayerType::CONV: {
      const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
      ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
      currentShape = {conv.numFilters, outH, outW};
      break;
    }

    case LayerType::RELU:
      break;
    case LayerType::POOL: {
      const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);
      ulong outH = (currentShape.h - pool.poolH) / pool.strideY + 1;
      ulong outW = (currentShape.w - pool.poolW) / pool.strideX + 1;
      currentShape = {currentShape.c, outH, outW};
      break;
    }

    case LayerType::GLOBALAVGPOOL:
      currentShape = {currentShape.c, 1, 1};
      break;
    case LayerType::INSTANCENORM:
      break;
    case LayerType::BATCHNORM:
      break;
    case LayerType::FLATTEN:
      break;
    }
  }

  this->cnnOutputShape = currentShape;
  this->flattenSize = currentShape.size();

  // Build ANN layers: first layer = flatten size (input), rest from denseLayers
  ANN::LayersConfig annLayers;

  ANN::Layer inputLayer;
  inputLayer.numNeurons = this->flattenSize;
  inputLayer.actvFuncType = ANN::ActvFuncType::RELU;
  annLayers.push_back(inputLayer);

  for (const auto& denseConfig : this->coreConfig.layersConfig.denseLayers) {
    ANN::Layer layer;
    layer.numNeurons = denseConfig.numNeurons;
    layer.actvFuncType = denseConfig.actvFuncType;
    annLayers.push_back(layer);
  }

  // Training config
  ANN::TrainingConfig<T> annTrainingConfig;
  annTrainingConfig.numEpochs = this->coreConfig.trainingConfig.numEpochs;
  annTrainingConfig.learningRate = this->coreConfig.trainingConfig.learningRate;
  annTrainingConfig.dropoutRate = this->coreConfig.trainingConfig.dropoutRate;
  annTrainingConfig.optimizer.type = static_cast<ANN::OptimizerType>(this->coreConfig.trainingConfig.optimizer.type);
  annTrainingConfig.optimizer.beta1 = this->coreConfig.trainingConfig.optimizer.beta1;
  annTrainingConfig.optimizer.beta2 = this->coreConfig.trainingConfig.optimizer.beta2;
  annTrainingConfig.optimizer.epsilon = this->coreConfig.trainingConfig.optimizer.epsilon;

  // Cost function config
  ANN::CostFunctionConfig<T> annCostFunctionConfig;
  annCostFunctionConfig.type = static_cast<ANN::CostFunctionType>(this->coreConfig.costFunctionConfig.type);
  annCostFunctionConfig.weights = this->coreConfig.costFunctionConfig.weights;

  // Create ANN GPU worker on the shared core
  this->annGPUWorker = std::make_unique<ANN::CoreGPUWorker<T>>(
    annLayers, annTrainingConfig, this->coreConfig.parameters.denseParams, annCostFunctionConfig, *this->core,
    this->coreConfig.progressReports, static_cast<ANN::LogLevel>(this->coreConfig.logLevel));

  // Load ANN sources (skip defines — CNN_Defines.hpp.cl already defined TYPE, ActvFuncType, Layer)
  this->annGPUWorker->bufferManager->loadSources(true);

  // Allocate ANN GPU buffers
  this->annGPUWorker->bufferManager->allocateBuffers();

  // Buffer for accumulating loss on GPU (avoids per-sample GPU→CPU readback)
  this->core->template allocateBuffer<T>("accum_loss", 1);
}

//===================================================================================================================//
//-- Parameter synchronization: GPU → CPU --//
//===================================================================================================================//

template <typename T>
void GPUBufferManager<T>::syncParametersFromGPU()
{
  // Read CNN filters from GPU and scatter back to per-layer parameters
  if (this->totalFilterSize > 0) {
    std::vector<T> flatFilters(this->totalFilterSize);
    this->core->template readBuffer<T>("cnn_filters", flatFilters, 0);

    for (ulong i = 0; i < this->convInfos.size(); i++) {
      ulong offset = this->convInfos[i].filterOffset;
      ulong count = this->convInfos[i].numFilterElems;
      this->parameters.convParams[i].filters.assign(flatFilters.begin() + static_cast<long>(offset),
                                                    flatFilters.begin() + static_cast<long>(offset + count));
    }
  }

  // Read CNN biases from GPU and scatter back to per-layer parameters
  if (this->totalBiasSize > 0) {
    std::vector<T> flatBiases(this->totalBiasSize);
    this->core->template readBuffer<T>("cnn_biases", flatBiases, 0);

    for (ulong i = 0; i < this->convInfos.size(); i++) {
      ulong offset = this->convInfos[i].biasOffset;
      ulong count = this->convInfos[i].numBiases;
      this->parameters.convParams[i].biases.assign(flatBiases.begin() + static_cast<long>(offset),
                                                   flatBiases.begin() + static_cast<long>(offset + count));
    }
  }

  // Read batch norm parameters from GPU
  if (this->totalNormParamSize > 0) {
    std::vector<T> flatGamma(this->totalNormParamSize);
    std::vector<T> flatBeta(this->totalNormParamSize);
    std::vector<T> flatRunningMean(this->totalNormParamSize);
    std::vector<T> flatRunningVar(this->totalNormParamSize);

    this->core->template readBuffer<T>("cnn_norm_gamma", flatGamma, 0);
    this->core->template readBuffer<T>("cnn_norm_beta", flatBeta, 0);
    this->core->template readBuffer<T>("cnn_norm_running_mean", flatRunningMean, 0);
    this->core->template readBuffer<T>("cnn_norm_running_var", flatRunningVar, 0);

    for (ulong i = 0; i < this->normInfos.size(); i++) {
      ulong offset = this->normInfos[i].paramOffset;
      ulong count = this->normInfos[i].numChannels;
      this->parameters.normParams[i].gamma.assign(flatGamma.begin() + static_cast<long>(offset),
                                                  flatGamma.begin() + static_cast<long>(offset + count));
      this->parameters.normParams[i].beta.assign(flatBeta.begin() + static_cast<long>(offset),
                                                 flatBeta.begin() + static_cast<long>(offset + count));
      this->parameters.normParams[i].runningMean.assign(flatRunningMean.begin() + static_cast<long>(offset),
                                                        flatRunningMean.begin() + static_cast<long>(offset + count));
      this->parameters.normParams[i].runningVar.assign(flatRunningVar.begin() + static_cast<long>(offset),
                                                       flatRunningVar.begin() + static_cast<long>(offset + count));
    }
  }

  // Sync ANN dense layer parameters from GPU
  this->annGPUWorker->bufferManager->syncParametersFromGPU();
  this->parameters.denseParams = this->annGPUWorker->getParameters();
}

//===================================================================================================================//
//-- Accumulator operations --//
//===================================================================================================================//

template <typename T>
void GPUBufferManager<T>::resetAccumulators()
{
  T zero = static_cast<T>(0);

  if (this->totalFilterSize > 0) {
    this->core->template fillBuffer<T>("cnn_accum_dFilters", zero, this->totalFilterSize);
  }

  if (this->totalBiasSize > 0) {
    this->core->template fillBuffer<T>("cnn_accum_dBiases", zero, this->totalBiasSize);
  }

  if (this->totalNormParamSize > 0) {
    this->core->template fillBuffer<T>("cnn_accum_norm_dGamma", zero, this->totalNormParamSize);
    this->core->template fillBuffer<T>("cnn_accum_norm_dBeta", zero, this->totalNormParamSize);
    this->core->template fillBuffer<T>("cnn_accum_norm_batch_mean", zero, this->totalNormParamSize);
    this->core->template fillBuffer<T>("cnn_accum_norm_batch_var", zero, this->totalNormParamSize);
  }

  this->annGPUWorker->resetAccumulators();
}

//===================================================================================================================//

template <typename T>
void GPUBufferManager<T>::readAccumulatedGradients(std::vector<T>& accumFilters, std::vector<T>& accumBiases)
{
  accumFilters.resize(this->totalFilterSize);
  accumBiases.resize(this->totalBiasSize);

  if (this->totalFilterSize > 0) {
    this->core->template readBuffer<T>("cnn_accum_dFilters", accumFilters, 0);
  }

  if (this->totalBiasSize > 0) {
    this->core->template readBuffer<T>("cnn_accum_dBiases", accumBiases, 0);
  }
}

//===================================================================================================================//

template <typename T>
void GPUBufferManager<T>::setAccumulators(const std::vector<T>& accumFilters, const std::vector<T>& accumBiases)
{
  if (this->totalFilterSize > 0) {
    this->core->template writeBuffer<T>("cnn_accum_dFilters", accumFilters, 0);
  }

  if (this->totalBiasSize > 0) {
    this->core->template writeBuffer<T>("cnn_accum_dBiases", accumBiases, 0);
  }
}

//===================================================================================================================//

template <typename T>
void GPUBufferManager<T>::readBNAccumulatedGradients(std::vector<T>& accumGamma, std::vector<T>& accumBeta)
{
  accumGamma.resize(this->totalNormParamSize);
  accumBeta.resize(this->totalNormParamSize);

  if (this->totalNormParamSize > 0) {
    this->core->template readBuffer<T>("cnn_accum_norm_dGamma", accumGamma, 0);
    this->core->template readBuffer<T>("cnn_accum_norm_dBeta", accumBeta, 0);
  }
}

//===================================================================================================================//

template <typename T>
void GPUBufferManager<T>::setBNAccumulators(const std::vector<T>& accumGamma, const std::vector<T>& accumBeta)
{
  if (this->totalNormParamSize > 0) {
    this->core->template writeBuffer<T>("cnn_accum_norm_dGamma", accumGamma, 0);
    this->core->template writeBuffer<T>("cnn_accum_norm_dBeta", accumBeta, 0);
  }
}

//===================================================================================================================//

template <typename T>
void GPUBufferManager<T>::readANNAccumulatedGradients(ANN::Tensor1D<T>& accumWeights, ANN::Tensor1D<T>& accumBiases)
{
  this->annGPUWorker->bufferManager->readAccumulatedGradients(accumWeights, accumBiases);
}

//===================================================================================================================//

template <typename T>
void GPUBufferManager<T>::setANNAccumulators(const ANN::Tensor1D<T>& accumWeights, const ANN::Tensor1D<T>& accumBiases)
{
  this->annGPUWorker->bufferManager->setAccumulators(accumWeights, accumBiases);
}

//===================================================================================================================//
// Explicit template instantiations.
//===================================================================================================================//

template class CNN::GPUBufferManager<int>;
template class CNN::GPUBufferManager<float>;
template class CNN::GPUBufferManager<double>;