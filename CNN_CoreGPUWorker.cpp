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

  // Compute CNN output shape
  this->cnnOutputShape = config.layersConfig.validateShapes(config.inputShape);
  this->flattenSize = this->cnnOutputShape.size();

  // Initialize conv parameters (He initialization if not loaded)
  Worker<T>::initializeConvParams(config.layersConfig, config.inputShape, this->parameters);

  // Compute buffer offsets for all layers
  this->computeLayerOffsets();

  // Load OpenCL sources (defines first, then kernels)
  this->loadSources(false);

  // Build ANN GPU worker on the shared core (loads ANN sources + allocates ANN buffers)
  this->buildANNWorker();

  // Allocate CNN GPU buffers and write initial parameters
  this->allocateBuffers();
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

  // Compute CNN output shape
  this->cnnOutputShape = config.layersConfig.validateShapes(config.inputShape);
  this->flattenSize = this->cnnOutputShape.size();

  // Initialize conv parameters (He initialization if not loaded)
  Worker<T>::initializeConvParams(config.layersConfig, config.inputShape, this->parameters);

  // Compute buffer offsets for all layers
  this->computeLayerOffsets();

  // Caller must call loadSources(), buildANNWorker(), allocateBuffers() manually
}

//===================================================================================================================//
//-- Initialization: compute layer offsets --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::computeLayerOffsets()
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
  this->convInfos.clear();
  this->poolInfos.clear();

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
      pi.indexSize = outShape.size();
      this->poolInfos.push_back(pi);

      this->totalPoolIndexSize += pi.indexSize;
      poolIdx++;
      break;
    }

    case LayerType::FLATTEN: {
      // Flatten shares the same buffer as its input — no GPU kernel copies data,
      // so we point to the same offset as the previous layer's output.
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
//===================================================================================================================//
//-- Initialization: load OpenCL sources --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::loadSources(bool skipDefines)
{
  if (this->logLevel >= CNN::LogLevel::INFO)
    std::cout << "Loading CNN OpenCL kernels...\n";

  // Resolve .cl file paths relative to the source file's directory (via __FILE__),
  // so the kernels are found regardless of the current working directory.
  std::string srcFile = __FILE__;
  std::string srcDir = srcFile.substr(0, srcFile.find_last_of("/\\") + 1);

  if (!skipDefines) {
    this->core->addSourceFile(srcDir + "opencl/CNN_Defines.hpp.cl");
  }

  this->core->addSourceFile(srcDir + "opencl/CNN_Kernels.cpp.cl");

  if (this->logLevel >= CNN::LogLevel::INFO)
    std::cout << "CNN OpenCL kernels loaded.\n";
}

//===================================================================================================================//
//-- Initialization: allocate GPU buffers --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::allocateBuffers()
{
  if (this->logLevel >= CNN::LogLevel::INFO)
    std::cout << "Allocating CNN GPU buffers...\n";

  // Activation and gradient buffers (same layout)
  this->core->template allocateBuffer<T>("cnn_actvs", this->totalActvSize);
  this->core->template allocateBuffer<T>("cnn_grads", this->totalActvSize);

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

  // Pool index buffer
  if (this->totalPoolIndexSize > 0) {
    this->core->template allocateBuffer<ulong>("cnn_pool_indices", this->totalPoolIndexSize);
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

  if (this->logLevel >= CNN::LogLevel::INFO)
    std::cout << "CNN GPU buffers allocated.\n";
}

//===================================================================================================================//
//-- Initialization: build ANN GPU worker --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::buildANNWorker()
{
  // Build ANN layers: first layer = flatten size (input), rest from denseLayers
  ANN::LayersConfig annLayers;

  ANN::Layer inputLayer;
  inputLayer.numNeurons = this->flattenSize;
  inputLayer.actvFuncType = ANN::ActvFuncType::RELU; // placeholder, never used by ANN
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

  // Cost function config
  ANN::CostFunctionConfig<T> annCostFunctionConfig;
  annCostFunctionConfig.type = static_cast<ANN::CostFunctionType>(this->coreConfig.costFunctionConfig.type);
  annCostFunctionConfig.weights = this->coreConfig.costFunctionConfig.weights;

  // Create ANN GPU worker on the shared core
  this->annGPUWorker = std::make_unique<ANN::CoreGPUWorker<T>>(
    annLayers, annTrainingConfig, this->coreConfig.parameters.denseParams, annCostFunctionConfig, *this->core,
    this->coreConfig.progressReports, static_cast<ANN::LogLevel>(this->coreConfig.logLevel));

  // Load ANN sources (skip defines — CNN_Defines.hpp.cl already defined TYPE, ActvFuncType, Layer)
  this->annGPUWorker->loadSources(true);

  // Allocate ANN GPU buffers
  this->annGPUWorker->allocateBuffers();

  // Buffer for accumulating loss on GPU (avoids per-sample GPU→CPU readback)
  this->core->template allocateBuffer<T>("accum_loss", 1);
}

//===================================================================================================================//
//-- Kernel building: propagate --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::addPropagateKernels()
{
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  Shape3D currentShape = this->coreConfig.inputShape;
  ulong convIdx = 0;
  ulong poolIdx = 0;

  for (ulong i = 0; i < cnnLayers.size(); i++) {
    const auto& layerConfig = cnnLayers[i];
    std::string layerStr = std::to_string(i);

    ulong inOffset = this->layerInfos[i].actvOffset;
    ulong outOffset = this->layerInfos[i + 1].actvOffset;

    switch (layerConfig.type) {
    case LayerType::CONV: {
      const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
      ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
      ulong nElements = conv.numFilters * outH * outW;

      std::string kernelId = "calculate_conv2d_layer" + layerStr;
      this->core->addKernel(kernelId, "calculate_conv2d", nElements, 0);
      this->core->template addArgument<T>(kernelId, "cnn_actvs");
      this->core->template addArgument<T>(kernelId, "cnn_filters");
      this->core->template addArgument<T>(kernelId, "cnn_biases");
      this->core->template addArgument<ulong>(kernelId, inOffset);
      this->core->template addArgument<ulong>(kernelId, outOffset);
      this->core->template addArgument<ulong>(kernelId, this->convInfos[convIdx].filterOffset);
      this->core->template addArgument<ulong>(kernelId, this->convInfos[convIdx].biasOffset);
      this->core->template addArgument<ulong>(kernelId, currentShape.c);
      this->core->template addArgument<ulong>(kernelId, currentShape.h);
      this->core->template addArgument<ulong>(kernelId, currentShape.w);
      this->core->template addArgument<ulong>(kernelId, conv.numFilters);
      this->core->template addArgument<ulong>(kernelId, conv.filterH);
      this->core->template addArgument<ulong>(kernelId, conv.filterW);
      this->core->template addArgument<ulong>(kernelId, conv.strideY);
      this->core->template addArgument<ulong>(kernelId, conv.strideX);
      this->core->template addArgument<ulong>(kernelId, padY);
      this->core->template addArgument<ulong>(kernelId, padX);
      this->core->template addArgument<ulong>(kernelId, outH);
      this->core->template addArgument<ulong>(kernelId, outW);

      currentShape = {conv.numFilters, outH, outW};
      convIdx++;
      break;
    }

    case LayerType::RELU: {
      ulong size = currentShape.size();
      std::string kernelId = "calculate_relu_layer" + layerStr;
      this->core->addKernel(kernelId, "calculate_relu", size, 0);
      this->core->template addArgument<T>(kernelId, "cnn_actvs");
      this->core->template addArgument<ulong>(kernelId, inOffset);
      this->core->template addArgument<ulong>(kernelId, outOffset);
      this->core->template addArgument<ulong>(kernelId, size);
      break;
    }

    case LayerType::POOL: {
      const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);
      ulong outH = (currentShape.h - pool.poolH) / pool.strideY + 1;
      ulong outW = (currentShape.w - pool.poolW) / pool.strideX + 1;
      ulong nElements = currentShape.c * outH * outW;

      std::string kernelId = "calculate_maxpool_layer" + layerStr;
      this->core->addKernel(kernelId, "calculate_maxpool", nElements, 0);
      this->core->template addArgument<T>(kernelId, "cnn_actvs");
      this->core->template addArgument<ulong>(kernelId, "cnn_pool_indices");
      this->core->template addArgument<ulong>(kernelId, inOffset);
      this->core->template addArgument<ulong>(kernelId, outOffset);
      this->core->template addArgument<ulong>(kernelId, this->poolInfos[poolIdx].indexOffset);
      this->core->template addArgument<ulong>(kernelId, currentShape.c);
      this->core->template addArgument<ulong>(kernelId, currentShape.h);
      this->core->template addArgument<ulong>(kernelId, currentShape.w);
      this->core->template addArgument<ulong>(kernelId, pool.poolH);
      this->core->template addArgument<ulong>(kernelId, pool.poolW);
      this->core->template addArgument<ulong>(kernelId, pool.strideY);
      this->core->template addArgument<ulong>(kernelId, pool.strideX);
      this->core->template addArgument<ulong>(kernelId, outH);
      this->core->template addArgument<ulong>(kernelId, outW);

      currentShape = {currentShape.c, outH, outW};
      poolIdx++;
      break;
    }

    case LayerType::FLATTEN: {
      // No GPU kernel needed for flatten - data stays in cnn_actvs at same offset
      break;
    }
    }
  }
}

//===================================================================================================================//
//-- Kernel building: backpropagate --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::addBackpropagateKernels()
{
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  ulong numLayers = cnnLayers.size();

  // Precompute shapes for each layer (propagate direction)
  std::vector<Shape3D> shapes(numLayers + 1);
  shapes[0] = this->coreConfig.inputShape;

  for (ulong i = 0; i < numLayers; i++) {
    Shape3D inShape = shapes[i];

    switch (cnnLayers[i].type) {
    case LayerType::CONV: {
      const auto& conv = std::get<ConvLayerConfig>(cnnLayers[i].config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = (inShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
      ulong outW = (inShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
      shapes[i + 1] = {conv.numFilters, outH, outW};
      break;
    }

    case LayerType::RELU:
      shapes[i + 1] = inShape;
      break;
    case LayerType::POOL: {
      const auto& pool = std::get<PoolLayerConfig>(cnnLayers[i].config);
      ulong outH = (inShape.h - pool.poolH) / pool.strideY + 1;
      ulong outW = (inShape.w - pool.poolW) / pool.strideX + 1;
      shapes[i + 1] = {inShape.c, outH, outW};
      break;
    }

    case LayerType::FLATTEN:
      shapes[i + 1] = inShape;
      break;
    }
  }

  // Iterate through layers in reverse
  ulong convIdx = this->convInfos.size();
  ulong poolIdx = this->poolInfos.size();

  for (long i = static_cast<long>(numLayers) - 1; i >= 0; i--) {
    const auto& layerConfig = cnnLayers[static_cast<ulong>(i)];
    std::string layerStr = std::to_string(i);

    Shape3D inShape = shapes[static_cast<ulong>(i)];
    Shape3D outShape = shapes[static_cast<ulong>(i) + 1];

    ulong gradInOffset = this->layerInfos[static_cast<ulong>(i)].actvOffset;
    ulong gradOutOffset = this->layerInfos[static_cast<ulong>(i) + 1].actvOffset;
    ulong actvInOffset = this->layerInfos[static_cast<ulong>(i)].actvOffset;

    switch (layerConfig.type) {
    case LayerType::CONV: {
      convIdx--;
      const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = outShape.h;
      ulong outW = outShape.w;

      // calculate_dCost_dFilters — one work-group per filter element, tree-reduce output positions
      ulong nFilterElems = this->convInfos[convIdx].numFilterElems;
      std::string filterId = "calculate_dCost_dFilters_layer" + layerStr;
      ulong filterLocalWS = 256;
      ulong filterGlobalWS = nFilterElems * filterLocalWS;
      this->core->addKernel(filterId, "calculate_dCost_dFilters", filterGlobalWS, 0, filterLocalWS);
      this->core->template addArgument<T>(filterId, "cnn_grads");
      this->core->template addArgument<T>(filterId, "cnn_actvs");
      this->core->template addArgument<T>(filterId, "cnn_dFilters");
      this->core->template addArgument<ulong>(filterId, gradOutOffset);
      this->core->template addArgument<ulong>(filterId, actvInOffset);
      this->core->template addArgument<ulong>(filterId, this->convInfos[convIdx].filterOffset);
      this->core->template addArgument<ulong>(filterId, inShape.c);
      this->core->template addArgument<ulong>(filterId, inShape.h);
      this->core->template addArgument<ulong>(filterId, inShape.w);
      this->core->template addArgument<ulong>(filterId, conv.numFilters);
      this->core->template addArgument<ulong>(filterId, conv.filterH);
      this->core->template addArgument<ulong>(filterId, conv.filterW);
      this->core->template addArgument<ulong>(filterId, conv.strideY);
      this->core->template addArgument<ulong>(filterId, conv.strideX);
      this->core->template addArgument<ulong>(filterId, padY);
      this->core->template addArgument<ulong>(filterId, padX);
      this->core->template addArgument<ulong>(filterId, outH);
      this->core->template addArgument<ulong>(filterId, outW);

      // calculate_dCost_dBiases — one work-group per filter, tree-reduce output positions
      std::string biasId = "calculate_dCost_dBiases_layer" + layerStr;
      ulong biasLocalWS = 256;
      ulong biasGlobalWS = conv.numFilters * biasLocalWS;
      this->core->addKernel(biasId, "calculate_dCost_dBiases", biasGlobalWS, 0, biasLocalWS);
      this->core->template addArgument<T>(biasId, "cnn_grads");
      this->core->template addArgument<T>(biasId, "cnn_dBiases");
      this->core->template addArgument<ulong>(biasId, gradOutOffset);
      this->core->template addArgument<ulong>(biasId, this->convInfos[convIdx].biasOffset);
      this->core->template addArgument<ulong>(biasId, conv.numFilters);
      this->core->template addArgument<ulong>(biasId, outH);
      this->core->template addArgument<ulong>(biasId, outW);

      // calculate_dCost_dInput (skip if first layer — no one reads the gradient)
      if (i > 0) {
        ulong nInputElems = inShape.size();
        std::string inputId = "calculate_dCost_dInput_layer" + layerStr;
        this->core->addKernel(inputId, "calculate_dCost_dInput", nInputElems, 0);
        this->core->template addArgument<T>(inputId, "cnn_grads");
        this->core->template addArgument<T>(inputId, "cnn_filters");
        this->core->template addArgument<ulong>(inputId, gradOutOffset);
        this->core->template addArgument<ulong>(inputId, gradInOffset);
        this->core->template addArgument<ulong>(inputId, this->convInfos[convIdx].filterOffset);
        this->core->template addArgument<ulong>(inputId, inShape.c);
        this->core->template addArgument<ulong>(inputId, inShape.h);
        this->core->template addArgument<ulong>(inputId, inShape.w);
        this->core->template addArgument<ulong>(inputId, conv.numFilters);
        this->core->template addArgument<ulong>(inputId, conv.filterH);
        this->core->template addArgument<ulong>(inputId, conv.filterW);
        this->core->template addArgument<ulong>(inputId, conv.strideY);
        this->core->template addArgument<ulong>(inputId, conv.strideX);
        this->core->template addArgument<ulong>(inputId, padY);
        this->core->template addArgument<ulong>(inputId, padX);
        this->core->template addArgument<ulong>(inputId, outH);
        this->core->template addArgument<ulong>(inputId, outW);
      }

      break;
    }

    case LayerType::RELU: {
      ulong size = inShape.size();
      std::string kernelId = "calculate_dCost_dRelu_layer" + layerStr;
      this->core->addKernel(kernelId, "calculate_dCost_dRelu", size, 0);
      this->core->template addArgument<T>(kernelId, "cnn_grads");
      this->core->template addArgument<T>(kernelId, "cnn_actvs");
      this->core->template addArgument<ulong>(kernelId, gradInOffset);
      this->core->template addArgument<ulong>(kernelId, gradOutOffset);
      this->core->template addArgument<ulong>(kernelId, actvInOffset);
      this->core->template addArgument<ulong>(kernelId, size);
      break;
    }

    case LayerType::POOL: {
      poolIdx--;

      // Zero the input gradient region first
      ulong inSize = inShape.size();
      std::string zeroId = "zero_pool_grad_layer" + layerStr;
      this->core->addKernel(zeroId, "zero_buffer", inSize, 0);
      this->core->template addArgument<T>(zeroId, "cnn_grads");
      this->core->template addArgument<ulong>(zeroId, gradInOffset);
      this->core->template addArgument<ulong>(zeroId, inSize);

      // calculate_dCost_dMaxpool
      ulong outSize = outShape.size();
      std::string poolId = "calculate_dCost_dMaxpool_layer" + layerStr;
      this->core->addKernel(poolId, "calculate_dCost_dMaxpool", outSize, 0);
      this->core->template addArgument<T>(poolId, "cnn_grads");
      this->core->template addArgument<ulong>(poolId, "cnn_pool_indices");
      this->core->template addArgument<ulong>(poolId, gradOutOffset);
      this->core->template addArgument<ulong>(poolId, this->poolInfos[poolIdx].indexOffset);
      this->core->template addArgument<ulong>(poolId, outSize);
      break;
    }

    case LayerType::FLATTEN: {
      // No GPU kernel needed - gradient is written to cnn_grads by bridge kernel
      break;
    }
    }
  }
}

//===================================================================================================================//
//-- Kernel building: CNN accumulate gradients --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::addCNNAccumulateKernels()
{
  if (this->totalFilterSize > 0) {
    this->core->addKernel("accumulate_gradients_filters", "accumulate_gradients", this->totalFilterSize, 0);
    this->core->template addArgument<T>("accumulate_gradients_filters", "cnn_accum_dFilters");
    this->core->template addArgument<T>("accumulate_gradients_filters", "cnn_dFilters");
    this->core->template addArgument<ulong>("accumulate_gradients_filters", static_cast<ulong>(0));
    this->core->template addArgument<ulong>("accumulate_gradients_filters", this->totalFilterSize);
  }

  if (this->totalBiasSize > 0) {
    this->core->addKernel("accumulate_gradients_biases", "accumulate_gradients", this->totalBiasSize, 0);
    this->core->template addArgument<T>("accumulate_gradients_biases", "cnn_accum_dBiases");
    this->core->template addArgument<T>("accumulate_gradients_biases", "cnn_dBiases");
    this->core->template addArgument<ulong>("accumulate_gradients_biases", static_cast<ulong>(0));
    this->core->template addArgument<ulong>("accumulate_gradients_biases", this->totalBiasSize);
  }
}

//===================================================================================================================//
//-- Kernel building: CNN update parameters --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::addCNNUpdateKernels(ulong numSamples)
{
  if (this->totalFilterSize > 0) {
    this->core->addKernel("update_parameters_filters", "update_parameters", this->totalFilterSize, 0);
    this->core->template addArgument<T>("update_parameters_filters", "cnn_filters");
    this->core->template addArgument<T>("update_parameters_filters", "cnn_accum_dFilters");
    this->core->template addArgument<ulong>("update_parameters_filters", static_cast<ulong>(0));
    this->core->template addArgument<ulong>("update_parameters_filters", this->totalFilterSize);
    this->core->template addArgument<ulong>("update_parameters_filters", numSamples);
    this->core->template addArgument<float>("update_parameters_filters",
                                            static_cast<float>(this->coreConfig.trainingConfig.learningRate));
  }

  if (this->totalBiasSize > 0) {
    this->core->addKernel("update_parameters_biases", "update_parameters", this->totalBiasSize, 0);
    this->core->template addArgument<T>("update_parameters_biases", "cnn_biases");
    this->core->template addArgument<T>("update_parameters_biases", "cnn_accum_dBiases");
    this->core->template addArgument<ulong>("update_parameters_biases", static_cast<ulong>(0));
    this->core->template addArgument<ulong>("update_parameters_biases", this->totalBiasSize);
    this->core->template addArgument<ulong>("update_parameters_biases", numSamples);
    this->core->template addArgument<float>("update_parameters_biases",
                                            static_cast<float>(this->coreConfig.trainingConfig.learningRate));
  }
}

//===================================================================================================================//
//-- Kernel building: copy bridge (CNN output → ANN input) --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::addCopyBridgeKernels()
{
  ulong lastLayerIdx = this->layerInfos.size() - 1;
  ulong cnnOutputOffset = this->layerInfos[lastLayerIdx].actvOffset;

  this->core->addKernel("copy_cnn_to_ann", this->flattenSize, 0);
  this->core->template addArgument<T>("copy_cnn_to_ann", "cnn_actvs");
  this->core->template addArgument<T>("copy_cnn_to_ann", "actvs");
  this->core->template addArgument<ulong>("copy_cnn_to_ann", cnnOutputOffset);
  this->core->template addArgument<ulong>("copy_cnn_to_ann", this->flattenSize);
}

//===================================================================================================================//
//-- Kernel setup: predict (CNN propagate → bridge → ANN propagate) --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setupPredictKernels()
{
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  this->addPropagateKernels();
  this->addCopyBridgeKernels();
  this->annGPUWorker->addPropagateKernels();

  this->predictKernelsSetup = true;
}

//===================================================================================================================//
//-- Kernel setup: training (full propagate + backpropagate + accumulate pipeline) --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setupTrainingKernels()
{
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  // Propagate pipeline: CNN propagate → copy → ANN propagate
  this->addPropagateKernels();
  this->addCopyBridgeKernels();
  this->annGPUWorker->addPropagateKernels();

  // Backpropagate pipeline: ANN backpropagate (with input gradients) → reverse bridge → CNN backpropagate
  this->annGPUWorker->addBackpropagateKernels(true);

  // Reverse bridge: copy ANN input gradients to CNN gradient buffer
  ulong lastLayerIdx = this->layerInfos.size() - 1;
  ulong cnnOutputOffset = this->layerInfos[lastLayerIdx].actvOffset;
  this->core->addKernel("copy_ann_grad_to_cnn", this->flattenSize, 0);
  this->core->template addArgument<T>("copy_ann_grad_to_cnn", "dCost_dActvs");
  this->core->template addArgument<T>("copy_ann_grad_to_cnn", "cnn_grads");
  this->core->template addArgument<ulong>("copy_ann_grad_to_cnn", cnnOutputOffset);
  this->core->template addArgument<ulong>("copy_ann_grad_to_cnn", this->flattenSize);

  this->addBackpropagateKernels();

  // Accumulate: CNN + ANN
  this->addCNNAccumulateKernels();
  this->annGPUWorker->addAccumulateKernels();

  // Loss: compute weighted MSE on GPU and accumulate into accum_loss buffer
  ulong outputActvOffset = this->annGPUWorker->getOutputActvOffset();
  ulong numOutputNeurons = this->annGPUWorker->getNumOutputNeurons();
  this->core->addKernel("calculate_sample_loss", "calculate_sample_loss", 1, 0);
  this->core->template addArgument<T>("calculate_sample_loss", "actvs");
  this->core->template addArgument<T>("calculate_sample_loss", "outputs");
  this->core->template addArgument<T>("calculate_sample_loss", "lossWeights");
  this->core->template addArgument<T>("calculate_sample_loss", "accum_loss");
  this->core->template addArgument<ulong>("calculate_sample_loss", outputActvOffset);
  this->core->template addArgument<ulong>("calculate_sample_loss", numOutputNeurons);
  this->core->template addArgument<ulong>("calculate_sample_loss",
                                          static_cast<ulong>(this->coreConfig.costFunctionConfig.type));

  this->trainingKernelsSetup = true;
}

//===================================================================================================================//
//-- Kernel setup: update parameters --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setupUpdateKernels(ulong numSamples)
{
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  this->addCNNUpdateKernels(numSamples);
  this->annGPUWorker->addUpdateKernels(numSamples);

  this->updateKernelsSetup = true;
}

//===================================================================================================================//
//-- Helper: invalidate all kernel flags --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::invalidateAllKernelFlags()
{
  this->predictKernelsSetup = false;
  this->trainingKernelsSetup = false;
  this->updateKernelsSetup = false;
}

//===================================================================================================================//
//-- Predict --//
//===================================================================================================================//

template <typename T>
Output<T> CoreGPUWorker<T>::predict(const Input<T>& input)
{
  // Set up predict kernels if needed (CNN propagate → bridge → ANN propagate)
  if (!this->predictKernelsSetup) {
    this->setupPredictKernels();
  }

  // Write input to cnn_actvs at offset 0
  std::vector<T> inputVec(input.data.begin(), input.data.end());
  this->core->template writeBuffer<T>("cnn_actvs", inputVec, 0);

  // Single run: CNN propagate → copy_cnn_to_ann → ANN propagate
  this->core->run();

  // Read ANN output
  ANN::Output<T> annOutput = this->annGPUWorker->readOutput();

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
  this->resetAccumulators();

  // Set up training kernels once (full propagate + backpropagate + accumulate pipeline)
  if (!this->trainingKernelsSetup) {
    this->setupTrainingKernels();
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
    if (this->annGPUWorker->hasDropout)
      this->annGPUWorker->generateAndUploadDropoutMask();

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
  if (!this->trainingKernelsSetup) {
    this->setupTrainingKernels();
  }

  // Write CNN input to GPU
  std::vector<T> inputVec(input.data.begin(), input.data.end());
  this->core->template writeBuffer<T>("cnn_actvs", inputVec, 0);

  // Write ANN expected output to GPU
  std::vector<T> expectedVec(expected.begin(), expected.end());
  this->core->template writeBuffer<T>("outputs", expectedVec, 0);

  // Generate and upload dropout mask for ANN dense layers (different mask per sample)
  if (this->annGPUWorker->hasDropout)
    this->annGPUWorker->generateAndUploadDropoutMask();

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
//-- Step-by-step: reset accumulators --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::resetAccumulators()
{
  T zero = static_cast<T>(0);

  // Reset CNN accumulators using GPU fill (no host-side vector allocation)
  if (this->totalFilterSize > 0) {
    this->core->template fillBuffer<T>("cnn_accum_dFilters", zero, this->totalFilterSize);
  }

  if (this->totalBiasSize > 0) {
    this->core->template fillBuffer<T>("cnn_accum_dBiases", zero, this->totalBiasSize);
  }

  // Reset ANN accumulators
  this->annGPUWorker->resetAccumulators();
}

//===================================================================================================================//
//-- CNN Gradient access (for multi-GPU merging) --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::readAccumulatedGradients(std::vector<T>& accumFilters, std::vector<T>& accumBiases)
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
void CoreGPUWorker<T>::setAccumulators(const std::vector<T>& accumFilters, const std::vector<T>& accumBiases)
{
  if (this->totalFilterSize > 0) {
    this->core->template writeBuffer<T>("cnn_accum_dFilters", accumFilters, 0);
  }

  if (this->totalBiasSize > 0) {
    this->core->template writeBuffer<T>("cnn_accum_dBiases", accumBiases, 0);
  }
}

//===================================================================================================================//
//-- ANN Gradient access (for multi-GPU merging) --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::readANNAccumulatedGradients(ANN::Tensor1D<T>& accumWeights, ANN::Tensor1D<T>& accumBiases)
{
  this->annGPUWorker->readAccumulatedGradients(accumWeights, accumBiases);
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setANNAccumulators(const ANN::Tensor1D<T>& accumWeights, const ANN::Tensor1D<T>& accumBiases)
{
  this->annGPUWorker->setAccumulators(accumWeights, accumBiases);
}

//===================================================================================================================//
//-- Weight update --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::update(ulong numSamples)
{
  this->setupUpdateKernels(numSamples);
  this->core->run();
  this->invalidateAllKernelFlags();
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
  this->trainingKernelsSetup = ready;
  this->updateKernelsSetup = false;
  this->predictKernelsSetup = false;
}

//===================================================================================================================//
//-- Parameter synchronization: GPU -> CPU --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::syncParametersFromGPU()
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

  // Sync ANN dense layer parameters from GPU
  this->annGPUWorker->syncParametersFromGPU();
  this->parameters.denseParams = this->annGPUWorker->getParameters();
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
