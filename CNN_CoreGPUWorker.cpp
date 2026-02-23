#include "CNN_CoreGPUWorker.hpp"
#include "CNN_SlidingStrategy.hpp"
#include "CNN_Conv2D.hpp"
#include "CNN_ReLU.hpp"
#include "CNN_Pool.hpp"
#include "CNN_Flatten.hpp"

#include <ANN_CoreGPUWorker.hpp>
#include <OCLW_Core.hpp>

#include <QDebug>

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
      logLevel(config.logLevel) {

  this->ownedCore = std::make_unique<OpenCLWrapper::Core>(false);
  this->core = this->ownedCore.get();
  this->core->setVerbose(this->logLevel >= CNN::LogLevel::DEBUG);

  // Compute CNN output shape
  this->cnnOutputShape = config.layersConfig.validateShapes(config.inputShape);
  this->flattenSize = this->cnnOutputShape.size();

  // Initialize conv parameters (He initialization if not loaded)
  this->initializeConvParams();

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
      core(&sharedCore) {

  // Compute CNN output shape
  this->cnnOutputShape = config.layersConfig.validateShapes(config.inputShape);
  this->flattenSize = this->cnnOutputShape.size();

  // Initialize conv parameters (He initialization if not loaded)
  this->initializeConvParams();

  // Compute buffer offsets for all layers
  this->computeLayerOffsets();

  // Caller must call loadSources(), buildANNWorker(), allocateBuffers() manually
}

//===================================================================================================================//
//-- Initialization: compute layer offsets --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::computeLayerOffsets() {
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
        continue;  // Skip the normal push_back and offset advancement below
      }
    }

    this->layerInfos.push_back({offset, outShape.size()});
    offset += outShape.size();
    currentShape = outShape;
  }

  this->totalActvSize = offset;
}

//===================================================================================================================//
//-- Initialization: He-init conv parameters --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::initializeConvParams() {
  ulong convIdx = 0;
  Shape3D currentShape = this->coreConfig.inputShape;

  for (const auto& layerConfig : this->coreConfig.layersConfig.cnnLayers) {
    if (layerConfig.type != LayerType::CONV) {
      if (layerConfig.type == LayerType::POOL) {
        const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);
        ulong outH = (currentShape.h - pool.poolH) / pool.strideY + 1;
        ulong outW = (currentShape.w - pool.poolW) / pool.strideX + 1;
        currentShape = {currentShape.c, outH, outW};
      }
      continue;
    }

    const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);

    if (convIdx < this->parameters.convParams.size() &&
        !this->parameters.convParams[convIdx].filters.empty()) {
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
      ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
      currentShape = {conv.numFilters, outH, outW};
      convIdx++;
      continue;
    }

    if (convIdx >= this->parameters.convParams.size()) {
      this->parameters.convParams.resize(convIdx + 1);
    }

    ConvParameters<T>& cp = this->parameters.convParams[convIdx];
    cp.numFilters = conv.numFilters;
    cp.inputC = currentShape.c;
    cp.filterH = conv.filterH;
    cp.filterW = conv.filterW;

    ulong filterSize = cp.numFilters * cp.inputC * cp.filterH * cp.filterW;
    cp.filters.resize(filterSize);
    cp.biases.assign(cp.numFilters, static_cast<T>(0));

    T fanIn = static_cast<T>(cp.inputC * cp.filterH * cp.filterW);
    T stddev = std::sqrt(static_cast<T>(2) / fanIn);

    std::mt19937 gen(42 + convIdx);
    std::normal_distribution<double> dist(0.0, static_cast<double>(stddev));

    for (ulong i = 0; i < filterSize; i++) {
      cp.filters[i] = static_cast<T>(dist(gen));
    }

    ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
    ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
    ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
    ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
    currentShape = {conv.numFilters, outH, outW};
    convIdx++;
  }
}

//===================================================================================================================//
//-- Initialization: load OpenCL sources --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::loadSources(bool skipDefines) {
  if (this->logLevel >= CNN::LogLevel::INFO) std::cout << "Loading CNN OpenCL kernels...\n";

  // Resolve .cl file paths relative to the source file's directory (via __FILE__),
  // so the kernels are found regardless of the current working directory.
  std::string srcFile = __FILE__;
  std::string srcDir = srcFile.substr(0, srcFile.find_last_of("/\\") + 1);

  if (!skipDefines) {
    this->core->addSourceFile(srcDir + "opencl/CNN_Defines.hpp.cl");
  }

  this->core->addSourceFile(srcDir + "opencl/CNN_Kernels.cpp.cl");

  if (this->logLevel >= CNN::LogLevel::INFO) std::cout << "CNN OpenCL kernels loaded.\n";
}

//===================================================================================================================//
//-- Initialization: allocate GPU buffers --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::allocateBuffers() {
  if (this->logLevel >= CNN::LogLevel::INFO) std::cout << "Allocating CNN GPU buffers...\n";

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

  if (this->logLevel >= CNN::LogLevel::INFO) std::cout << "CNN GPU buffers allocated.\n";
}

//===================================================================================================================//
//-- Initialization: build ANN GPU worker --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::buildANNWorker() {
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
  annTrainingConfig.numThreads = 1;

  // Loss function config
  ANN::LossFunctionConfig<T> annLossFunctionConfig;
  annLossFunctionConfig.type = static_cast<ANN::LossFunctionType>(this->coreConfig.lossFunctionConfig.type);
  annLossFunctionConfig.weights = this->coreConfig.lossFunctionConfig.weights;

  // Create ANN GPU worker on the shared core
  this->annGPUWorker = std::make_unique<ANN::CoreGPUWorker<T>>(
      annLayers, annTrainingConfig, this->coreConfig.parameters.denseParams, annLossFunctionConfig,
      *this->core, this->coreConfig.progressReports, static_cast<ANN::LogLevel>(this->coreConfig.logLevel));

  // Load ANN sources (skip defines — CNN_Defines.hpp.cl already defined TYPE, ActvFuncType, Layer)
  this->annGPUWorker->loadSources(true);

  // Allocate ANN GPU buffers
  this->annGPUWorker->allocateBuffers();
}



//===================================================================================================================//
//-- Kernel building: forward pass --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::addForwardKernels() {
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

        std::string kernelId = "conv2d_forward_layer" + layerStr;
        this->core->addKernel(kernelId, "conv2d_forward", nElements, 0);
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
        std::string kernelId = "relu_forward_layer" + layerStr;
        this->core->addKernel(kernelId, "relu_forward", size, 0);
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

        std::string kernelId = "maxpool_forward_layer" + layerStr;
        this->core->addKernel(kernelId, "maxpool_forward", nElements, 0);
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
//-- Kernel building: backward pass --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::addBackwardKernels() {
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  ulong numLayers = cnnLayers.size();

  // Precompute shapes for each layer (forward direction)
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

  // Iterate backward through layers
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

        // conv2d_backward_filters
        ulong nFilterElems = this->convInfos[convIdx].numFilterElems;
        std::string filterId = "conv2d_backward_filters_layer" + layerStr;
        this->core->addKernel(filterId, "conv2d_backward_filters", nFilterElems, 0);
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

        // conv2d_backward_biases
        std::string biasId = "conv2d_backward_biases_layer" + layerStr;
        this->core->addKernel(biasId, "conv2d_backward_biases", conv.numFilters, 0);
        this->core->template addArgument<T>(biasId, "cnn_grads");
        this->core->template addArgument<T>(biasId, "cnn_dBiases");
        this->core->template addArgument<ulong>(biasId, gradOutOffset);
        this->core->template addArgument<ulong>(biasId, this->convInfos[convIdx].biasOffset);
        this->core->template addArgument<ulong>(biasId, conv.numFilters);
        this->core->template addArgument<ulong>(biasId, outH);
        this->core->template addArgument<ulong>(biasId, outW);

        // conv2d_backward_input (skip if first layer — no one reads the gradient)
        if (i > 0) {
          ulong nInputElems = inShape.size();
          std::string inputId = "conv2d_backward_input_layer" + layerStr;
          this->core->addKernel(inputId, "conv2d_backward_input", nInputElems, 0);
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
        std::string kernelId = "relu_backward_layer" + layerStr;
        this->core->addKernel(kernelId, "relu_backward", size, 0);
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

        // maxpool_backward
        ulong outSize = outShape.size();
        std::string poolId = "maxpool_backward_layer" + layerStr;
        this->core->addKernel(poolId, "maxpool_backward", outSize, 0);
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
void CoreGPUWorker<T>::addCNNAccumulateKernels() {
  if (this->totalFilterSize > 0) {
    this->core->addKernel("cnn_accumulate_filters", "cnn_accumulate", this->totalFilterSize, 0);
    this->core->template addArgument<T>("cnn_accumulate_filters", "cnn_accum_dFilters");
    this->core->template addArgument<T>("cnn_accumulate_filters", "cnn_dFilters");
    this->core->template addArgument<ulong>("cnn_accumulate_filters", static_cast<ulong>(0));
    this->core->template addArgument<ulong>("cnn_accumulate_filters", this->totalFilterSize);
  }

  if (this->totalBiasSize > 0) {
    this->core->addKernel("cnn_accumulate_biases", "cnn_accumulate", this->totalBiasSize, 0);
    this->core->template addArgument<T>("cnn_accumulate_biases", "cnn_accum_dBiases");
    this->core->template addArgument<T>("cnn_accumulate_biases", "cnn_dBiases");
    this->core->template addArgument<ulong>("cnn_accumulate_biases", static_cast<ulong>(0));
    this->core->template addArgument<ulong>("cnn_accumulate_biases", this->totalBiasSize);
  }
}

//===================================================================================================================//
//-- Kernel building: CNN update parameters --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::addCNNUpdateKernels(ulong numSamples) {
  if (this->totalFilterSize > 0) {
    this->core->addKernel("cnn_update_filters", "cnn_update", this->totalFilterSize, 0);
    this->core->template addArgument<T>("cnn_update_filters", "cnn_filters");
    this->core->template addArgument<T>("cnn_update_filters", "cnn_accum_dFilters");
    this->core->template addArgument<ulong>("cnn_update_filters", static_cast<ulong>(0));
    this->core->template addArgument<ulong>("cnn_update_filters", this->totalFilterSize);
    this->core->template addArgument<ulong>("cnn_update_filters", numSamples);
    this->core->template addArgument<float>("cnn_update_filters",
        static_cast<float>(this->coreConfig.trainingConfig.learningRate));
  }

  if (this->totalBiasSize > 0) {
    this->core->addKernel("cnn_update_biases", "cnn_update", this->totalBiasSize, 0);
    this->core->template addArgument<T>("cnn_update_biases", "cnn_biases");
    this->core->template addArgument<T>("cnn_update_biases", "cnn_accum_dBiases");
    this->core->template addArgument<ulong>("cnn_update_biases", static_cast<ulong>(0));
    this->core->template addArgument<ulong>("cnn_update_biases", this->totalBiasSize);
    this->core->template addArgument<ulong>("cnn_update_biases", numSamples);
    this->core->template addArgument<float>("cnn_update_biases",
        static_cast<float>(this->coreConfig.trainingConfig.learningRate));
  }
}

//===================================================================================================================//
//-- Kernel building: copy bridge (CNN output → ANN input) --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::addCopyBridgeKernels() {
  ulong lastLayerIdx = this->layerInfos.size() - 1;
  ulong cnnOutputOffset = this->layerInfos[lastLayerIdx].actvOffset;

  this->core->addKernel("copy_cnn_to_ann", this->flattenSize, 0);
  this->core->template addArgument<T>("copy_cnn_to_ann", "cnn_actvs");
  this->core->template addArgument<T>("copy_cnn_to_ann", "actvs");
  this->core->template addArgument<ulong>("copy_cnn_to_ann", cnnOutputOffset);
  this->core->template addArgument<ulong>("copy_cnn_to_ann", this->flattenSize);
}

//===================================================================================================================//
//-- Kernel setup: predict (CNN forward → bridge → ANN forward) --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setupPredictKernels() {
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  this->addForwardKernels();
  this->addCopyBridgeKernels();
  this->annGPUWorker->addPropagateKernels();

  this->predictKernelsSetup = true;
}

//===================================================================================================================//
//-- Kernel setup: training (full forward + backward + accumulate pipeline) --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setupTrainingKernels() {
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  // Forward pipeline: CNN forward → copy → ANN forward
  this->addForwardKernels();
  this->addCopyBridgeKernels();
  this->annGPUWorker->addPropagateKernels();

  // Backward pipeline: ANN backward (with input gradients) → reverse bridge → CNN backward
  this->annGPUWorker->addBackpropagateKernels(true);

  // Reverse bridge: copy ANN input gradients to CNN gradient buffer
  ulong lastLayerIdx = this->layerInfos.size() - 1;
  ulong cnnOutputOffset = this->layerInfos[lastLayerIdx].actvOffset;
  this->core->addKernel("copy_ann_grad_to_cnn", this->flattenSize, 0);
  this->core->template addArgument<T>("copy_ann_grad_to_cnn", "dCost_dActvs");
  this->core->template addArgument<T>("copy_ann_grad_to_cnn", "cnn_grads");
  this->core->template addArgument<ulong>("copy_ann_grad_to_cnn", cnnOutputOffset);
  this->core->template addArgument<ulong>("copy_ann_grad_to_cnn", this->flattenSize);

  this->addBackwardKernels();

  // Accumulate: CNN + ANN
  this->addCNNAccumulateKernels();
  this->annGPUWorker->addAccumulateKernels();

  this->trainingKernelsSetup = true;
}

//===================================================================================================================//
//-- Kernel setup: update parameters --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setupUpdateKernels(ulong numSamples) {
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
void CoreGPUWorker<T>::invalidateAllKernelFlags() {
  this->predictKernelsSetup = false;
  this->trainingKernelsSetup = false;
  this->updateKernelsSetup = false;
}


//===================================================================================================================//
//-- Predict --//
//===================================================================================================================//

template <typename T>
Output<T> CoreGPUWorker<T>::predict(const Input<T>& input) {
  // Set up predict kernels if needed (CNN forward → bridge → ANN forward)
  if (!this->predictKernelsSetup) {
    this->setupPredictKernels();
  }

  // Write input to cnn_actvs at offset 0
  std::vector<T> inputVec(input.data.begin(), input.data.end());
  this->core->template writeBuffer<T>("cnn_actvs", inputVec, 0);

  // Single run: CNN forward → copy_cnn_to_ann → ANN forward
  this->core->run();

  // Read ANN output
  ANN::Output<T> annOutput = this->annGPUWorker->readOutput();

  return Output<T>(annOutput.begin(), annOutput.end());
}

//===================================================================================================================//
//-- Training --//
//===================================================================================================================//

template <typename T>
T CoreGPUWorker<T>::trainSubset(const Samples<T>& samples, ulong startIdx, ulong endIdx,
                                ulong epoch, ulong totalEpochs, const TrainingCallback<T>& callback) {
  ulong numSamplesInSubset = endIdx - startIdx;
  ulong totalSamples = samples.size();

  T subsetLoss = static_cast<T>(0);

  // Reset CNN and ANN accumulators
  this->resetAccumulators();

  // Set up training kernels once (full forward + backward + accumulate pipeline)
  if (!this->trainingKernelsSetup) {
    this->setupTrainingKernels();
  }

  ulong lastANNLayerNeurons = this->annGPUWorker->getParameters().biases.back().size();

  for (ulong s = startIdx; s < endIdx; s++) {
    const Sample<T>& sample = samples[s];

    // Write CNN input to GPU
    std::vector<T> inputVec(sample.input.data.begin(), sample.input.data.end());
    this->core->template writeBuffer<T>("cnn_actvs", inputVec, 0);

    // Write ANN expected output to GPU (for loss computation in backward pass)
    std::vector<T> expectedVec(sample.output.begin(), sample.output.end());
    this->core->template writeBuffer<T>("outputs", expectedVec, 0);

    // Single run: CNN forward → bridge → ANN forward → ANN backward → reverse bridge → CNN backward → accumulate
    this->core->run();

    // Read ANN output for loss calculation
    ANN::Output<T> annOutput = this->annGPUWorker->readOutput();
    Output<T> predicted(annOutput.begin(), annOutput.end());
    T sampleLoss = this->calculateLoss(predicted, sample.output);
    subsetLoss += sampleLoss;

    // Report progress
    if (callback) {
      TrainingProgress<T> progress;
      progress.currentEpoch = epoch;
      progress.totalEpochs = totalEpochs;
      progress.currentSample = s + 1;
      progress.totalSamples = totalSamples;
      progress.sampleLoss = sampleLoss;
      progress.epochLoss = static_cast<T>(0);
      callback(progress);
    }
  }

  return subsetLoss;
}

//===================================================================================================================//
//-- Testing --//
//===================================================================================================================//

template <typename T>
T CoreGPUWorker<T>::testSubset(const Samples<T>& samples, ulong startIdx, ulong endIdx) {
  T subsetLoss = static_cast<T>(0);

  for (ulong s = startIdx; s < endIdx; s++) {
    Output<T> predicted = this->predict(samples[s].input);
    subsetLoss += this->calculateLoss(predicted, samples[s].output);
  }

  return subsetLoss;
}


//===================================================================================================================//
//-- Step-by-step: backpropagate a single sample (for external orchestration) --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::backpropagateSample(const Input<T>& input, const Output<T>& expected) {
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

  // Single run: full forward + backward + accumulate
  this->core->run();
}

//===================================================================================================================//
//-- Step-by-step: accumulate (no-op — accumulation is baked into training pipeline) --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::accumulate() {
  // No-op: accumulation is part of the training kernel pipeline
}

//===================================================================================================================//
//-- Step-by-step: reset accumulators --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::resetAccumulators() {
  // Reset CNN accumulators
  if (this->totalFilterSize > 0) {
    std::vector<T> zeros(this->totalFilterSize, static_cast<T>(0));
    this->core->template writeBuffer<T>("cnn_accum_dFilters", zeros, 0);
  }

  if (this->totalBiasSize > 0) {
    std::vector<T> zeros(this->totalBiasSize, static_cast<T>(0));
    this->core->template writeBuffer<T>("cnn_accum_dBiases", zeros, 0);
  }

  // Reset ANN accumulators
  this->annGPUWorker->resetAccumulators();
}

//===================================================================================================================//
//-- CNN Gradient access (for multi-GPU merging) --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::readAccumulatedGradients(std::vector<T>& accumFilters, std::vector<T>& accumBiases) {
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
void CoreGPUWorker<T>::setAccumulators(const std::vector<T>& accumFilters, const std::vector<T>& accumBiases) {
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
void CoreGPUWorker<T>::readANNAccumulatedGradients(ANN::Tensor1D<T>& accumWeights, ANN::Tensor1D<T>& accumBiases) {
  this->annGPUWorker->readAccumulatedGradients(accumWeights, accumBiases);
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setANNAccumulators(const ANN::Tensor1D<T>& accumWeights, const ANN::Tensor1D<T>& accumBiases) {
  this->annGPUWorker->setAccumulators(accumWeights, accumBiases);
}

//===================================================================================================================//
//-- Weight update --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::update(ulong numSamples) {
  this->setupUpdateKernels(numSamples);
  this->core->run();
  this->invalidateAllKernelFlags();
}

//===================================================================================================================//
//-- Parameter synchronization: GPU -> CPU --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::syncParametersFromGPU() {
  // Read CNN filters from GPU and scatter back to per-layer parameters
  if (this->totalFilterSize > 0) {
    std::vector<T> flatFilters(this->totalFilterSize);
    this->core->template readBuffer<T>("cnn_filters", flatFilters, 0);

    for (ulong i = 0; i < this->convInfos.size(); i++) {
      ulong offset = this->convInfos[i].filterOffset;
      ulong count = this->convInfos[i].numFilterElems;
      this->parameters.convParams[i].filters.assign(
          flatFilters.begin() + static_cast<long>(offset),
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
      this->parameters.convParams[i].biases.assign(
          flatBiases.begin() + static_cast<long>(offset),
          flatBiases.begin() + static_cast<long>(offset + count));
    }
  }

  // Sync ANN dense layer parameters from GPU
  this->annGPUWorker->syncParametersFromGPU();
  this->parameters.denseParams = this->annGPUWorker->getParameters();
}

//===================================================================================================================//
//-- Loss calculation (MSE) --//
//===================================================================================================================//

template <typename T>
T CoreGPUWorker<T>::calculateLoss(const Output<T>& predicted, const Output<T>& expected) {
  T loss = static_cast<T>(0);

  for (ulong i = 0; i < expected.size(); i++) {
    T diff = predicted[i] - expected[i];
    T weight = (!this->coreConfig.lossFunctionConfig.weights.empty()) ? this->coreConfig.lossFunctionConfig.weights[i] : static_cast<T>(1);
    loss += weight * diff * diff;
  }

  return loss / static_cast<T>(expected.size());
}

//===================================================================================================================//
//-- Explicit template instantiations --//
//===================================================================================================================//

template class CNN::CoreGPUWorker<int>;
template class CNN::CoreGPUWorker<double>;
template class CNN::CoreGPUWorker<float>;
