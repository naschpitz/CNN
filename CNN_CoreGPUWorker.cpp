#include "CNN_CoreGPUWorker.hpp"
#include "CNN_SlidingStrategy.hpp"
#include "CNN_Conv2D.hpp"
#include "CNN_ReLU.hpp"
#include "CNN_Pool.hpp"
#include "CNN_Flatten.hpp"

#include <ANN_Core.hpp>
#include <ANN_ActvFunc.hpp>
#include <OCLW_Core.hpp>

#include <QDebug>

#include <cmath>
#include <iostream>
#include <random>

using namespace CNN;

//===================================================================================================================//
//-- Constructor --//
//===================================================================================================================//

template <typename T>
CoreGPUWorker<T>::CoreGPUWorker(const CoreConfig<T>& config)
    : coreConfig(config),
      parameters(config.parameters),
      verbose(config.verbose),
      oclwCore(OpenCLWrapper::Core(false)) {

  this->oclwCore.setVerbose(this->verbose);

  // Compute CNN output shape
  this->cnnOutputShape = config.layersConfig.validateShapes(config.inputShape);
  this->flattenSize = this->cnnOutputShape.size();

  // Initialize conv parameters (He initialization if not loaded)
  this->initializeConvParams();

  // Compute buffer offsets for all layers
  this->computeLayerOffsets();

  // Allocate GPU buffers and write initial parameters
  this->allocateBuffers();

  // Build ANN core for dense layers (CPU mode)
  this->buildANNCore();
}

//===================================================================================================================//
//-- Initialization: compute layer offsets --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::computeLayerOffsets() {
  const auto& cnnLayers = this->coreConfig.layersConfig.cnnLayers;
  Shape3D currentShape = this->coreConfig.inputShape;

  // First entry: input tensor
  layerInfos.clear();
  layerInfos.push_back({0, currentShape.size()});
  ulong offset = currentShape.size();

  ulong convIdx = 0;
  ulong poolIdx = 0;
  totalFilterSize = 0;
  totalBiasSize = 0;
  totalPoolIndexSize = 0;
  convInfos.clear();
  poolInfos.clear();

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
        ci.filterOffset = totalFilterSize;
        ci.biasOffset = totalBiasSize;
        ci.numFilterElems = conv.numFilters * currentShape.c * conv.filterH * conv.filterW;
        ci.numBiases = conv.numFilters;
        convInfos.push_back(ci);

        totalFilterSize += ci.numFilterElems;
        totalBiasSize += ci.numBiases;
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
        pi.indexOffset = totalPoolIndexSize;
        pi.indexSize = outShape.size();
        poolInfos.push_back(pi);

        totalPoolIndexSize += pi.indexSize;
        poolIdx++;
        break;
      }
      case LayerType::FLATTEN: {
        // Flatten shares the same buffer as its input — no GPU kernel copies data,
        // so we point to the same offset as the previous layer's output.
        layerInfos.push_back({layerInfos.back().actvOffset, outShape.size()});
        currentShape = outShape;
        continue;  // Skip the normal push_back and offset advancement below
      }
    }

    layerInfos.push_back({offset, outShape.size()});
    offset += outShape.size();
    currentShape = outShape;
  }

  totalActvSize = offset;
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
//-- Initialization: allocate GPU buffers --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::allocateBuffers() {
  if (this->verbose) std::cout << "Loading CNN OpenCL kernels...\n";

  this->oclwCore.addSourceFile("opencl/CNN_Defines.hpp.cl");
  this->oclwCore.addSourceFile("opencl/CNN_Kernels.cpp.cl");

  if (this->verbose) std::cout << "CNN OpenCL kernels loaded.\n";

  if (this->verbose) std::cout << "Allocating CNN GPU buffers...\n";

  // Activation and gradient buffers (same layout)
  this->oclwCore. template allocateBuffer<T>("cnn_actvs", this->totalActvSize);
  this->oclwCore. template allocateBuffer<T>("cnn_grads", this->totalActvSize);

  // Filter and bias parameter buffers
  if (this->totalFilterSize > 0) {
    this->oclwCore. template allocateBuffer<T>("cnn_filters", this->totalFilterSize);
    this->oclwCore. template allocateBuffer<T>("cnn_dFilters", this->totalFilterSize);
    this->oclwCore. template allocateBuffer<T>("cnn_accum_dFilters", this->totalFilterSize);
  }

  if (this->totalBiasSize > 0) {
    this->oclwCore. template allocateBuffer<T>("cnn_biases", this->totalBiasSize);
    this->oclwCore. template allocateBuffer<T>("cnn_dBiases", this->totalBiasSize);
    this->oclwCore. template allocateBuffer<T>("cnn_accum_dBiases", this->totalBiasSize);
  }

  // Pool index buffer
  if (this->totalPoolIndexSize > 0) {
    this->oclwCore. template allocateBuffer<ulong>("cnn_pool_indices", this->totalPoolIndexSize);
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

    this->oclwCore. template writeBuffer<T>("cnn_filters", flatFilters, 0);
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

    this->oclwCore. template writeBuffer<T>("cnn_biases", flatBiases, 0);
  }

  if (this->verbose) std::cout << "CNN GPU buffers allocated.\n";
}

//===================================================================================================================//
//-- Initialization: build ANN configuration --//
//===================================================================================================================//

template <typename T>
ANN::CoreConfig<T> CoreGPUWorker<T>::buildANNConfig() {
  ANN::CoreConfig<T> annConfig;

  // Map CNN mode to ANN mode
  switch (this->coreConfig.modeType) {
    case ModeType::TRAIN:
      annConfig.modeType = ANN::ModeType::TRAIN;
      break;
    case ModeType::TEST:
      annConfig.modeType = ANN::ModeType::TEST;
      break;
    default:
      annConfig.modeType = ANN::ModeType::PREDICT;
      break;
  }

  // Dense layers run on CPU for step-by-step orchestration
  annConfig.deviceType = ANN::DeviceType::CPU;

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

  annConfig.layersConfig = annLayers;

  // Training config
  annConfig.trainingConfig.numEpochs = this->coreConfig.trainingConfig.numEpochs;
  annConfig.trainingConfig.learningRate = this->coreConfig.trainingConfig.learningRate;
  annConfig.trainingConfig.numThreads = 1;

  // Dense parameters (if loaded from file)
  annConfig.parameters = this->coreConfig.parameters.denseParams;
  annConfig.verbose = this->coreConfig.verbose;

  return annConfig;
}

//===================================================================================================================//
//-- Initialization: build ANN core --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::buildANNCore() {
  ANN::CoreConfig<T> annConfig = this->buildANNConfig();
  this->annCore = ANN::Core<T>::makeCore(annConfig);
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
        this->oclwCore.addKernel(kernelId, "conv2d_forward", nElements, 0);
        this->oclwCore. template addArgument<T>(kernelId, "cnn_actvs");
        this->oclwCore. template addArgument<T>(kernelId, "cnn_filters");
        this->oclwCore. template addArgument<T>(kernelId, "cnn_biases");
        this->oclwCore. template addArgument<ulong>(kernelId, inOffset);
        this->oclwCore. template addArgument<ulong>(kernelId, outOffset);
        this->oclwCore. template addArgument<ulong>(kernelId, this->convInfos[convIdx].filterOffset);
        this->oclwCore. template addArgument<ulong>(kernelId, this->convInfos[convIdx].biasOffset);
        this->oclwCore. template addArgument<ulong>(kernelId, currentShape.c);
        this->oclwCore. template addArgument<ulong>(kernelId, currentShape.h);
        this->oclwCore. template addArgument<ulong>(kernelId, currentShape.w);
        this->oclwCore. template addArgument<ulong>(kernelId, conv.numFilters);
        this->oclwCore. template addArgument<ulong>(kernelId, conv.filterH);
        this->oclwCore. template addArgument<ulong>(kernelId, conv.filterW);
        this->oclwCore. template addArgument<ulong>(kernelId, conv.strideY);
        this->oclwCore. template addArgument<ulong>(kernelId, conv.strideX);
        this->oclwCore. template addArgument<ulong>(kernelId, padY);
        this->oclwCore. template addArgument<ulong>(kernelId, padX);
        this->oclwCore. template addArgument<ulong>(kernelId, outH);
        this->oclwCore. template addArgument<ulong>(kernelId, outW);

        currentShape = {conv.numFilters, outH, outW};
        convIdx++;
        break;
      }
      case LayerType::RELU: {
        ulong size = currentShape.size();
        std::string kernelId = "relu_forward_layer" + layerStr;
        this->oclwCore.addKernel(kernelId, "relu_forward", size, 0);
        this->oclwCore. template addArgument<T>(kernelId, "cnn_actvs");
        this->oclwCore. template addArgument<ulong>(kernelId, inOffset);
        this->oclwCore. template addArgument<ulong>(kernelId, outOffset);
        this->oclwCore. template addArgument<ulong>(kernelId, size);
        break;
      }
      case LayerType::POOL: {
        const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);
        ulong outH = (currentShape.h - pool.poolH) / pool.strideY + 1;
        ulong outW = (currentShape.w - pool.poolW) / pool.strideX + 1;
        ulong nElements = currentShape.c * outH * outW;

        std::string kernelId = "maxpool_forward_layer" + layerStr;
        this->oclwCore.addKernel(kernelId, "maxpool_forward", nElements, 0);
        this->oclwCore. template addArgument<T>(kernelId, "cnn_actvs");
        this->oclwCore. template addArgument<ulong>(kernelId, "cnn_pool_indices");
        this->oclwCore. template addArgument<ulong>(kernelId, inOffset);
        this->oclwCore. template addArgument<ulong>(kernelId, outOffset);
        this->oclwCore. template addArgument<ulong>(kernelId, this->poolInfos[poolIdx].indexOffset);
        this->oclwCore. template addArgument<ulong>(kernelId, currentShape.c);
        this->oclwCore. template addArgument<ulong>(kernelId, currentShape.h);
        this->oclwCore. template addArgument<ulong>(kernelId, currentShape.w);
        this->oclwCore. template addArgument<ulong>(kernelId, pool.poolH);
        this->oclwCore. template addArgument<ulong>(kernelId, pool.poolW);
        this->oclwCore. template addArgument<ulong>(kernelId, pool.strideY);
        this->oclwCore. template addArgument<ulong>(kernelId, pool.strideX);
        this->oclwCore. template addArgument<ulong>(kernelId, outH);
        this->oclwCore. template addArgument<ulong>(kernelId, outW);

        currentShape = {currentShape.c, outH, outW};
        poolIdx++;
        break;
      }
      case LayerType::FLATTEN: {
        // No GPU kernel needed for flatten - data is read back to CPU
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
        this->oclwCore.addKernel(filterId, "conv2d_backward_filters", nFilterElems, 0);
        this->oclwCore. template addArgument<T>(filterId, "cnn_grads");
        this->oclwCore. template addArgument<T>(filterId, "cnn_actvs");
        this->oclwCore. template addArgument<T>(filterId, "cnn_dFilters");
        this->oclwCore. template addArgument<ulong>(filterId, gradOutOffset);
        this->oclwCore. template addArgument<ulong>(filterId, actvInOffset);
        this->oclwCore. template addArgument<ulong>(filterId, this->convInfos[convIdx].filterOffset);
        this->oclwCore. template addArgument<ulong>(filterId, inShape.c);
        this->oclwCore. template addArgument<ulong>(filterId, inShape.h);
        this->oclwCore. template addArgument<ulong>(filterId, inShape.w);
        this->oclwCore. template addArgument<ulong>(filterId, conv.numFilters);
        this->oclwCore. template addArgument<ulong>(filterId, conv.filterH);
        this->oclwCore. template addArgument<ulong>(filterId, conv.filterW);
        this->oclwCore. template addArgument<ulong>(filterId, conv.strideY);
        this->oclwCore. template addArgument<ulong>(filterId, conv.strideX);
        this->oclwCore. template addArgument<ulong>(filterId, padY);
        this->oclwCore. template addArgument<ulong>(filterId, padX);
        this->oclwCore. template addArgument<ulong>(filterId, outH);
        this->oclwCore. template addArgument<ulong>(filterId, outW);

        // conv2d_backward_biases
        std::string biasId = "conv2d_backward_biases_layer" + layerStr;
        this->oclwCore.addKernel(biasId, "conv2d_backward_biases", conv.numFilters, 0);
        this->oclwCore. template addArgument<T>(biasId, "cnn_grads");
        this->oclwCore. template addArgument<T>(biasId, "cnn_dBiases");
        this->oclwCore. template addArgument<ulong>(biasId, gradOutOffset);
        this->oclwCore. template addArgument<ulong>(biasId, this->convInfos[convIdx].biasOffset);
        this->oclwCore. template addArgument<ulong>(biasId, conv.numFilters);
        this->oclwCore. template addArgument<ulong>(biasId, outH);
        this->oclwCore. template addArgument<ulong>(biasId, outW);

        // conv2d_backward_input (skip if first layer — no one reads the gradient)
        if (i > 0) {
          ulong nInputElems = inShape.size();
          std::string inputId = "conv2d_backward_input_layer" + layerStr;
          this->oclwCore.addKernel(inputId, "conv2d_backward_input", nInputElems, 0);
          this->oclwCore. template addArgument<T>(inputId, "cnn_grads");
          this->oclwCore. template addArgument<T>(inputId, "cnn_filters");
          this->oclwCore. template addArgument<ulong>(inputId, gradOutOffset);
          this->oclwCore. template addArgument<ulong>(inputId, gradInOffset);
          this->oclwCore. template addArgument<ulong>(inputId, this->convInfos[convIdx].filterOffset);
          this->oclwCore. template addArgument<ulong>(inputId, inShape.c);
          this->oclwCore. template addArgument<ulong>(inputId, inShape.h);
          this->oclwCore. template addArgument<ulong>(inputId, inShape.w);
          this->oclwCore. template addArgument<ulong>(inputId, conv.numFilters);
          this->oclwCore. template addArgument<ulong>(inputId, conv.filterH);
          this->oclwCore. template addArgument<ulong>(inputId, conv.filterW);
          this->oclwCore. template addArgument<ulong>(inputId, conv.strideY);
          this->oclwCore. template addArgument<ulong>(inputId, conv.strideX);
          this->oclwCore. template addArgument<ulong>(inputId, padY);
          this->oclwCore. template addArgument<ulong>(inputId, padX);
          this->oclwCore. template addArgument<ulong>(inputId, outH);
          this->oclwCore. template addArgument<ulong>(inputId, outW);
        }
        break;
      }
      case LayerType::RELU: {
        ulong size = inShape.size();
        std::string kernelId = "relu_backward_layer" + layerStr;
        this->oclwCore.addKernel(kernelId, "relu_backward", size, 0);
        this->oclwCore. template addArgument<T>(kernelId, "cnn_grads");
        this->oclwCore. template addArgument<T>(kernelId, "cnn_actvs");
        this->oclwCore. template addArgument<ulong>(kernelId, gradInOffset);
        this->oclwCore. template addArgument<ulong>(kernelId, gradOutOffset);
        this->oclwCore. template addArgument<ulong>(kernelId, actvInOffset);
        this->oclwCore. template addArgument<ulong>(kernelId, size);
        break;
      }
      case LayerType::POOL: {
        poolIdx--;

        // Zero the input gradient region first
        ulong inSize = inShape.size();
        std::string zeroId = "zero_pool_grad_layer" + layerStr;
        this->oclwCore.addKernel(zeroId, "zero_buffer", inSize, 0);
        this->oclwCore. template addArgument<T>(zeroId, "cnn_grads");
        this->oclwCore. template addArgument<ulong>(zeroId, gradInOffset);
        this->oclwCore. template addArgument<ulong>(zeroId, inSize);

        // maxpool_backward
        ulong outSize = outShape.size();
        std::string poolId = "maxpool_backward_layer" + layerStr;
        this->oclwCore.addKernel(poolId, "maxpool_backward", outSize, 0);
        this->oclwCore. template addArgument<T>(poolId, "cnn_grads");
        this->oclwCore. template addArgument<ulong>(poolId, "cnn_pool_indices");
        this->oclwCore. template addArgument<ulong>(poolId, gradOutOffset);
        this->oclwCore. template addArgument<ulong>(poolId, this->poolInfos[poolIdx].indexOffset);
        this->oclwCore. template addArgument<ulong>(poolId, outSize);
        break;
      }
      case LayerType::FLATTEN: {
        // No GPU kernel needed - gradient is written to cnn_grads by CPU
        break;
      }
    }
  }
}


//===================================================================================================================//
//-- Kernel building: accumulate gradients --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::addAccumulateKernels() {
  if (this->totalFilterSize > 0) {
    this->oclwCore.addKernel("cnn_accumulate_filters", "cnn_accumulate", this->totalFilterSize, 0);
    this->oclwCore. template addArgument<T>("cnn_accumulate_filters", "cnn_accum_dFilters");
    this->oclwCore. template addArgument<T>("cnn_accumulate_filters", "cnn_dFilters");
    this->oclwCore. template addArgument<ulong>("cnn_accumulate_filters", static_cast<ulong>(0));
    this->oclwCore. template addArgument<ulong>("cnn_accumulate_filters", this->totalFilterSize);
  }

  if (this->totalBiasSize > 0) {
    this->oclwCore.addKernel("cnn_accumulate_biases", "cnn_accumulate", this->totalBiasSize, 0);
    this->oclwCore. template addArgument<T>("cnn_accumulate_biases", "cnn_accum_dBiases");
    this->oclwCore. template addArgument<T>("cnn_accumulate_biases", "cnn_dBiases");
    this->oclwCore. template addArgument<ulong>("cnn_accumulate_biases", static_cast<ulong>(0));
    this->oclwCore. template addArgument<ulong>("cnn_accumulate_biases", this->totalBiasSize);
  }
}

//===================================================================================================================//
//-- Kernel setup: predict (forward only) --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setupPredictKernels() {
  this->oclwCore.clearKernels();
  this->invalidateAllKernelFlags();

  this->addForwardKernels();

  this->predictKernelsSetup = true;
}

//===================================================================================================================//
//-- Kernel setup: backpropagate (backward + accumulate) --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setupBackpropagateKernels() {
  this->oclwCore.clearKernels();
  this->invalidateAllKernelFlags();

  this->addBackwardKernels();
  this->addAccumulateKernels();

  this->backpropagateKernelsSetup = true;
}

//===================================================================================================================//
//-- Kernel setup: accumulate only --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setupAccumulateKernels() {
  this->oclwCore.clearKernels();
  this->invalidateAllKernelFlags();

  this->addAccumulateKernels();

  this->accumulateKernelsSetup = true;
}

//===================================================================================================================//
//-- Kernel setup: update parameters --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setupUpdateKernels(ulong numSamples) {
  this->oclwCore.clearKernels();
  this->invalidateAllKernelFlags();

  if (this->totalFilterSize > 0) {
    this->oclwCore.addKernel("cnn_update_filters", "cnn_update", this->totalFilterSize, 0);
    this->oclwCore. template addArgument<T>("cnn_update_filters", "cnn_filters");
    this->oclwCore. template addArgument<T>("cnn_update_filters", "cnn_accum_dFilters");
    this->oclwCore. template addArgument<ulong>("cnn_update_filters", static_cast<ulong>(0));
    this->oclwCore. template addArgument<ulong>("cnn_update_filters", this->totalFilterSize);
    this->oclwCore. template addArgument<ulong>("cnn_update_filters", numSamples);
    this->oclwCore. template addArgument<float>("cnn_update_filters",
        static_cast<float>(this->coreConfig.trainingConfig.learningRate));
  }

  if (this->totalBiasSize > 0) {
    this->oclwCore.addKernel("cnn_update_biases", "cnn_update", this->totalBiasSize, 0);
    this->oclwCore. template addArgument<T>("cnn_update_biases", "cnn_biases");
    this->oclwCore. template addArgument<T>("cnn_update_biases", "cnn_accum_dBiases");
    this->oclwCore. template addArgument<ulong>("cnn_update_biases", static_cast<ulong>(0));
    this->oclwCore. template addArgument<ulong>("cnn_update_biases", this->totalBiasSize);
    this->oclwCore. template addArgument<ulong>("cnn_update_biases", numSamples);
    this->oclwCore. template addArgument<float>("cnn_update_biases",
        static_cast<float>(this->coreConfig.trainingConfig.learningRate));
  }

  this->updateKernelsSetup = true;
}

//===================================================================================================================//
//-- Helper: invalidate all kernel flags --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::invalidateAllKernelFlags() {
  this->predictKernelsSetup = false;
  this->backpropagateKernelsSetup = false;
  this->accumulateKernelsSetup = false;
  this->updateKernelsSetup = false;
}


//===================================================================================================================//
//-- Predict --//
//===================================================================================================================//

template <typename T>
Output<T> CoreGPUWorker<T>::predict(const Input<T>& input) {
  // Set up forward kernels if needed
  if (!this->predictKernelsSetup) {
    this->setupPredictKernels();
  }

  // Write input to cnn_actvs at offset 0
  std::vector<T> inputVec(input.data.begin(), input.data.end());
  this->oclwCore. template writeBuffer<T>("cnn_actvs", inputVec, 0);

  // Run forward CNN kernels
  this->oclwCore.run();

  // Read CNN output from the last CNN layer's activation region
  ulong lastLayerIdx = this->layerInfos.size() - 1;
  ulong outputOffset = this->layerInfos[lastLayerIdx].actvOffset;
  std::vector<T> cnnOutput(this->flattenSize);
  this->oclwCore. template readBuffer<T>("cnn_actvs", cnnOutput, outputOffset);

  // Forward through ANN dense layers
  ANN::Input<T> annInput(cnnOutput.begin(), cnnOutput.end());
  ANN::Output<T> annOutput = this->annCore->predict(annInput);

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

  // Reset accumulators
  this->resetAccumulators();
  this->annCore->resetAccumulators();

  // Progress reporting
  ulong progressReports = this->coreConfig.trainingConfig.progressReports;
  if (progressReports == 0) progressReports = 1000;
  const ulong progressInterval = std::max(static_cast<ulong>(1), numSamplesInSubset / progressReports);
  ulong lastReportedSample = 0;

  for (ulong s = startIdx; s < endIdx; s++) {
    const Sample<T>& sample = samples[s];

    // === Phase 1: Forward CNN on GPU ===
    if (!this->predictKernelsSetup) {
      this->setupPredictKernels();
    }

    std::vector<T> inputVec(sample.input.data.begin(), sample.input.data.end());
    this->oclwCore. template writeBuffer<T>("cnn_actvs", inputVec, 0);
    this->oclwCore.run();

    // Read CNN output
    ulong lastLayerIdx = this->layerInfos.size() - 1;
    ulong outputOffset = this->layerInfos[lastLayerIdx].actvOffset;
    std::vector<T> cnnOutput(this->flattenSize);
    this->oclwCore. template readBuffer<T>("cnn_actvs", cnnOutput, outputOffset);

    // === Phase 2: ANN forward + backward on CPU ===
    ANN::Input<T> annInput(cnnOutput.begin(), cnnOutput.end());
    ANN::Output<T> annOutput = this->annCore->predict(annInput);

    Output<T> predicted(annOutput.begin(), annOutput.end());
    T sampleLoss = this->calculateLoss(predicted, sample.output);
    subsetLoss += sampleLoss;

    ANN::Output<T> annExpected(sample.output.begin(), sample.output.end());
    ANN::Tensor1D<T> dFlatInput = this->annCore->backpropagate(annExpected);
    this->annCore->accumulate();

    // === Phase 3: Backward + Accumulate CNN on GPU ===
    if (!this->backpropagateKernelsSetup) {
      this->setupBackpropagateKernels();
    }

    // Write gradient to the CNN output region in cnn_grads
    std::vector<T> gradVec(dFlatInput.begin(), dFlatInput.end());
    this->oclwCore. template writeBuffer<T>("cnn_grads", gradVec, outputOffset);
    this->oclwCore.run();

    // Report progress
    ulong currentSample = s - startIdx + 1;
    if (callback && currentSample >= lastReportedSample + progressInterval) {
      lastReportedSample = currentSample;
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
  // Forward CNN on GPU
  if (!this->predictKernelsSetup) {
    this->setupPredictKernels();
  }

  std::vector<T> inputVec(input.data.begin(), input.data.end());
  this->oclwCore. template writeBuffer<T>("cnn_actvs", inputVec, 0);
  this->oclwCore.run();

  // Read CNN output
  ulong lastLayerIdx = this->layerInfos.size() - 1;
  ulong outputOffset = this->layerInfos[lastLayerIdx].actvOffset;
  std::vector<T> cnnOutput(this->flattenSize);
  this->oclwCore. template readBuffer<T>("cnn_actvs", cnnOutput, outputOffset);

  // ANN forward + backward on CPU
  ANN::Input<T> annInput(cnnOutput.begin(), cnnOutput.end());
  this->annCore->predict(annInput);

  ANN::Output<T> annExpected(expected.begin(), expected.end());
  ANN::Tensor1D<T> dFlatInput = this->annCore->backpropagate(annExpected);
  this->annCore->accumulate();

  // Backward + Accumulate CNN on GPU
  if (!this->backpropagateKernelsSetup) {
    this->setupBackpropagateKernels();
  }

  std::vector<T> gradVec(dFlatInput.begin(), dFlatInput.end());
  this->oclwCore. template writeBuffer<T>("cnn_grads", gradVec, outputOffset);
  this->oclwCore.run();
}

//===================================================================================================================//
//-- Step-by-step: accumulate --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::accumulate() {
  if (!this->accumulateKernelsSetup) {
    this->setupAccumulateKernels();
  }

  this->oclwCore.run();
}

//===================================================================================================================//
//-- Step-by-step: reset accumulators --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::resetAccumulators() {
  if (this->totalFilterSize > 0) {
    std::vector<T> zeros(this->totalFilterSize, static_cast<T>(0));
    this->oclwCore. template writeBuffer<T>("cnn_accum_dFilters", zeros, 0);
  }

  if (this->totalBiasSize > 0) {
    std::vector<T> zeros(this->totalBiasSize, static_cast<T>(0));
    this->oclwCore. template writeBuffer<T>("cnn_accum_dBiases", zeros, 0);
  }
}

//===================================================================================================================//
//-- Gradient access (for multi-GPU merging) --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::readAccumulatedGradients(std::vector<T>& accumFilters, std::vector<T>& accumBiases) {
  accumFilters.resize(this->totalFilterSize);
  accumBiases.resize(this->totalBiasSize);

  if (this->totalFilterSize > 0) {
    this->oclwCore. template readBuffer<T>("cnn_accum_dFilters", accumFilters, 0);
  }

  if (this->totalBiasSize > 0) {
    this->oclwCore. template readBuffer<T>("cnn_accum_dBiases", accumBiases, 0);
  }
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setAccumulators(const std::vector<T>& accumFilters, const std::vector<T>& accumBiases) {
  if (this->totalFilterSize > 0) {
    this->oclwCore. template writeBuffer<T>("cnn_accum_dFilters", accumFilters, 0);
  }

  if (this->totalBiasSize > 0) {
    this->oclwCore. template writeBuffer<T>("cnn_accum_dBiases", accumBiases, 0);
  }
}


//===================================================================================================================//
//-- Weight update --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::update(ulong numSamples) {
  this->updateCNN(numSamples);
  this->updateANN(numSamples);
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::updateCNN(ulong numSamples) {
  if (!this->updateKernelsSetup) {
    this->setupUpdateKernels(numSamples);
  }

  this->oclwCore.run();
  this->invalidateAllKernelFlags();
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::updateANN(ulong numSamples) {
  this->annCore->update(numSamples);
}

//===================================================================================================================//
//-- Parameter synchronization: GPU -> CPU --//
//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::syncParametersFromGPU() {
  // Read filters from GPU and scatter back to per-layer parameters
  if (this->totalFilterSize > 0) {
    std::vector<T> flatFilters(this->totalFilterSize);
    this->oclwCore. template readBuffer<T>("cnn_filters", flatFilters, 0);

    for (ulong i = 0; i < this->convInfos.size(); i++) {
      ulong offset = this->convInfos[i].filterOffset;
      ulong count = this->convInfos[i].numFilterElems;
      this->parameters.convParams[i].filters.assign(
          flatFilters.begin() + static_cast<long>(offset),
          flatFilters.begin() + static_cast<long>(offset + count));
    }
  }

  // Read biases from GPU and scatter back to per-layer parameters
  if (this->totalBiasSize > 0) {
    std::vector<T> flatBiases(this->totalBiasSize);
    this->oclwCore. template readBuffer<T>("cnn_biases", flatBiases, 0);

    for (ulong i = 0; i < this->convInfos.size(); i++) {
      ulong offset = this->convInfos[i].biasOffset;
      ulong count = this->convInfos[i].numBiases;
      this->parameters.convParams[i].biases.assign(
          flatBiases.begin() + static_cast<long>(offset),
          flatBiases.begin() + static_cast<long>(offset + count));
    }
  }

  // Sync ANN dense layer parameters
  this->parameters.denseParams = this->annCore->getParameters();
}

//===================================================================================================================//
//-- Parameter access: ANN dense layers --//
//===================================================================================================================//

template <typename T>
ANN::Parameters<T> CoreGPUWorker<T>::getANNParameters() const {
  return this->annCore->getParameters();
}

//===================================================================================================================//

template <typename T>
void CoreGPUWorker<T>::setANNParameters(const ANN::Parameters<T>& params) {
  this->annCore->setParameters(params);
}

//===================================================================================================================//
//-- Loss calculation (MSE) --//
//===================================================================================================================//

template <typename T>
T CoreGPUWorker<T>::calculateLoss(const Output<T>& predicted, const Output<T>& expected) {
  T loss = static_cast<T>(0);

  for (ulong i = 0; i < expected.size(); i++) {
    T diff = predicted[i] - expected[i];
    loss += diff * diff;
  }

  return loss / static_cast<T>(expected.size());
}

//===================================================================================================================//
//-- Explicit template instantiations --//
//===================================================================================================================//

template class CNN::CoreGPUWorker<int>;
template class CNN::CoreGPUWorker<double>;
template class CNN::CoreGPUWorker<float>;
