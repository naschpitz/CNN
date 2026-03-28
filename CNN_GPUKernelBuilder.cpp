#include "CNN_GPUKernelBuilder.hpp"
#include "CNN_SlidingStrategy.hpp"

#include <cmath>
#include <string>

using namespace CNN;

//===================================================================================================================//

template <typename T>
GPUKernelBuilder<T>::GPUKernelBuilder(OpenCLWrapper::Core* core, const CoreGPUWorkerConfig<T>& workerConfig,
                                      GPUBufferManager<T>& bufferManager)
  : core(core),
    workerConfig(workerConfig),
    bufferManager(bufferManager)
{
}

//===================================================================================================================//
//-- Kernel setup --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::setupPredictKernels()
{
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  ulong numLayers = this->workerConfig.layersConfig.cnnLayers.size();

  this->addPropagateKernels(0, 0, numLayers, false);
  this->addCopyBridgeKernels(0);
  this->bufferManager.annGPUWorker->kernelBuilder->addPropagateKernels();

  this->predictKernelsSetup = true;
}

//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::setupTrainingKernels()
{
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  ulong numLayers = this->workerConfig.layersConfig.cnnLayers.size();

  // Propagate pipeline: CNN propagate → copy → ANN propagate
  this->addPropagateKernels(0, 0, numLayers, true);
  this->addCopyBridgeKernels(0);
  this->bufferManager.annGPUWorker->kernelBuilder->addPropagateKernels();

  // Backpropagate pipeline: ANN backpropagate (with input gradients) → reverse bridge → CNN backpropagate
  this->bufferManager.annGPUWorker->kernelBuilder->addBackpropagateKernels(true);
  this->addReverseBridgeKernels(0);
  this->addBackpropagateKernels(0, 0, numLayers);

  // Accumulate: CNN + ANN
  this->addCNNAccumulateKernels(0, 0, numLayers);
  this->bufferManager.annGPUWorker->kernelBuilder->addAccumulateKernels();

  // Loss: compute weighted MSE on GPU and accumulate into accum_loss buffer
  ulong outputActvOffset = this->bufferManager.annGPUWorker->bufferManager->getOutputActvOffset();
  ulong numOutputNeurons = this->bufferManager.annGPUWorker->bufferManager->getNumOutputNeurons();
  this->core->addKernel("calculate_sample_loss", "calculate_sample_loss", 1, 0);
  this->core->template addArgument<T>("calculate_sample_loss", "actvs");
  this->core->template addArgument<T>("calculate_sample_loss", "outputs");
  this->core->template addArgument<T>("calculate_sample_loss", "lossWeights");
  this->core->template addArgument<T>("calculate_sample_loss", "accum_loss");
  this->core->template addArgument<ulong>("calculate_sample_loss", outputActvOffset);
  this->core->template addArgument<ulong>("calculate_sample_loss", numOutputNeurons);
  this->core->template addArgument<ulong>("calculate_sample_loss",
                                          static_cast<ulong>(this->workerConfig.costFunctionConfig.type));

  this->trainingKernelsSetup = true;
}

//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::setupUpdateKernels(ulong numSamples)
{
  this->core->clearKernels();
  this->invalidateAllKernelFlags();

  this->addCNNUpdateKernels(numSamples, this->skipBNRunningStatsInUpdate);
  this->bufferManager.annGPUWorker->kernelBuilder->addUpdateKernels(numSamples);

  this->updateKernelsSetup = true;
}

//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::invalidateAllKernelFlags()
{
  this->predictKernelsSetup = false;
  this->trainingKernelsSetup = false;
  this->updateKernelsSetup = false;
}

//===================================================================================================================//
//-- addPropagateKernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addPropagateKernels(ulong sampleIdx, ulong layerStart, ulong layerEnd, bool training)
{
  const auto& cnnLayers = this->workerConfig.layersConfig.cnnLayers;
  ulong sampleStride = this->bufferManager.totalActvSize;
  ulong sampleOffset = sampleIdx * sampleStride;

  // Compute shape and type-specific indices at layerStart
  Shape3D currentShape = this->workerConfig.inputShape;
  ulong convIdx = 0;
  ulong poolIdx = 0;
  ulong normIdx = 0;

  for (ulong i = 0; i < layerStart; i++) {
    switch (cnnLayers[i].type) {
    case LayerType::CONV: {
      const auto& conv = std::get<ConvLayerConfig>(cnnLayers[i].config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
      ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
      currentShape = {conv.numFilters, outH, outW};
      convIdx++;
      break;
    }

    case LayerType::POOL: {
      const auto& pool = std::get<PoolLayerConfig>(cnnLayers[i].config);
      ulong outH = (currentShape.h - pool.poolH) / pool.strideY + 1;
      ulong outW = (currentShape.w - pool.poolW) / pool.strideX + 1;
      currentShape = {currentShape.c, outH, outW};
      poolIdx++;
      break;
    }

    case LayerType::GLOBALAVGPOOL:
      currentShape = {currentShape.c, 1, 1};
      break;
    case LayerType::GLOBALDUALPOOL:
      currentShape = {currentShape.c * 2, 1, 1};
      break;
    case LayerType::INSTANCENORM:
    case LayerType::BATCHNORM:
      normIdx++;
      break;
    default:
      break;
    }
  }

  std::string sampleStr = std::to_string(sampleIdx);

  for (ulong i = layerStart; i < layerEnd; i++) {
    const auto& layerConfig = cnnLayers[i];
    std::string layerStr = std::to_string(i);
    std::string kernelSuffix = "_s" + sampleStr + "_l" + layerStr;

    ulong inOffset = sampleOffset + this->bufferManager.layerInfos[i].actvOffset;
    ulong outOffset = sampleOffset + this->bufferManager.layerInfos[i + 1].actvOffset;

    switch (layerConfig.type) {
    case LayerType::CONV: {
      // Forward convolution via im2col + GEMM:
      //   1. im2col rearranges input patches into a column matrix in cnn_im2col workspace
      //   2. gemm computes Output = Filters × im2col_matrix + Bias
      // See opencl/CNN_GEMM.cpp.cl and opencl/CNN_Im2Col.cpp.cl for detailed explanations.
      const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
      ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;

      // im2col matrix dimensions: rows = filter volume, cols = number of output positions
      ulong im2colRows = currentShape.c * conv.filterH * conv.filterW;
      ulong im2colCols = outH * outW;
      ulong im2colSize = im2colRows * im2colCols;

      // Step 1: im2col — rearrange input patches into column matrix
      std::string im2colId = "im2col" + kernelSuffix;
      this->core->addKernel(im2colId, "im2col", im2colSize, 0);
      this->core->template addArgument<T>(im2colId, "cnn_actvs");
      this->core->template addArgument<T>(im2colId, "cnn_im2col");
      this->core->template addArgument<ulong>(im2colId, inOffset);
      this->core->template addArgument<ulong>(im2colId, currentShape.c);
      this->core->template addArgument<ulong>(im2colId, currentShape.h);
      this->core->template addArgument<ulong>(im2colId, currentShape.w);
      this->core->template addArgument<ulong>(im2colId, conv.filterH);
      this->core->template addArgument<ulong>(im2colId, conv.filterW);
      this->core->template addArgument<ulong>(im2colId, conv.strideY);
      this->core->template addArgument<ulong>(im2colId, conv.strideX);
      this->core->template addArgument<ulong>(im2colId, padY);
      this->core->template addArgument<ulong>(im2colId, padX);
      this->core->template addArgument<ulong>(im2colId, outH);
      this->core->template addArgument<ulong>(im2colId, outW);

      // Step 2: GEMM — Output = Filters × im2col_matrix + Biases
      //   A = cnn_filters, shape (numFilters, C_in*kH*kW) = (M, K)
      //   B = cnn_im2col, shape (C_in*kH*kW, outH*outW) = (K, N)
      //   C = cnn_actvs at outOffset, shape (numFilters, outH*outW) = (M, N)
      ulong M = conv.numFilters;
      ulong N = im2colCols;
      ulong K = im2colRows;
      ulong TILE = 16;
      ulong globalX = ((N + TILE - 1) / TILE) * TILE;
      ulong globalY = ((M + TILE - 1) / TILE) * TILE;

      ulong filterOffset = this->bufferManager.convInfos[convIdx].filterOffset;
      ulong biasOffset = this->bufferManager.convInfos[convIdx].biasOffset;

      std::string gemmId = "gemm_conv" + kernelSuffix;
      this->core->addKernel(gemmId, "gemm", globalX, globalY, TILE, TILE);
      this->core->template addArgument<T>(gemmId, "cnn_filters");
      this->core->template addArgument<T>(gemmId, "cnn_im2col");
      this->core->template addArgument<T>(gemmId, "cnn_actvs");
      this->core->template addArgument<T>(gemmId, "cnn_biases");
      this->core->template addArgument<ulong>(gemmId, filterOffset);
      this->core->template addArgument<ulong>(gemmId, static_cast<ulong>(0));
      this->core->template addArgument<ulong>(gemmId, outOffset);
      this->core->template addArgument<ulong>(gemmId, biasOffset);
      this->core->template addArgument<ulong>(gemmId, M);
      this->core->template addArgument<ulong>(gemmId, N);
      this->core->template addArgument<ulong>(gemmId, K);

      currentShape = {conv.numFilters, outH, outW};
      convIdx++;
      break;
    }

    case LayerType::RELU: {
      ulong size = currentShape.size();
      std::string kernelId = "relu" + kernelSuffix;
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
      ulong poolIdxOffset =
        sampleIdx * this->bufferManager.totalPoolIndexSize + this->bufferManager.poolInfos[poolIdx].indexOffset;

      if (pool.poolType == PoolTypeEnum::MAX) {
        std::string kernelId = "maxpool" + kernelSuffix;
        this->core->addKernel(kernelId, "calculate_maxpool", nElements, 0);
        this->core->template addArgument<T>(kernelId, "cnn_actvs");
        this->core->template addArgument<ulong>(kernelId, "cnn_pool_indices");
        this->core->template addArgument<ulong>(kernelId, inOffset);
        this->core->template addArgument<ulong>(kernelId, outOffset);
        this->core->template addArgument<ulong>(kernelId, poolIdxOffset);
        this->core->template addArgument<ulong>(kernelId, currentShape.c);
        this->core->template addArgument<ulong>(kernelId, currentShape.h);
        this->core->template addArgument<ulong>(kernelId, currentShape.w);
        this->core->template addArgument<ulong>(kernelId, pool.poolH);
        this->core->template addArgument<ulong>(kernelId, pool.poolW);
        this->core->template addArgument<ulong>(kernelId, pool.strideY);
        this->core->template addArgument<ulong>(kernelId, pool.strideX);
        this->core->template addArgument<ulong>(kernelId, outH);
        this->core->template addArgument<ulong>(kernelId, outW);
      } else {
        std::string kernelId = "avgpool" + kernelSuffix;
        this->core->addKernel(kernelId, "calculate_avgpool", nElements, 0);
        this->core->template addArgument<T>(kernelId, "cnn_actvs");
        this->core->template addArgument<ulong>(kernelId, inOffset);
        this->core->template addArgument<ulong>(kernelId, outOffset);
        this->core->template addArgument<ulong>(kernelId, currentShape.c);
        this->core->template addArgument<ulong>(kernelId, currentShape.h);
        this->core->template addArgument<ulong>(kernelId, currentShape.w);
        this->core->template addArgument<ulong>(kernelId, pool.poolH);
        this->core->template addArgument<ulong>(kernelId, pool.poolW);
        this->core->template addArgument<ulong>(kernelId, pool.strideY);
        this->core->template addArgument<ulong>(kernelId, pool.strideX);
        this->core->template addArgument<ulong>(kernelId, outH);
        this->core->template addArgument<ulong>(kernelId, outW);
      }

      currentShape = {currentShape.c, outH, outW};
      poolIdx++;
      break;
    }

    case LayerType::GLOBALAVGPOOL: {
      ulong localWS = 256;
      ulong gapGlobalWS = currentShape.c * localWS;

      std::string kernelId = "gap_propagate" + kernelSuffix;
      this->core->addKernel(kernelId, "gap_propagate", gapGlobalWS, 0, localWS);
      this->core->template addArgument<T>(kernelId, "cnn_actvs");
      this->core->template addArgument<ulong>(kernelId, inOffset);
      this->core->template addArgument<ulong>(kernelId, outOffset);
      this->core->template addArgument<ulong>(kernelId, currentShape.c);
      this->core->template addArgument<ulong>(kernelId, currentShape.h);
      this->core->template addArgument<ulong>(kernelId, currentShape.w);

      currentShape = {currentShape.c, 1, 1};
      break;
    }

    case LayerType::GLOBALDUALPOOL: {
      ulong localWS = 256;
      ulong gdpGlobalWS = currentShape.c * localWS;

      std::string kernelId = "gdp_propagate" + kernelSuffix;
      this->core->addKernel(kernelId, "gdp_propagate", gdpGlobalWS, 0, localWS);
      this->core->template addArgument<T>(kernelId, "cnn_actvs");
      this->core->template addArgument<ulong>(kernelId, inOffset);
      this->core->template addArgument<ulong>(kernelId, outOffset);
      this->core->template addArgument<ulong>(kernelId, currentShape.c);
      this->core->template addArgument<ulong>(kernelId, currentShape.h);
      this->core->template addArgument<ulong>(kernelId, currentShape.w);

      currentShape = {currentShape.c * 2, 1, 1};
      break;
    }

    case LayerType::INSTANCENORM: {
      const auto& bn = std::get<NormLayerConfig>(layerConfig.config);
      ulong size = currentShape.size();
      ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;

      // Per-sample spatial mean/var (N=1, sampleStride=0)
      ulong localWS = 256;
      ulong meanGlobalWS = currentShape.c * localWS;

      std::string meanId = "norm_mean" + kernelSuffix;
      this->core->addKernel(meanId, "norm_compute_mean", meanGlobalWS, 0, localWS);
      this->core->template addArgument<T>(meanId, "cnn_actvs");
      this->core->template addArgument<T>(meanId, "cnn_norm_batch_mean");
      this->core->template addArgument<ulong>(meanId, inOffset);
      this->core->template addArgument<ulong>(meanId, normParamOffset);
      this->core->template addArgument<ulong>(meanId, currentShape.c);
      this->core->template addArgument<ulong>(meanId, currentShape.h);
      this->core->template addArgument<ulong>(meanId, currentShape.w);
      this->core->template addArgument<ulong>(meanId, static_cast<ulong>(1));
      this->core->template addArgument<ulong>(meanId, static_cast<ulong>(0));

      std::string varId = "norm_var" + kernelSuffix;
      this->core->addKernel(varId, "norm_compute_var", meanGlobalWS, 0, localWS);
      this->core->template addArgument<T>(varId, "cnn_actvs");
      this->core->template addArgument<T>(varId, "cnn_norm_batch_mean");
      this->core->template addArgument<T>(varId, "cnn_norm_batch_var");
      this->core->template addArgument<ulong>(varId, inOffset);
      this->core->template addArgument<ulong>(varId, normParamOffset);
      this->core->template addArgument<ulong>(varId, currentShape.c);
      this->core->template addArgument<ulong>(varId, currentShape.h);
      this->core->template addArgument<ulong>(varId, currentShape.w);
      this->core->template addArgument<ulong>(varId, static_cast<ulong>(1));
      this->core->template addArgument<ulong>(varId, static_cast<ulong>(0));

      std::string normId = "norm_normalize" + kernelSuffix;
      this->core->addKernel(normId, "norm_normalize", size, 0);
      this->core->template addArgument<T>(normId, "cnn_actvs");
      this->core->template addArgument<T>(normId, "cnn_norm_xnorm");
      this->core->template addArgument<T>(normId, "cnn_norm_gamma");
      this->core->template addArgument<T>(normId, "cnn_norm_beta");
      this->core->template addArgument<T>(normId, "cnn_norm_batch_mean");
      this->core->template addArgument<T>(normId, "cnn_norm_batch_var");
      this->core->template addArgument<ulong>(normId, inOffset);
      this->core->template addArgument<ulong>(normId, outOffset);
      this->core->template addArgument<ulong>(normId, sampleOffset + this->bufferManager.layerInfos[i].actvOffset);
      this->core->template addArgument<ulong>(normId, normParamOffset);
      this->core->template addArgument<ulong>(normId, currentShape.c);
      this->core->template addArgument<ulong>(normId, currentShape.h);
      this->core->template addArgument<ulong>(normId, currentShape.w);
      this->core->template addArgument<ulong>(normId, static_cast<ulong>(1));
      this->core->template addArgument<ulong>(normId, static_cast<ulong>(0));
      this->core->template addArgument<float>(normId, bn.epsilon);

      normIdx++;
      break;
    }

    case LayerType::BATCHNORM: {
      const auto& bn = std::get<NormLayerConfig>(layerConfig.config);
      ulong size = currentShape.size();
      ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;

      if (training) {
        // Skip — handled by addBatchNormForwardKernels in the segment-based batch path
      } else {
        // Inference: use running stats (N=1, sampleStride=0)
        std::string normId = "norm_normalize" + kernelSuffix;
        this->core->addKernel(normId, "norm_normalize", size, 0);
        this->core->template addArgument<T>(normId, "cnn_actvs");
        this->core->template addArgument<T>(normId, "cnn_norm_xnorm");
        this->core->template addArgument<T>(normId, "cnn_norm_gamma");
        this->core->template addArgument<T>(normId, "cnn_norm_beta");
        this->core->template addArgument<T>(normId, "cnn_norm_running_mean");
        this->core->template addArgument<T>(normId, "cnn_norm_running_var");
        this->core->template addArgument<ulong>(normId, inOffset);
        this->core->template addArgument<ulong>(normId, outOffset);
        this->core->template addArgument<ulong>(normId, sampleOffset + this->bufferManager.layerInfos[i].actvOffset);
        this->core->template addArgument<ulong>(normId, normParamOffset);
        this->core->template addArgument<ulong>(normId, currentShape.c);
        this->core->template addArgument<ulong>(normId, currentShape.h);
        this->core->template addArgument<ulong>(normId, currentShape.w);
        this->core->template addArgument<ulong>(normId, static_cast<ulong>(1));
        this->core->template addArgument<ulong>(normId, static_cast<ulong>(0));
        this->core->template addArgument<float>(normId, bn.epsilon);
      }

      normIdx++;
      break;
    }

    case LayerType::FLATTEN: {
      break;
    }
    }
  }
}

//===================================================================================================================//
//-- addBackpropagateKernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addBackpropagateKernels(ulong sampleIdx, ulong layerStart, ulong layerEnd)
{
  const auto& cnnLayers = this->workerConfig.layersConfig.cnnLayers;
  ulong numLayers = cnnLayers.size();
  ulong sampleStride = this->bufferManager.totalActvSize;
  ulong sampleOffset = sampleIdx * sampleStride;

  // Precompute shapes for each layer (propagate direction)
  std::vector<Shape3D> shapes(numLayers + 1);
  shapes[0] = this->workerConfig.inputShape;

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

    case LayerType::POOL: {
      const auto& pool = std::get<PoolLayerConfig>(cnnLayers[i].config);
      ulong outH = (inShape.h - pool.poolH) / pool.strideY + 1;
      ulong outW = (inShape.w - pool.poolW) / pool.strideX + 1;
      shapes[i + 1] = {inShape.c, outH, outW};
      break;
    }

    case LayerType::GLOBALAVGPOOL:
      shapes[i + 1] = {inShape.c, 1, 1};
      break;
    case LayerType::GLOBALDUALPOOL:
      shapes[i + 1] = {inShape.c * 2, 1, 1};
      break;
    default:
      shapes[i + 1] = inShape;
      break;
    }
  }

  // Count type-specific indices up to layerEnd
  ulong convIdx = 0;
  ulong poolIdx = 0;
  ulong normIdx = 0;

  for (ulong i = 0; i < layerEnd; i++) {
    switch (cnnLayers[i].type) {
    case LayerType::CONV:
      convIdx++;
      break;
    case LayerType::POOL:
      poolIdx++;
      break;
    case LayerType::INSTANCENORM:
    case LayerType::BATCHNORM:
      normIdx++;
      break;
    default:
      break;
    }
  }

  std::string sampleStr = std::to_string(sampleIdx);

  // Iterate in reverse from layerEnd-1 down to layerStart
  for (long i = static_cast<long>(layerEnd) - 1; i >= static_cast<long>(layerStart); i--) {
    const auto& layerConfig = cnnLayers[static_cast<ulong>(i)];
    std::string layerStr = std::to_string(i);
    std::string kernelSuffix = "_s" + sampleStr + "_l" + layerStr;

    Shape3D inShape = shapes[static_cast<ulong>(i)];
    Shape3D outShape = shapes[static_cast<ulong>(i) + 1];

    ulong gradInOffset = sampleOffset + this->bufferManager.layerInfos[static_cast<ulong>(i)].actvOffset;
    ulong gradOutOffset = sampleOffset + this->bufferManager.layerInfos[static_cast<ulong>(i) + 1].actvOffset;
    ulong actvInOffset = sampleOffset + this->bufferManager.layerInfos[static_cast<ulong>(i)].actvOffset;

    switch (layerConfig.type) {
    case LayerType::CONV: {
      // Backward convolution via im2col + GEMM (mirrors the forward pass approach):
      //   dFilters = dOut × im2col(input)^T          (gemm_transB)
      //   dBiases  = sum of dOut over spatial dims    (existing reduction kernel)
      //   dInput   = col2im(Filters^T × dOut)         (gemm_transA + col2im)
      // See opencl/CNN_GEMM.cpp.cl and opencl/CNN_Im2Col.cpp.cl for detailed explanations.
      convIdx--;
      const auto& conv = std::get<ConvLayerConfig>(layerConfig.config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = outShape.h;
      ulong outW = outShape.w;

      ulong im2colRows = inShape.c * conv.filterH * conv.filterW;
      ulong im2colCols = outH * outW;
      ulong im2colSize = im2colRows * im2colCols;
      ulong filterOffset = this->bufferManager.convInfos[convIdx].filterOffset;
      ulong biasOffset = this->bufferManager.convInfos[convIdx].biasOffset;
      ulong TILE = 16;

      // --- dFilters: im2col(input) then dFilters = dOut × im2col^T ---

      // Step 1: im2col — rearrange input patches into column matrix
      std::string im2colFiltId = "im2col_bk_filt" + kernelSuffix;
      this->core->addKernel(im2colFiltId, "im2col", im2colSize, 0);
      this->core->template addArgument<T>(im2colFiltId, "cnn_actvs");
      this->core->template addArgument<T>(im2colFiltId, "cnn_im2col");
      this->core->template addArgument<ulong>(im2colFiltId, actvInOffset);
      this->core->template addArgument<ulong>(im2colFiltId, inShape.c);
      this->core->template addArgument<ulong>(im2colFiltId, inShape.h);
      this->core->template addArgument<ulong>(im2colFiltId, inShape.w);
      this->core->template addArgument<ulong>(im2colFiltId, conv.filterH);
      this->core->template addArgument<ulong>(im2colFiltId, conv.filterW);
      this->core->template addArgument<ulong>(im2colFiltId, conv.strideY);
      this->core->template addArgument<ulong>(im2colFiltId, conv.strideX);
      this->core->template addArgument<ulong>(im2colFiltId, padY);
      this->core->template addArgument<ulong>(im2colFiltId, padX);
      this->core->template addArgument<ulong>(im2colFiltId, outH);
      this->core->template addArgument<ulong>(im2colFiltId, outW);

      // Step 2: gemm_transB — dFilters = dOut × im2col^T
      //   A = cnn_grads at gradOutOffset, shape (numFilters, outH*outW) = (M, K)
      //   B = cnn_im2col, shape (C_in*kH*kW, outH*outW) — transposed → (outH*outW, C_in*kH*kW)
      //   C = cnn_dFilters at filterOffset, shape (numFilters, C_in*kH*kW) = (M, N)
      ulong dF_M = conv.numFilters;
      ulong dF_N = im2colRows;
      ulong dF_K = im2colCols;
      ulong dF_globalX = ((dF_N + TILE - 1) / TILE) * TILE;
      ulong dF_globalY = ((dF_M + TILE - 1) / TILE) * TILE;

      std::string gemmFiltId = "gemm_dFilters" + kernelSuffix;
      this->core->addKernel(gemmFiltId, "gemm_transB", dF_globalX, dF_globalY, TILE, TILE);
      this->core->template addArgument<T>(gemmFiltId, "cnn_grads");
      this->core->template addArgument<T>(gemmFiltId, "cnn_im2col");
      this->core->template addArgument<T>(gemmFiltId, "cnn_dFilters");
      this->core->template addArgument<ulong>(gemmFiltId, gradOutOffset);
      this->core->template addArgument<ulong>(gemmFiltId, static_cast<ulong>(0));
      this->core->template addArgument<ulong>(gemmFiltId, filterOffset);
      this->core->template addArgument<ulong>(gemmFiltId, dF_M);
      this->core->template addArgument<ulong>(gemmFiltId, dF_N);
      this->core->template addArgument<ulong>(gemmFiltId, dF_K);

      // --- dBiases: keep existing reduction kernel (unchanged) ---
      std::string biasId = "dBiases" + kernelSuffix;
      ulong biasLocalWS = 256;
      ulong biasGlobalWS = conv.numFilters * biasLocalWS;
      this->core->addKernel(biasId, "calculate_dCost_dBiases", biasGlobalWS, 0, biasLocalWS);
      this->core->template addArgument<T>(biasId, "cnn_grads");
      this->core->template addArgument<T>(biasId, "cnn_dBiases");
      this->core->template addArgument<ulong>(biasId, gradOutOffset);
      this->core->template addArgument<ulong>(biasId, biasOffset);
      this->core->template addArgument<ulong>(biasId, conv.numFilters);
      this->core->template addArgument<ulong>(biasId, outH);
      this->core->template addArgument<ulong>(biasId, outW);

      // --- dInput: gemm_transA then col2im (skip if first layer) ---
      if (i > 0) {
        ulong inSize = inShape.size();

        // Step 0: zero the input gradient buffer (col2im accumulates)
        std::string zeroId = "zero_conv" + kernelSuffix;
        this->core->addKernel(zeroId, "zero_buffer", inSize, 0);
        this->core->template addArgument<T>(zeroId, "cnn_grads");
        this->core->template addArgument<ulong>(zeroId, gradInOffset);
        this->core->template addArgument<ulong>(zeroId, inSize);

        // Step 1: gemm_transA — dInput_cols = Filter^T × dOut
        //   A = cnn_filters at filterOffset, shape (numFilters, C_in*kH*kW) — transposed
        //   B = cnn_grads at gradOutOffset, shape (numFilters, outH*outW) = (K, N)
        //   C = cnn_im2col, shape (C_in*kH*kW, outH*outW) = (M, N)
        ulong dI_M = im2colRows;
        ulong dI_N = im2colCols;
        ulong dI_K = conv.numFilters;
        ulong dI_globalX = ((dI_N + TILE - 1) / TILE) * TILE;
        ulong dI_globalY = ((dI_M + TILE - 1) / TILE) * TILE;

        std::string gemmInputId = "gemm_dInput" + kernelSuffix;
        this->core->addKernel(gemmInputId, "gemm_transA", dI_globalX, dI_globalY, TILE, TILE);
        this->core->template addArgument<T>(gemmInputId, "cnn_filters");
        this->core->template addArgument<T>(gemmInputId, "cnn_grads");
        this->core->template addArgument<T>(gemmInputId, "cnn_im2col");
        this->core->template addArgument<ulong>(gemmInputId, filterOffset);
        this->core->template addArgument<ulong>(gemmInputId, gradOutOffset);
        this->core->template addArgument<ulong>(gemmInputId, static_cast<ulong>(0));
        this->core->template addArgument<ulong>(gemmInputId, dI_M);
        this->core->template addArgument<ulong>(gemmInputId, dI_N);
        this->core->template addArgument<ulong>(gemmInputId, dI_K);

        // Step 2: col2im — scatter dInput_cols back to input gradient buffer
        std::string col2imId = "col2im" + kernelSuffix;
        this->core->addKernel(col2imId, "col2im", inSize, 0);
        this->core->template addArgument<T>(col2imId, "cnn_im2col");
        this->core->template addArgument<T>(col2imId, "cnn_grads");
        this->core->template addArgument<ulong>(col2imId, gradInOffset);
        this->core->template addArgument<ulong>(col2imId, inShape.c);
        this->core->template addArgument<ulong>(col2imId, inShape.h);
        this->core->template addArgument<ulong>(col2imId, inShape.w);
        this->core->template addArgument<ulong>(col2imId, conv.filterH);
        this->core->template addArgument<ulong>(col2imId, conv.filterW);
        this->core->template addArgument<ulong>(col2imId, conv.strideY);
        this->core->template addArgument<ulong>(col2imId, conv.strideX);
        this->core->template addArgument<ulong>(col2imId, padY);
        this->core->template addArgument<ulong>(col2imId, padX);
        this->core->template addArgument<ulong>(col2imId, outH);
        this->core->template addArgument<ulong>(col2imId, outW);
      }

      break;
    }

    case LayerType::RELU: {
      ulong size = inShape.size();
      std::string kernelId = "dRelu" + kernelSuffix;
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
      const auto& pool = std::get<PoolLayerConfig>(layerConfig.config);
      ulong inSize = inShape.size();
      ulong outSize = outShape.size();
      ulong poolIdxOffset =
        sampleIdx * this->bufferManager.totalPoolIndexSize + this->bufferManager.poolInfos[poolIdx].indexOffset;

      std::string zeroId = "zero_pool" + kernelSuffix;
      this->core->addKernel(zeroId, "zero_buffer", inSize, 0);
      this->core->template addArgument<T>(zeroId, "cnn_grads");
      this->core->template addArgument<ulong>(zeroId, gradInOffset);
      this->core->template addArgument<ulong>(zeroId, inSize);

      if (pool.poolType == PoolTypeEnum::MAX) {
        std::string poolId = "dMaxpool" + kernelSuffix;
        this->core->addKernel(poolId, "calculate_dCost_dMaxpool", outSize, 0);
        this->core->template addArgument<T>(poolId, "cnn_grads");
        this->core->template addArgument<ulong>(poolId, "cnn_pool_indices");
        this->core->template addArgument<ulong>(poolId, gradOutOffset);
        this->core->template addArgument<ulong>(poolId, poolIdxOffset);
        this->core->template addArgument<ulong>(poolId, outSize);
      } else {
        std::string poolId = "dAvgpool" + kernelSuffix;
        this->core->addKernel(poolId, "calculate_dCost_dAvgpool", outSize, 0);
        this->core->template addArgument<T>(poolId, "cnn_grads");
        this->core->template addArgument<ulong>(poolId, gradInOffset);
        this->core->template addArgument<ulong>(poolId, gradOutOffset);
        this->core->template addArgument<ulong>(poolId, inShape.c);
        this->core->template addArgument<ulong>(poolId, inShape.h);
        this->core->template addArgument<ulong>(poolId, inShape.w);
        this->core->template addArgument<ulong>(poolId, pool.poolH);
        this->core->template addArgument<ulong>(poolId, pool.poolW);
        this->core->template addArgument<ulong>(poolId, pool.strideY);
        this->core->template addArgument<ulong>(poolId, pool.strideX);
        this->core->template addArgument<ulong>(poolId, outShape.h);
        this->core->template addArgument<ulong>(poolId, outShape.w);
      }

      break;
    }

    case LayerType::GLOBALAVGPOOL: {
      ulong size = inShape.size();

      std::string kernelId = "gap_back" + kernelSuffix;
      this->core->addKernel(kernelId, "gap_backpropagate", size, 0);
      this->core->template addArgument<T>(kernelId, "cnn_grads");
      this->core->template addArgument<ulong>(kernelId, gradInOffset);
      this->core->template addArgument<ulong>(kernelId, gradOutOffset);
      this->core->template addArgument<ulong>(kernelId, inShape.c);
      this->core->template addArgument<ulong>(kernelId, inShape.h);
      this->core->template addArgument<ulong>(kernelId, inShape.w);
      break;
    }

    case LayerType::GLOBALDUALPOOL: {
      ulong size = inShape.size();

      std::string kernelId = "gdp_back" + kernelSuffix;
      this->core->addKernel(kernelId, "gdp_backpropagate", size, 0);
      this->core->template addArgument<T>(kernelId, "cnn_grads");
      this->core->template addArgument<T>(kernelId, "cnn_actvs");
      this->core->template addArgument<ulong>(kernelId, gradInOffset);
      this->core->template addArgument<ulong>(kernelId, gradOutOffset);
      this->core->template addArgument<ulong>(kernelId, actvInOffset);
      this->core->template addArgument<ulong>(kernelId, inShape.c);
      this->core->template addArgument<ulong>(kernelId, inShape.h);
      this->core->template addArgument<ulong>(kernelId, inShape.w);
      break;
    }

    case LayerType::INSTANCENORM: {
      normIdx--;
      const auto& bn = std::get<NormLayerConfig>(layerConfig.config);
      ulong size = inShape.size();
      ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;

      // dGamma and dBeta (N=1, sampleStride=0 for per-sample backprop)
      ulong localWS = 256;
      ulong dgGlobalWS = inShape.c * localWS;
      std::string dgId = "norm_dGammaBeta" + kernelSuffix;
      this->core->addKernel(dgId, "norm_dGammaBeta", dgGlobalWS, 0, localWS);
      this->core->template addArgument<T>(dgId, "cnn_grads");
      this->core->template addArgument<T>(dgId, "cnn_norm_xnorm");
      this->core->template addArgument<T>(dgId, "cnn_norm_dGamma");
      this->core->template addArgument<T>(dgId, "cnn_norm_dBeta");
      this->core->template addArgument<ulong>(dgId, gradOutOffset);
      this->core->template addArgument<ulong>(dgId, actvInOffset);
      this->core->template addArgument<ulong>(dgId, normParamOffset);
      this->core->template addArgument<ulong>(dgId, inShape.c);
      this->core->template addArgument<ulong>(dgId, inShape.h);
      this->core->template addArgument<ulong>(dgId, inShape.w);
      this->core->template addArgument<ulong>(dgId, static_cast<ulong>(1));
      this->core->template addArgument<ulong>(dgId, static_cast<ulong>(0));

      // dInput (N=1, sampleStride=0)
      std::string diId = "norm_dInput" + kernelSuffix;
      this->core->addKernel(diId, "norm_dInput", size, 0);
      this->core->template addArgument<T>(diId, "cnn_grads");
      this->core->template addArgument<T>(diId, "cnn_norm_xnorm");
      this->core->template addArgument<T>(diId, "cnn_norm_gamma");
      this->core->template addArgument<T>(diId, "cnn_norm_dGamma");
      this->core->template addArgument<T>(diId, "cnn_norm_dBeta");
      this->core->template addArgument<T>(diId, "cnn_norm_batch_var");
      this->core->template addArgument<ulong>(diId, gradInOffset);
      this->core->template addArgument<ulong>(diId, gradOutOffset);
      this->core->template addArgument<ulong>(diId, actvInOffset);
      this->core->template addArgument<ulong>(diId, normParamOffset);
      this->core->template addArgument<ulong>(diId, inShape.c);
      this->core->template addArgument<ulong>(diId, inShape.h);
      this->core->template addArgument<ulong>(diId, inShape.w);
      this->core->template addArgument<ulong>(diId, static_cast<ulong>(1));
      this->core->template addArgument<ulong>(diId, static_cast<ulong>(0));
      this->core->template addArgument<float>(diId, bn.epsilon);
      break;
    }

    case LayerType::BATCHNORM: {
      // Skip — handled by addBatchNormBackwardKernels
      normIdx--;
      break;
    }

    case LayerType::FLATTEN: {
      break;
    }
    }
  }
}

//===================================================================================================================//
//-- addCopyBridgeKernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addCopyBridgeKernels(ulong sampleIdx)
{
  ulong sampleStride = this->bufferManager.totalActvSize;
  ulong lastLayerIdx = this->bufferManager.layerInfos.size() - 1;
  ulong cnnOutputOffset = sampleIdx * sampleStride + this->bufferManager.layerInfos[lastLayerIdx].actvOffset;

  std::string kernelId = "copy_cnn_to_ann_s" + std::to_string(sampleIdx);
  this->core->addKernel(kernelId, "copy_cnn_to_ann", this->bufferManager.flattenSize, 0);
  this->core->template addArgument<T>(kernelId, "cnn_actvs");
  this->core->template addArgument<T>(kernelId, "actvs");
  this->core->template addArgument<ulong>(kernelId, cnnOutputOffset);
  this->core->template addArgument<ulong>(kernelId, this->bufferManager.flattenSize);
}

//===================================================================================================================//
//-- addReverseBridgeKernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addReverseBridgeKernels(ulong sampleIdx)
{
  ulong sampleStride = this->bufferManager.totalActvSize;
  ulong lastLayerIdx = this->bufferManager.layerInfos.size() - 1;
  ulong cnnOutputOffset = sampleIdx * sampleStride + this->bufferManager.layerInfos[lastLayerIdx].actvOffset;

  std::string kernelId = "copy_ann_grad_to_cnn_s" + std::to_string(sampleIdx);
  this->core->addKernel(kernelId, "copy_ann_grad_to_cnn", this->bufferManager.flattenSize, 0);
  this->core->template addArgument<T>(kernelId, "dCost_dActvs");
  this->core->template addArgument<T>(kernelId, "cnn_grads");
  this->core->template addArgument<ulong>(kernelId, cnnOutputOffset);
  this->core->template addArgument<ulong>(kernelId, this->bufferManager.flattenSize);
}

//===================================================================================================================//
//-- addCNNAccumulateKernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addCNNAccumulateKernels(ulong sampleIdx, ulong layerStart, ulong layerEnd)
{
  const auto& cnnLayers = this->workerConfig.layersConfig.cnnLayers;
  std::string sampleStr = std::to_string(sampleIdx);

  // Accumulate conv filter and bias gradients for conv layers within [layerStart, layerEnd)
  ulong convIdx = 0;

  for (ulong i = 0; i < layerEnd; i++) {
    if (cnnLayers[i].type == LayerType::CONV) {
      if (i >= layerStart) {
        ulong filterOffset = this->bufferManager.convInfos[convIdx].filterOffset;
        ulong numFilterElems = this->bufferManager.convInfos[convIdx].numFilterElems;
        ulong biasOffset = this->bufferManager.convInfos[convIdx].biasOffset;
        ulong numBiases = this->bufferManager.convInfos[convIdx].numBiases;

        std::string fId = "accum_filters_s" + sampleStr + "_c" + std::to_string(convIdx);
        this->core->addKernel(fId, "accumulate_gradients", numFilterElems, 0);
        this->core->template addArgument<T>(fId, "cnn_accum_dFilters");
        this->core->template addArgument<T>(fId, "cnn_dFilters");
        this->core->template addArgument<ulong>(fId, filterOffset);
        this->core->template addArgument<ulong>(fId, numFilterElems);

        std::string bId = "accum_biases_s" + sampleStr + "_c" + std::to_string(convIdx);
        this->core->addKernel(bId, "accumulate_gradients", numBiases, 0);
        this->core->template addArgument<T>(bId, "cnn_accum_dBiases");
        this->core->template addArgument<T>(bId, "cnn_dBiases");
        this->core->template addArgument<ulong>(bId, biasOffset);
        this->core->template addArgument<ulong>(bId, numBiases);
      }

      convIdx++;
    }
  }

  // Accumulate norm param gradients and batch mean/var for norm layers within [layerStart, layerEnd)
  ulong normIdx = 0;

  for (ulong i = 0; i < layerEnd; i++) {
    if (cnnLayers[i].type == LayerType::INSTANCENORM || cnnLayers[i].type == LayerType::BATCHNORM) {
      if (i >= layerStart) {
        ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;
        ulong numChannels = this->bufferManager.normInfos[normIdx].numChannels;
        std::string idx = std::to_string(normIdx);

        std::string dgId = "accum_norm_dGamma_s" + sampleStr + "_n" + idx;
        this->core->addKernel(dgId, "accumulate_gradients", numChannels, 0);
        this->core->template addArgument<T>(dgId, "cnn_accum_norm_dGamma");
        this->core->template addArgument<T>(dgId, "cnn_norm_dGamma");
        this->core->template addArgument<ulong>(dgId, normParamOffset);
        this->core->template addArgument<ulong>(dgId, numChannels);

        std::string dbId = "accum_norm_dBeta_s" + sampleStr + "_n" + idx;
        this->core->addKernel(dbId, "accumulate_gradients", numChannels, 0);
        this->core->template addArgument<T>(dbId, "cnn_accum_norm_dBeta");
        this->core->template addArgument<T>(dbId, "cnn_norm_dBeta");
        this->core->template addArgument<ulong>(dbId, normParamOffset);
        this->core->template addArgument<ulong>(dbId, numChannels);

        std::string mId = "accum_norm_mean_s" + sampleStr + "_n" + idx;
        this->core->addKernel(mId, "accumulate_gradients", numChannels, 0);
        this->core->template addArgument<T>(mId, "cnn_accum_norm_batch_mean");
        this->core->template addArgument<T>(mId, "cnn_norm_batch_mean");
        this->core->template addArgument<ulong>(mId, normParamOffset);
        this->core->template addArgument<ulong>(mId, numChannels);

        std::string vId = "accum_norm_var_s" + sampleStr + "_n" + idx;
        this->core->addKernel(vId, "accumulate_gradients", numChannels, 0);
        this->core->template addArgument<T>(vId, "cnn_accum_norm_batch_var");
        this->core->template addArgument<T>(vId, "cnn_norm_batch_var");
        this->core->template addArgument<ulong>(vId, normParamOffset);
        this->core->template addArgument<ulong>(vId, numChannels);
      }

      normIdx++;
    }
  }
}

//===================================================================================================================//
//-- addCNNUpdateKernels --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addCNNUpdateKernels(ulong numSamples, bool skipBNRunningStats)
{
  if (this->workerConfig.trainingConfig.optimizer.type == OptimizerType::ADAM) {
    const auto& opt = this->workerConfig.trainingConfig.optimizer;
    this->adam_t++;

    float bc1 = 1.0f - std::pow(static_cast<float>(opt.beta1), static_cast<float>(this->adam_t));
    float bc2 = 1.0f - std::pow(static_cast<float>(opt.beta2), static_cast<float>(this->adam_t));

    if (this->bufferManager.totalFilterSize > 0) {
      this->core->addKernel("update_parameters_filters", "update_parameters_adam", this->bufferManager.totalFilterSize,
                            0);
      this->core->template addArgument<T>("update_parameters_filters", "cnn_filters");
      this->core->template addArgument<T>("update_parameters_filters", "cnn_accum_dFilters");
      this->core->template addArgument<T>("update_parameters_filters", "cnn_adam_m_filters");
      this->core->template addArgument<T>("update_parameters_filters", "cnn_adam_v_filters");
      this->core->template addArgument<ulong>("update_parameters_filters", static_cast<ulong>(0));
      this->core->template addArgument<ulong>("update_parameters_filters", this->bufferManager.totalFilterSize);
      this->core->template addArgument<ulong>("update_parameters_filters", numSamples);
      this->core->template addArgument<float>("update_parameters_filters",
                                              static_cast<float>(this->workerConfig.trainingConfig.learningRate));
      this->core->template addArgument<float>("update_parameters_filters", static_cast<float>(opt.beta1));
      this->core->template addArgument<float>("update_parameters_filters", static_cast<float>(opt.beta2));
      this->core->template addArgument<float>("update_parameters_filters", static_cast<float>(opt.epsilon));
      this->core->template addArgument<float>("update_parameters_filters", bc1);
      this->core->template addArgument<float>("update_parameters_filters", bc2);
    }

    if (this->bufferManager.totalBiasSize > 0) {
      this->core->addKernel("update_parameters_biases", "update_parameters_adam", this->bufferManager.totalBiasSize, 0);
      this->core->template addArgument<T>("update_parameters_biases", "cnn_biases");
      this->core->template addArgument<T>("update_parameters_biases", "cnn_accum_dBiases");
      this->core->template addArgument<T>("update_parameters_biases", "cnn_adam_m_biases");
      this->core->template addArgument<T>("update_parameters_biases", "cnn_adam_v_biases");
      this->core->template addArgument<ulong>("update_parameters_biases", static_cast<ulong>(0));
      this->core->template addArgument<ulong>("update_parameters_biases", this->bufferManager.totalBiasSize);
      this->core->template addArgument<ulong>("update_parameters_biases", numSamples);
      this->core->template addArgument<float>("update_parameters_biases",
                                              static_cast<float>(this->workerConfig.trainingConfig.learningRate));
      this->core->template addArgument<float>("update_parameters_biases", static_cast<float>(opt.beta1));
      this->core->template addArgument<float>("update_parameters_biases", static_cast<float>(opt.beta2));
      this->core->template addArgument<float>("update_parameters_biases", static_cast<float>(opt.epsilon));
      this->core->template addArgument<float>("update_parameters_biases", bc1);
      this->core->template addArgument<float>("update_parameters_biases", bc2);
    }

    if (this->bufferManager.totalNormParamSize > 0) {
      this->core->addKernel("update_parameters_norm_gamma", "update_parameters_adam",
                            this->bufferManager.totalNormParamSize, 0);
      this->core->template addArgument<T>("update_parameters_norm_gamma", "cnn_norm_gamma");
      this->core->template addArgument<T>("update_parameters_norm_gamma", "cnn_accum_norm_dGamma");
      this->core->template addArgument<T>("update_parameters_norm_gamma", "cnn_adam_m_norm_gamma");
      this->core->template addArgument<T>("update_parameters_norm_gamma", "cnn_adam_v_norm_gamma");
      this->core->template addArgument<ulong>("update_parameters_norm_gamma", static_cast<ulong>(0));
      this->core->template addArgument<ulong>("update_parameters_norm_gamma", this->bufferManager.totalNormParamSize);
      this->core->template addArgument<ulong>("update_parameters_norm_gamma", numSamples);
      this->core->template addArgument<float>("update_parameters_norm_gamma",
                                              static_cast<float>(this->workerConfig.trainingConfig.learningRate));
      this->core->template addArgument<float>("update_parameters_norm_gamma", static_cast<float>(opt.beta1));
      this->core->template addArgument<float>("update_parameters_norm_gamma", static_cast<float>(opt.beta2));
      this->core->template addArgument<float>("update_parameters_norm_gamma", static_cast<float>(opt.epsilon));
      this->core->template addArgument<float>("update_parameters_norm_gamma", bc1);
      this->core->template addArgument<float>("update_parameters_norm_gamma", bc2);

      this->core->addKernel("update_parameters_norm_beta", "update_parameters_adam",
                            this->bufferManager.totalNormParamSize, 0);
      this->core->template addArgument<T>("update_parameters_norm_beta", "cnn_norm_beta");
      this->core->template addArgument<T>("update_parameters_norm_beta", "cnn_accum_norm_dBeta");
      this->core->template addArgument<T>("update_parameters_norm_beta", "cnn_adam_m_norm_beta");
      this->core->template addArgument<T>("update_parameters_norm_beta", "cnn_adam_v_norm_beta");
      this->core->template addArgument<ulong>("update_parameters_norm_beta", static_cast<ulong>(0));
      this->core->template addArgument<ulong>("update_parameters_norm_beta", this->bufferManager.totalNormParamSize);
      this->core->template addArgument<ulong>("update_parameters_norm_beta", numSamples);
      this->core->template addArgument<float>("update_parameters_norm_beta",
                                              static_cast<float>(this->workerConfig.trainingConfig.learningRate));
      this->core->template addArgument<float>("update_parameters_norm_beta", static_cast<float>(opt.beta1));
      this->core->template addArgument<float>("update_parameters_norm_beta", static_cast<float>(opt.beta2));
      this->core->template addArgument<float>("update_parameters_norm_beta", static_cast<float>(opt.epsilon));
      this->core->template addArgument<float>("update_parameters_norm_beta", bc1);
      this->core->template addArgument<float>("update_parameters_norm_beta", bc2);
    }
  } else {
    // SGD
    if (this->bufferManager.totalFilterSize > 0) {
      this->core->addKernel("update_parameters_filters", "update_parameters", this->bufferManager.totalFilterSize, 0);
      this->core->template addArgument<T>("update_parameters_filters", "cnn_filters");
      this->core->template addArgument<T>("update_parameters_filters", "cnn_accum_dFilters");
      this->core->template addArgument<ulong>("update_parameters_filters", static_cast<ulong>(0));
      this->core->template addArgument<ulong>("update_parameters_filters", this->bufferManager.totalFilterSize);
      this->core->template addArgument<ulong>("update_parameters_filters", numSamples);
      this->core->template addArgument<float>("update_parameters_filters",
                                              static_cast<float>(this->workerConfig.trainingConfig.learningRate));
    }

    if (this->bufferManager.totalBiasSize > 0) {
      this->core->addKernel("update_parameters_biases", "update_parameters", this->bufferManager.totalBiasSize, 0);
      this->core->template addArgument<T>("update_parameters_biases", "cnn_biases");
      this->core->template addArgument<T>("update_parameters_biases", "cnn_accum_dBiases");
      this->core->template addArgument<ulong>("update_parameters_biases", static_cast<ulong>(0));
      this->core->template addArgument<ulong>("update_parameters_biases", this->bufferManager.totalBiasSize);
      this->core->template addArgument<ulong>("update_parameters_biases", numSamples);
      this->core->template addArgument<float>("update_parameters_biases",
                                              static_cast<float>(this->workerConfig.trainingConfig.learningRate));
    }

    if (this->bufferManager.totalNormParamSize > 0) {
      this->core->addKernel("update_parameters_norm_gamma", "update_parameters", this->bufferManager.totalNormParamSize,
                            0);
      this->core->template addArgument<T>("update_parameters_norm_gamma", "cnn_norm_gamma");
      this->core->template addArgument<T>("update_parameters_norm_gamma", "cnn_accum_norm_dGamma");
      this->core->template addArgument<ulong>("update_parameters_norm_gamma", static_cast<ulong>(0));
      this->core->template addArgument<ulong>("update_parameters_norm_gamma", this->bufferManager.totalNormParamSize);
      this->core->template addArgument<ulong>("update_parameters_norm_gamma", numSamples);
      this->core->template addArgument<float>("update_parameters_norm_gamma",
                                              static_cast<float>(this->workerConfig.trainingConfig.learningRate));

      this->core->addKernel("update_parameters_norm_beta", "update_parameters", this->bufferManager.totalNormParamSize,
                            0);
      this->core->template addArgument<T>("update_parameters_norm_beta", "cnn_norm_beta");
      this->core->template addArgument<T>("update_parameters_norm_beta", "cnn_accum_norm_dBeta");
      this->core->template addArgument<ulong>("update_parameters_norm_beta", static_cast<ulong>(0));
      this->core->template addArgument<ulong>("update_parameters_norm_beta", this->bufferManager.totalNormParamSize);
      this->core->template addArgument<ulong>("update_parameters_norm_beta", numSamples);
      this->core->template addArgument<float>("update_parameters_norm_beta",
                                              static_cast<float>(this->workerConfig.trainingConfig.learningRate));
    }
  }

  // Running stats update (same for both Adam and SGD)
  if (this->bufferManager.totalNormParamSize > 0) {
    // Build a list of norm layer types
    std::vector<LayerType> normLayerTypes;

    for (const auto& layerConfig : this->workerConfig.layersConfig.cnnLayers) {
      if (layerConfig.type == LayerType::INSTANCENORM || layerConfig.type == LayerType::BATCHNORM) {
        normLayerTypes.push_back(layerConfig.type);
      }
    }

    // Get momentum from first norm layer config
    float momentum = 0.1f;

    for (const auto& layerConfig : this->workerConfig.layersConfig.cnnLayers) {
      if (layerConfig.type == LayerType::INSTANCENORM || layerConfig.type == LayerType::BATCHNORM) {
        const auto& bn = std::get<NormLayerConfig>(layerConfig.config);
        momentum = bn.momentum;
        break;
      }
    }

    for (ulong b = 0; b < this->bufferManager.normInfos.size(); b++) {
      // Skip BATCHNORM running stats update when using batch norm training path
      // (batch norm path updates running stats directly with batch-wide mean/var)
      if (skipBNRunningStats && normLayerTypes[b] == LayerType::BATCHNORM) {
        continue;
      }

      ulong normParamOffset = this->bufferManager.normInfos[b].paramOffset;
      ulong numChannels = this->bufferManager.normInfos[b].numChannels;
      std::string kernelId = "norm_update_running_stats_" + std::to_string(b);
      this->core->addKernel(kernelId, "norm_update_running_stats", numChannels, 0);
      this->core->template addArgument<T>(kernelId, "cnn_norm_running_mean");
      this->core->template addArgument<T>(kernelId, "cnn_norm_running_var");
      this->core->template addArgument<T>(kernelId, "cnn_accum_norm_batch_mean");
      this->core->template addArgument<T>(kernelId, "cnn_accum_norm_batch_var");
      this->core->template addArgument<ulong>(kernelId, normParamOffset);
      this->core->template addArgument<ulong>(kernelId, numChannels);
      this->core->template addArgument<float>(kernelId, momentum);
      this->core->template addArgument<ulong>(kernelId, numSamples);
    }
  }
}

//===================================================================================================================//
//-- Cross-sample batch normalization forward --//
//===================================================================================================================//

//===================================================================================================================//
//-- Cross-sample batch normalization forward --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addBatchNormForwardKernels(ulong layerIdx, ulong batchSize)
{
  const auto& cnnLayers = this->workerConfig.layersConfig.cnnLayers;
  const auto& bn = std::get<NormLayerConfig>(cnnLayers[layerIdx].config);
  std::string layerStr = std::to_string(layerIdx);

  // Compute shape and normIdx at this layer
  Shape3D currentShape = this->workerConfig.inputShape;
  ulong normIdx = 0;

  for (ulong i = 0; i < layerIdx; i++) {
    switch (cnnLayers[i].type) {
    case LayerType::CONV: {
      const auto& conv = std::get<ConvLayerConfig>(cnnLayers[i].config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
      ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
      currentShape = {conv.numFilters, outH, outW};
      break;
    }

    case LayerType::POOL: {
      const auto& pool = std::get<PoolLayerConfig>(cnnLayers[i].config);
      ulong outH = (currentShape.h - pool.poolH) / pool.strideY + 1;
      ulong outW = (currentShape.w - pool.poolW) / pool.strideX + 1;
      currentShape = {currentShape.c, outH, outW};
      break;
    }

    case LayerType::INSTANCENORM:
    case LayerType::BATCHNORM:
      normIdx++;
      break;
    default:
      break;
    }
  }

  ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;
  ulong actvInOffset = this->bufferManager.layerInfos[layerIdx].actvOffset;
  ulong actvOutOffset = this->bufferManager.layerInfos[layerIdx + 1].actvOffset;
  ulong sampleStride = this->bufferManager.totalActvSize;

  // Batch-wide mean
  ulong localWS = 256;
  ulong meanGlobalWS = currentShape.c * localWS;

  std::string meanId = "bn_batch_mean_l" + layerStr;
  this->core->addKernel(meanId, "norm_compute_mean", meanGlobalWS, 0, localWS);
  this->core->template addArgument<T>(meanId, "cnn_actvs");
  this->core->template addArgument<T>(meanId, "cnn_norm_batch_mean");
  this->core->template addArgument<ulong>(meanId, actvInOffset);
  this->core->template addArgument<ulong>(meanId, normParamOffset);
  this->core->template addArgument<ulong>(meanId, currentShape.c);
  this->core->template addArgument<ulong>(meanId, currentShape.h);
  this->core->template addArgument<ulong>(meanId, currentShape.w);
  this->core->template addArgument<ulong>(meanId, batchSize);
  this->core->template addArgument<ulong>(meanId, sampleStride);

  // Batch-wide variance
  std::string varId = "bn_batch_var_l" + layerStr;
  this->core->addKernel(varId, "norm_compute_var", meanGlobalWS, 0, localWS);
  this->core->template addArgument<T>(varId, "cnn_actvs");
  this->core->template addArgument<T>(varId, "cnn_norm_batch_mean");
  this->core->template addArgument<T>(varId, "cnn_norm_batch_var");
  this->core->template addArgument<ulong>(varId, actvInOffset);
  this->core->template addArgument<ulong>(varId, normParamOffset);
  this->core->template addArgument<ulong>(varId, currentShape.c);
  this->core->template addArgument<ulong>(varId, currentShape.h);
  this->core->template addArgument<ulong>(varId, currentShape.w);
  this->core->template addArgument<ulong>(varId, batchSize);
  this->core->template addArgument<ulong>(varId, sampleStride);

  // Batch-wide normalize (all N samples at once)
  ulong totalElements = batchSize * currentShape.size();
  std::string normId = "bn_batch_normalize_l" + layerStr;
  this->core->addKernel(normId, "norm_normalize", totalElements, 0);
  this->core->template addArgument<T>(normId, "cnn_actvs");
  this->core->template addArgument<T>(normId, "cnn_norm_xnorm");
  this->core->template addArgument<T>(normId, "cnn_norm_gamma");
  this->core->template addArgument<T>(normId, "cnn_norm_beta");
  this->core->template addArgument<T>(normId, "cnn_norm_batch_mean");
  this->core->template addArgument<T>(normId, "cnn_norm_batch_var");
  this->core->template addArgument<ulong>(normId, actvInOffset);
  this->core->template addArgument<ulong>(normId, actvOutOffset);
  this->core->template addArgument<ulong>(normId, actvInOffset); // xnorm offset
  this->core->template addArgument<ulong>(normId, normParamOffset);
  this->core->template addArgument<ulong>(normId, currentShape.c);
  this->core->template addArgument<ulong>(normId, currentShape.h);
  this->core->template addArgument<ulong>(normId, currentShape.w);
  this->core->template addArgument<ulong>(normId, batchSize);
  this->core->template addArgument<ulong>(normId, sampleStride);
  this->core->template addArgument<float>(normId, bn.epsilon);
}

//===================================================================================================================//
//-- Cross-sample batch normalization backward --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addBatchNormBackwardKernels(ulong layerIdx, ulong batchSize)
{
  const auto& cnnLayers = this->workerConfig.layersConfig.cnnLayers;
  const auto& bn = std::get<NormLayerConfig>(cnnLayers[layerIdx].config);
  std::string layerStr = std::to_string(layerIdx);

  // Compute shape and normIdx at this layer
  Shape3D currentShape = this->workerConfig.inputShape;
  ulong normIdx = 0;

  for (ulong i = 0; i < layerIdx; i++) {
    switch (cnnLayers[i].type) {
    case LayerType::CONV: {
      const auto& conv = std::get<ConvLayerConfig>(cnnLayers[i].config);
      ulong padY = SlidingStrategy::computePadding(conv.filterH, conv.slidingStrategy);
      ulong padX = SlidingStrategy::computePadding(conv.filterW, conv.slidingStrategy);
      ulong outH = (currentShape.h + 2 * padY - conv.filterH) / conv.strideY + 1;
      ulong outW = (currentShape.w + 2 * padX - conv.filterW) / conv.strideX + 1;
      currentShape = {conv.numFilters, outH, outW};
      break;
    }

    case LayerType::POOL: {
      const auto& pool = std::get<PoolLayerConfig>(cnnLayers[i].config);
      ulong outH = (currentShape.h - pool.poolH) / pool.strideY + 1;
      ulong outW = (currentShape.w - pool.poolW) / pool.strideX + 1;
      currentShape = {currentShape.c, outH, outW};
      break;
    }

    case LayerType::INSTANCENORM:
    case LayerType::BATCHNORM:
      normIdx++;
      break;
    default:
      break;
    }
  }

  ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;
  ulong gradOutOffset = this->bufferManager.layerInfos[layerIdx + 1].actvOffset;
  ulong gradInOffset = this->bufferManager.layerInfos[layerIdx].actvOffset;
  ulong xnormOffset = this->bufferManager.layerInfos[layerIdx].actvOffset;
  ulong sampleStride = this->bufferManager.totalActvSize;

  // Batch-wide dGamma/dBeta (N=batchSize, sampleStride=sampleStride)
  ulong localWS = 256;
  ulong dgGlobalWS = currentShape.c * localWS;
  std::string dgId = "bn_batch_dGammaBeta_l" + layerStr;
  this->core->addKernel(dgId, "norm_dGammaBeta", dgGlobalWS, 0, localWS);
  this->core->template addArgument<T>(dgId, "cnn_grads");
  this->core->template addArgument<T>(dgId, "cnn_norm_xnorm");
  this->core->template addArgument<T>(dgId, "cnn_norm_dGamma");
  this->core->template addArgument<T>(dgId, "cnn_norm_dBeta");
  this->core->template addArgument<ulong>(dgId, gradOutOffset);
  this->core->template addArgument<ulong>(dgId, xnormOffset);
  this->core->template addArgument<ulong>(dgId, normParamOffset);
  this->core->template addArgument<ulong>(dgId, currentShape.c);
  this->core->template addArgument<ulong>(dgId, currentShape.h);
  this->core->template addArgument<ulong>(dgId, currentShape.w);
  this->core->template addArgument<ulong>(dgId, batchSize);
  this->core->template addArgument<ulong>(dgId, sampleStride);

  // Batch-wide dInput (all N samples at once)
  ulong totalElements = batchSize * currentShape.size();
  std::string diId = "bn_batch_dInput_l" + layerStr;
  this->core->addKernel(diId, "norm_dInput", totalElements, 0);
  this->core->template addArgument<T>(diId, "cnn_grads");
  this->core->template addArgument<T>(diId, "cnn_norm_xnorm");
  this->core->template addArgument<T>(diId, "cnn_norm_gamma");
  this->core->template addArgument<T>(diId, "cnn_norm_dGamma");
  this->core->template addArgument<T>(diId, "cnn_norm_dBeta");
  this->core->template addArgument<T>(diId, "cnn_norm_batch_var");
  this->core->template addArgument<ulong>(diId, gradInOffset);
  this->core->template addArgument<ulong>(diId, gradOutOffset);
  this->core->template addArgument<ulong>(diId, xnormOffset);
  this->core->template addArgument<ulong>(diId, normParamOffset);
  this->core->template addArgument<ulong>(diId, currentShape.c);
  this->core->template addArgument<ulong>(diId, currentShape.h);
  this->core->template addArgument<ulong>(diId, currentShape.w);
  this->core->template addArgument<ulong>(diId, batchSize);
  this->core->template addArgument<ulong>(diId, sampleStride);
  this->core->template addArgument<float>(diId, bn.epsilon);
}

//===================================================================================================================//
//-- Cross-sample batch normalization running stats update --//
//===================================================================================================================//

template <typename T>
void GPUKernelBuilder<T>::addBatchNormRunningStatsUpdate(ulong batchSize)
{
  if (this->bufferManager.totalNormParamSize == 0)
    return;

  // Get momentum from first BN layer config
  float momentum = 0.1f;

  for (const auto& layerConfig : this->workerConfig.layersConfig.cnnLayers) {
    if (layerConfig.type == LayerType::BATCHNORM) {
      const auto& bn = std::get<NormLayerConfig>(layerConfig.config);
      momentum = bn.momentum;
      break;
    }
  }

  // For BatchNorm layers, the batch mean/var are already the true batch statistics
  // (computed by norm_compute_mean/var with N=batchSize). We update running stats directly.
  const auto& cnnLayers = this->workerConfig.layersConfig.cnnLayers;
  ulong normIdx = 0;

  for (ulong i = 0; i < cnnLayers.size(); i++) {
    if (cnnLayers[i].type == LayerType::BATCHNORM) {
      ulong normParamOffset = this->bufferManager.normInfos[normIdx].paramOffset;
      ulong numChannels = this->bufferManager.normInfos[normIdx].numChannels;
      std::string kernelId = "bn_update_running_stats_" + std::to_string(normIdx);
      this->core->addKernel(kernelId, "norm_update_running_stats", numChannels, 0);
      this->core->template addArgument<T>(kernelId, "cnn_norm_running_mean");
      this->core->template addArgument<T>(kernelId, "cnn_norm_running_var");
      this->core->template addArgument<T>(kernelId, "cnn_norm_batch_mean");
      this->core->template addArgument<T>(kernelId, "cnn_norm_batch_var");
      this->core->template addArgument<ulong>(kernelId, normParamOffset);
      this->core->template addArgument<ulong>(kernelId, numChannels);
      this->core->template addArgument<float>(kernelId, momentum);
      // For batch-wide stats, numSamples=1 since mean/var are already the batch average
      this->core->template addArgument<ulong>(kernelId, static_cast<ulong>(1));
    }

    if (cnnLayers[i].type == LayerType::INSTANCENORM || cnnLayers[i].type == LayerType::BATCHNORM) {
      normIdx++;
    }
  }
}

//===================================================================================================================//
// Explicit template instantiations.
//===================================================================================================================//

template class CNN::GPUKernelBuilder<int>;
template class CNN::GPUKernelBuilder<float>;
template class CNN::GPUKernelBuilder<double>;