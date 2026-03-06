# Proper Batch Normalization Implementation

## Current State (Instance Normalization)

The CNN currently processes samples **one at a time** through the entire network. This means
that during training, each batch normalization layer computes mean and variance from only the
spatial dimensions (H × W) of a single image per channel — effectively performing **Instance
Normalization**, not Batch Normalization.

As of the current fix, inference also uses per-image spatial statistics (instance norm) to
match training behavior. This resolved a critical train-test mismatch where inference used
population running statistics that were inconsistent with the per-image statistics used during
training, causing near-random accuracy on ISIC (16% → 72% after fix).

### Files Changed for Instance Norm Fix

- `CNN_GPUKernelBuilder.cpp`: `setupPredictKernels()` calls `addPropagateKernels(true)` so
  BN computes per-image spatial stats during inference.
- `CNN_CoreCPUWorker.cpp`: predict path calls `BatchNorm::propagate` with `training=true`
  using temporary buffers, so it also uses per-image stats at inference.

## Why Proper Batch Normalization Would Be Better

True batch normalization (as implemented in PyTorch/TensorFlow) normalizes each channel across
**all images in the mini-batch simultaneously**:

- **PyTorch BN** (batch size N): for each channel, compute mean/var across N × H × W elements.
- **Current implementation** (instance norm): for each channel, compute mean/var across 1 × H × W.

### Advantages of True BN Over Instance Norm

1. **Better classification accuracy**: BN preserves population-relative magnitude information
   that is discriminative for classification. Instance norm normalizes this away. Published
   research shows BN outperforms IN by ~2–4 percentage points on ImageNet.

2. **Regularization effect**: batch statistics introduce noise (different batches have different
   stats), which acts as a regularizer and improves generalization.

3. **Correct running statistics**: with true BN, running mean/var would track actual population
   statistics using the law of total variance, making them usable at inference.

## Architecture Change Required

### Current Flow (Image-by-Image)

```
for each image in batch:
    image → Conv1 → BN1 → ReLU1 → Conv2 → BN2 → ReLU2 → ... → output
    accumulate gradients
update weights
```

Each image goes through ALL layers before the next image is processed. BN only sees one image.

### Required Flow (Layer-by-Layer)

```
for each layer in network:
    pass ALL images in batch through this layer
    if layer is BN:
        compute mean/var across all N images (N × H × W per channel)
        normalize all N images using these batch statistics
        store normalized values for backprop
```

All images flow through each layer together, so BN sees the full batch at each layer.

## Implementation Plan

### Phase 1: Forward Pass Restructuring

1. **Allocate batch activation buffers**: instead of a single activation buffer that gets
   overwritten per image, allocate storage for all N images' activations at each layer.
   - GPU memory cost: for ISIC (225×300×3) with 256 channels and batch size 128, each layer
     could need several GB. May need to limit batch size or use gradient checkpointing.

2. **Layer-by-layer forward pass**:
   - For each layer, process all N images through that layer.
   - For non-BN layers (Conv, ReLU, Pool): these are per-image operations, so they can still
     be parallelized independently per image.
   - For BN layers: compute batch statistics across all N images, then normalize all N images.

3. **BN batch statistics computation** (GPU kernel):
   ```
   // Pass 1: compute per-image spatial mean for each channel
   // Pass 2: reduce across images to get batch mean
   // Pass 3: compute per-image spatial variance around batch mean
   // Pass 4: reduce across images to get batch variance
   // Pass 5: normalize all images using batch mean/var
   ```

### Phase 2: Backward Pass Restructuring

4. **BN backpropagation**: the gradient through BN depends on batch statistics, so all images'
   gradients must be computed together at each BN layer. The standard BN backward pass requires:
   - `dxnorm = dy * gamma`
   - `dvar = sum(dxnorm * (x - mean) * -0.5 * (var + eps)^(-3/2))` — summed across batch
   - `dmean = sum(dxnorm * -1/sqrt(var + eps)) + dvar * sum(-2*(x-mean))/N` — summed across batch
   - `dx = dxnorm / sqrt(var + eps) + dvar * 2*(x-mean)/N + dmean/N`

5. **Gradient accumulation**: gradients for Conv filters/biases and BN gamma/beta are
   accumulated across the batch as before, but now computed layer-by-layer.

### Phase 3: Running Statistics Fix

6. **Correct running stats update**: with true batch statistics available, the EMA update
   becomes straightforward:
   ```
   running_mean = (1 - momentum) * running_mean + momentum * batch_mean
   running_var  = (1 - momentum) * running_var  + momentum * batch_var
   ```
   These running stats would be correct and usable at inference.

7. **Inference mode**: use running statistics (standard BN behavior) instead of per-image
   instance norm. This is the default behavior in PyTorch `model.eval()`.

### Phase 4: Multi-GPU Considerations

8. **Cross-GPU batch statistics**: if the batch is split across GPUs, each GPU only sees a
   subset. Options:
   - **SyncBatchNorm**: synchronize BN statistics across GPUs (adds communication overhead).
   - **Per-GPU BN**: each GPU computes BN stats from its subset (simpler, but effective batch
     size per GPU is smaller).
   - PyTorch uses per-GPU BN by default, SyncBatchNorm is opt-in.

## Memory Considerations

| Component | Current (1 image) | Batch BN (N=128, 256ch, 56×75) |
|-----------|-------------------|-------------------------------|
| Activations per layer | ~4 MB | ~512 MB |
| All 8 BN layers | ~32 MB | ~4 GB |
| Backprop storage | ~32 MB | ~4 GB |

For large images and batch sizes, memory could be prohibitive. Mitigations:
- Reduce batch size (e.g., 16–32 instead of 128)
- Gradient checkpointing (recompute activations during backward instead of storing)
- Mixed precision (float16 for activations)

## Alternative: Group Normalization

If memory constraints make true BN impractical, **Group Normalization** is a viable alternative:
- Divides channels into groups, normalizes within each group per image
- No batch dependency — works with any batch size
- Performance close to BN on many tasks
- Instance Norm is a special case of Group Norm (groups = num_channels)
- Could be implemented without restructuring the forward pass

## Priority

Medium-to-low. The instance norm fix provides correct train-test consistency. The accuracy
gap (estimated 2–4%) may be acceptable for many use cases. The architectural restructuring
is a significant effort that should be weighed against other improvements.

