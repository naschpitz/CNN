# CNN Feature Roadmap

Features missing compared to modern CNN implementations, ordered by impact.

---

## 🔴 High Impact

### 1. Learning Rate Scheduling
Fixed learning rate is a major limitation. Options:
- **Step decay** — reduce LR by a factor every N epochs
- **Cosine annealing** — smoothly decays LR following a cosine curve
- **Warmup** — ramp up LR over the first few epochs (important for Adam with large batches)

### 2. Residual / Skip Connections
Sequential-only architecture limits effective depth. ResNet-style residual connections (`output = F(x) + x`) enable training of much deeper networks (50–100+ layers) by preventing gradient vanishing. Requires changes to the layer pipeline, backprop, and GPU kernel builder. A `ResidualBlock` meta-layer wrapping sub-layers is the cleanest approach.

### 3. Global Average Pooling Layer
Currently simulated with a manually-sized average pool. A proper GAP layer would automatically adapt to whatever spatial size comes in, making architectures more flexible and less error-prone when changing input sizes.

---

## 🟡 Medium Impact

### 4. Weight Decay / L2 Regularization
Not currently implemented in the optimizer. Standard regularizer in every modern CNN. Implementation: `w = w - lr * (grad + λ * w)`.

### 5. More Activation Functions
Currently available: ReLU, Sigmoid, Tanh, Softmax. Missing:
- **LeakyReLU / PReLU** — avoids "dying ReLU" problem
- **GELU / SiLU (Swish)** — used in modern architectures (EfficientNet, Transformers), smoother than ReLU

### 6. Depthwise Separable Convolutions
Standard convolutions are expensive. Depthwise separable convolutions (MobileNet-style) split into depthwise conv + 1×1 pointwise conv, reducing parameters by ~8–9×. Enables larger/deeper networks within the same compute budget.

### 7. Gradient Clipping
Prevents gradient explosion during training, especially important with deep networks. Cap the gradient norm before applying updates.

### 8. 1×1 Convolutions (Bottleneck Layers)
The engine technically supports `filterH: 1, filterW: 1`, but the architectural pattern of bottleneck blocks (1×1 reduce → 3×3 conv → 1×1 expand) requires skip connections to be useful.

---

## 🟢 Low Impact

### 9. Dropout Variants for Conv Layers
Current dropout only applies to dense layers. For conv layers:
- **Spatial Dropout** — drops entire feature maps
- **DropBlock** — drops contiguous regions of feature maps

### 10. More Loss Functions
- **Focal Loss** — handles class imbalance better than weighted cross-entropy (down-weights easy examples)
- **Label Smoothing** — small modification to cross-entropy, typically gives 0.5–1% accuracy boost

### 11. Multi-scale / Feature Pyramid
Processing features at multiple resolutions simultaneously (FPN-style). Very impactful for detection/segmentation, less critical for pure classification.

### 12. Grouped Convolutions
Split channels into groups and convolve independently. Used in ResNeXt and EfficientNet for better accuracy/compute tradeoff.

---

## Priority Summary

| Priority | Feature                        | Effort | Impact |
|----------|--------------------------------|--------|--------|
| 1        | Learning rate scheduling       | Low    | High   |
| 2        | Weight decay in optimizer      | Low    | Medium |
| 3        | LeakyReLU / GELU              | Low    | Medium |
| 4        | Gradient clipping              | Low    | Medium |
| 5        | Residual / skip connections    | High   | High   |
| 6        | Global average pooling layer   | Low    | Medium |
| 7        | Focal loss / label smoothing   | Low    | Medium |
| 8        | Depthwise separable conv       | Medium | Medium |

