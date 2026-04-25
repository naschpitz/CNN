#ifndef CNN_TYPES_HPP
#define CNN_TYPES_HPP

#include "CNN_Shape3D.hpp"
#include "CNN_Tensor3D.hpp"

#include <functional>
#include <sys/types.h>
#include <vector>

//===================================================================================================================//

namespace CNN
{
  // 1D tensor (used for flattened output)
  template <typename T>
  using Tensor1D = std::vector<T>;

  // Input to the CNN is a 3D tensor
  template <typename T>
  using Input = Tensor3D<T>;

  template <typename T>
  using Inputs = std::vector<Input<T>>;

  // Lazy supplier for streaming predict(): given a batch size and a 0-based
  // batch index, returns the corresponding chunk of inputs. The last batch
  // may be shorter than batchSize. Mirrors SampleProvider used by train/test
  // and lets callers calibrate / score arbitrarily large image sets without
  // holding everything in host memory.
  template <typename T>
  using InputProvider = std::function<Inputs<T>(ulong batchSize, ulong batchIndex)>;

  // Output from the CNN is a 1D vector (from dense layers)
  template <typename T>
  using Output = std::vector<T>;

  // Pre-activation values (z) of the ANN dense head's last layer.
  // Useful for OOD-detection scores (max-logit, logit-norm, free-energy)
  // that softmax discards.
  template <typename T>
  using Logits = std::vector<T>;
}

//===================================================================================================================//

#endif // CNN_TYPES_HPP
