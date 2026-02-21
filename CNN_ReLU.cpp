#include "CNN_ReLU.hpp"

#include <algorithm>

using namespace CNN;

//===================================================================================================================//

template <typename T>
Tensor3D<T> ReLU<T>::predict(const Tensor3D<T>& input) {
  Tensor3D<T> output(input.shape);

  for (ulong i = 0; i < input.data.size(); i++) {
    output.data[i] = std::max(static_cast<T>(0), input.data[i]);
  }

  return output;
}

//===================================================================================================================//

template <typename T>
Tensor3D<T> ReLU<T>::backpropagate(const Tensor3D<T>& dOut, const Tensor3D<T>& input) {
  Tensor3D<T> dInput(input.shape);

  for (ulong i = 0; i < input.data.size(); i++) {
    dInput.data[i] = (input.data[i] > static_cast<T>(0)) ? dOut.data[i] : static_cast<T>(0);
  }

  return dInput;
}

//===================================================================================================================//

// Explicit template instantiations
template class CNN::ReLU<int>;
template class CNN::ReLU<double>;
template class CNN::ReLU<float>;

