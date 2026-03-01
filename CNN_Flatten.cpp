#include "CNN_Flatten.hpp"

using namespace CNN;

//===================================================================================================================//

template <typename T>
Tensor1D<T> Flatten<T>::predict(const Tensor3D<T>& input)
{
  // Simply return the flat data - already stored in NCHW order
  return input.data;
}

//===================================================================================================================//

template <typename T>
Tensor3D<T> Flatten<T>::backpropagate(const Tensor1D<T>& dOut, const Shape3D& inputShape)
{
  Tensor3D<T> dInput(inputShape);
  dInput.data = dOut;
  return dInput;
}

//===================================================================================================================//

// Explicit template instantiations
template class CNN::Flatten<int>;
template class CNN::Flatten<double>;
template class CNN::Flatten<float>;
