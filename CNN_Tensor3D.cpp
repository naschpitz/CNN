#include "CNN_Tensor3D.hpp"

#include <algorithm>

using namespace CNN;

//===================================================================================================================//
//-- Constructors --//
//===================================================================================================================//

template <typename T>
Tensor3D<T>::Tensor3D(const Shape3D& shape) : shape(shape),
                                              data(shape.size(), static_cast<T>(0))
{
}

//===================================================================================================================//

template <typename T>
Tensor3D<T>::Tensor3D(const Shape3D& shape, T value) : shape(shape),
                                                       data(shape.size(), value)
{
}

//===================================================================================================================//
//-- Element access --//
//===================================================================================================================//

template <typename T>
T& Tensor3D<T>::at(ulong c, ulong h, ulong w)
{
  return data[c * shape.h * shape.w + h * shape.w + w];
}

//===================================================================================================================//

template <typename T>
const T& Tensor3D<T>::at(ulong c, ulong h, ulong w) const
{
  return data[c * shape.h * shape.w + h * shape.w + w];
}

//===================================================================================================================//
//-- Size / resize --//
//===================================================================================================================//

template <typename T>
ulong Tensor3D<T>::size() const
{
  return data.size();
}

//===================================================================================================================//

template <typename T>
void Tensor3D<T>::resize(const Shape3D& newShape)
{
  shape = newShape;
  data.resize(shape.size(), static_cast<T>(0));
}

//===================================================================================================================//

template <typename T>
void Tensor3D<T>::resize(const Shape3D& newShape, T value)
{
  shape = newShape;
  data.assign(shape.size(), value);
}

//===================================================================================================================//

template <typename T>
void Tensor3D<T>::fill(T value)
{
  std::fill(data.begin(), data.end(), value);
}

//===================================================================================================================//

// Explicit template instantiations
template class CNN::Tensor3D<int>;
template class CNN::Tensor3D<double>;
template class CNN::Tensor3D<float>;
