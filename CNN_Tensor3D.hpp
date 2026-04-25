#ifndef CNN_TENSOR3D_HPP
#define CNN_TENSOR3D_HPP

#include "CNN_Shape3D.hpp"

#include <vector>

//===================================================================================================================//

namespace CNN
{
  // 3D tensor stored as flat vector in NCHW order: data[c * H * W + h * W + w]
  template <typename T>
  class Tensor3D
  {
    public:
      Shape3D shape;
      std::vector<T> data;

      //-- Constructors --//
      Tensor3D() = default;
      Tensor3D(const Shape3D& shape);
      Tensor3D(const Shape3D& shape, T value);

      //-- Element access --//
      T& at(ulong c, ulong h, ulong w);
      const T& at(ulong c, ulong h, ulong w) const;

      //-- Size / resize --//
      ulong size() const;
      void resize(const Shape3D& newShape);
      void resize(const Shape3D& newShape, T value);
      void fill(T value);
  };
}

//===================================================================================================================//

#endif // CNN_TENSOR3D_HPP
