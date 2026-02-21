#ifndef CNN_TYPES_HPP
#define CNN_TYPES_HPP

#include <cstddef>
#include <sys/types.h>
#include <vector>

//===================================================================================================================//

namespace CNN {
  // Shape of a 3D tensor (Channels, Height, Width) - NCHW layout
  struct Shape3D {
    ulong c;  // channels
    ulong h;  // height
    ulong w;  // width

    ulong size() const { return c * h * w; }

    bool operator==(const Shape3D& other) const {
      return c == other.c && h == other.h && w == other.w;
    }

    bool operator!=(const Shape3D& other) const {
      return !(*this == other);
    }
  };

  // 3D tensor stored as flat vector in NCHW order: data[c * H * W + h * W + w]
  template <typename T>
  struct Tensor3D {
    Shape3D shape;
    std::vector<T> data;

    Tensor3D() = default;

    Tensor3D(const Shape3D& shape) : shape(shape), data(shape.size(), static_cast<T>(0)) {}

    Tensor3D(const Shape3D& shape, T value) : shape(shape), data(shape.size(), value) {}

    T& at(ulong c, ulong h, ulong w) {
      return data[c * shape.h * shape.w + h * shape.w + w];
    }

    const T& at(ulong c, ulong h, ulong w) const {
      return data[c * shape.h * shape.w + h * shape.w + w];
    }

    ulong size() const { return data.size(); }

    void resize(const Shape3D& newShape) {
      shape = newShape;
      data.resize(shape.size(), static_cast<T>(0));
    }

    void resize(const Shape3D& newShape, T value) {
      shape = newShape;
      data.assign(shape.size(), value);
    }

    void fill(T value) {
      std::fill(data.begin(), data.end(), value);
    }
  };

  // 1D tensor (used for flattened output)
  template <typename T>
  using Tensor1D = std::vector<T>;

  // Input to the CNN is a 3D tensor
  template <typename T>
  using Input = Tensor3D<T>;

  // Output from the CNN is a 1D vector (from dense layers)
  template <typename T>
  using Output = std::vector<T>;
}

//===================================================================================================================//

#endif // CNN_TYPES_HPP

