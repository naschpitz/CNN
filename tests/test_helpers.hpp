#pragma once

#include "CNN_Types.hpp"
#include "CNN_Conv2D.hpp"
#include "CNN_ReLU.hpp"
#include "CNN_Pool.hpp"
#include "CNN_BatchNorm.hpp"
#include "CNN_InstanceNorm.hpp"
#include "CNN_Flatten.hpp"
#include "CNN_Core.hpp"
#include "CNN_CoreConfig.hpp"
#include "CNN_Sample.hpp"
#include "CNN_SlidingStrategy.hpp"
#include "CNN_LayersConfig.hpp"

#include <ANN_ActvFunc.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>
#include <stdexcept>

extern int testsPassed;
extern int testsFailed;

// clang-format off
#define CHECK(cond, msg) do { \
  if (!(cond)) { \
    std::cerr << "FAIL: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
    testsFailed++; \
  } else { \
    testsPassed++; \
  } \
} while(0)

#define CHECK_NEAR(a, b, tol, msg) CHECK(std::fabs((a) - (b)) < (tol), msg)

#define CHECK_THROWS(expr, msg) do { \
  bool threw = false; \
  try { expr; } catch (...) { threw = true; } \
  CHECK(threw, msg); \
} while(0)
// clang-format on

// Helper: create gradient-filled tensor (values from lo to hi across spatial dims)
// This produces diverse CNN features, avoiding the uniform-input problem where
// all ANN inputs are identical and random weight initialization can stall learning.
template <typename T>
CNN::Tensor3D<T> makeGradientInput(CNN::Shape3D shape, T lo = T(0.5), T hi = T(1.0))
{
  CNN::Tensor3D<T> t(shape);
  ulong total = shape.c * shape.h * shape.w;

  for (ulong c = 0; c < shape.c; ++c)

    for (ulong h = 0; h < shape.h; ++h)

      for (ulong w = 0; w < shape.w; ++w) {
        ulong idx = c * shape.h * shape.w + h * shape.w + w;
        t.at(c, h, w) = lo + (hi - lo) * T(idx) / T(total > 1 ? total - 1 : 1);
      }

  return t;
}
