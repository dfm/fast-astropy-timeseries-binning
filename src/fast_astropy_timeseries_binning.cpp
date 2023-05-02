#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <cmath>

namespace nb = nanobind;

struct mean {
  template <typename T>
  static inline T evaluate(size_t start, size_t end, const T* data) {
    if (end <= start) return T(0.0);
    T result = T(0.0);
    size_t count = 0;
    for (size_t m = start; m < end; ++m) {
      auto value = data[m];
      if (isnan(value)) continue;
      result += value;
      count++;
    }
    return result / static_cast<T>(count);
  }
};

struct ivar {
  template <typename T>
  static inline T evaluate(size_t start, size_t end, const T* data) {
    if (end <= start) return T(0.0);
    T result = T(0.0);
    for (size_t m = start; m < end; ++m) {
      auto value = data[m];
      if (isnan(value)) continue;
      result += T(1.0) / (value * value);
    }
    return T(1.0) / sqrt(result);
  }
};

// Using Welford's algorithm
struct rms {
  template <typename T>
  static inline T evaluate(size_t start, size_t end, const T* data) {
    if (end <= start + 1) return T(0.0);
    size_t count = 0;
    T mean = T(0.0), carry = T(0.0);
    for (size_t m = start; m < end; ++m) {
      auto value = data[m];
      if (isnan(value)) continue;
      auto delta = value - mean;
      mean += delta / static_cast<T>(count);
      carry += delta * (value - mean);
    }
    return sqrt(carry) / static_cast<T>(count);
  }
};

template <typename T, typename F>
void reduceat(nb::ndarray<T, nb::shape<nb::any>> array,
              nb::ndarray<size_t, nb::shape<nb::any>> indices,
              nb::ndarray<T, nb::shape<nb::any>> out) {
  for (size_t n = 0; n < indices.shape(0); ++n) {
    size_t top =
        (n < indices.shape(0) - 1) ? indices(n + 1) : array.shape(0) - 1;
    out(n) = F::evaluate(indices(n), top, array.data());
  }
}

NB_MODULE(_fast_astropy_timeseries_binning, m) {
  m.def("reducemean_f32", &reduceat<float, mean>);
  m.def("reducemean_f64", &reduceat<double, mean>);
  m.def("reduceivar_f32", &reduceat<float, ivar>);
  m.def("reduceivar_f64", &reduceat<double, ivar>);
  m.def("reducerms_f32", &reduceat<float, rms>);
  m.def("reducerms_f64", &reduceat<double, rms>);
}
