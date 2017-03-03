#pragma once

#include <array>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <tuple>
#include <queue>

template<typename Stream, typename T>
Stream& operator << (Stream& s, std::vector<T>& v) {
  s << "[";
  for (const auto &e : v) {
    s << e << ",";
  }
  s << "]";
  return s;
}


using Real = double;
using Size = std::array<double, 1>::size_type;
using std::get;

inline double  square(double v) { return v * v; }

template<Size dims>
double distSquared(std::array<double, dims> p1, std::array<double, dims> p2) {
  double d = 0;
  for (auto i = 0; i < dims; ++i) {
    d += square(p1[i] - p2[i]);
  }
  return d;
}

// in a full grid get the direct neighbours (up to 2*D many)
inline std::vector<int> get_neighbours(int D, int n, int p) {
  // s == n^d
  // s * n == n^(d+1)
  std::vector<int> result{p};
  for (int d = 0, s = 1; d < D - 1; ++d, s *= n) {
    auto l = p - s;
    if (l >= 0
        && l / (s * n) == p / (s * n)) { // "same level"
      result.push_back(l);
    }
    auto r = p + s;
    if (r / (s * n) == p / (s * n)) { // "same level"
      result.push_back(r);
    }
  }
  return result;
}

template<Size DIMS>
std::vector<Size> simple_knn(std::vector<std::array<double, DIMS>> points, int k, std::array<double, DIMS> p) {
  using E = std::tuple<double, Size>;
  std::vector<E> queueContainer{};
  queueContainer.reserve(k);
  auto compare = [](const E &e1, const E &e2){ return get<double>(e1) < get<double>(e2); };
  std::priority_queue<E, std::vector<E>, decltype(compare)> nearest{compare, queueContainer};
  for (int i = 0; i < k && i < points.size(); ++i) {
    auto d = distSquared(points[i], p);
    nearest.push({d, i});
  }
  for (int i = k; i < points.size(); ++i) {
    auto d = distSquared(points[i], p);
    if (d < get<double>(nearest.top())) {
      nearest.pop();
      nearest.push({d, i});
    }
  }
  std::vector<Size> result{};
  result.reserve(k);
  while (nearest.size() > 0) {
    result.push_back(get<Size>(nearest.top()));
    nearest.pop();
  }
  return result;
}

template<Size dims>
inline void gen_full_grid_impl(std::vector<std::array<double, dims>> &ps,
    std::array<double, dims> &p, int d, int n) {
  if (d == dims) {
    ps.push_back(p);
  } else {
    for (int i = 0; i < n; ++i) {
      p[d] = static_cast<double>(i);
      gen_full_grid_impl(ps, p, d + 1, n);
    }
  }
}

template<Size dims>
inline auto gen_full_grid(int pointsPerDim) {
  std::vector<std::array<double, dims>> points;
  std::array<double, dims> dummy;
  gen_full_grid_impl(points, dummy, 0, pointsPerDim);
  return points;
}

