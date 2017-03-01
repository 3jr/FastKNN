#include <iostream>
#include <array>
#include <string>
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

#define BOOST_TEST_MODULE FastKNN-Boost-Tests
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "kdtree.hpp"

//BOOST_AUTO_TEST_SUITE(simple_tests)

BOOST_AUTO_TEST_CASE(log2ceil) {
  BOOST_CHECK_EQUAL(kdtree::log2ceil(1024), 10);
  BOOST_CHECK_EQUAL(kdtree::log2ceil(1023), 10);
  BOOST_CHECK_EQUAL(kdtree::log2ceil(1025), 11);
  BOOST_CHECK_EQUAL(kdtree::log2ceil(14), 4);
  BOOST_CHECK_EQUAL(kdtree::log2ceil(15), 4);
  BOOST_CHECK_EQUAL(kdtree::log2ceil(16), 4);
  BOOST_CHECK_EQUAL(kdtree::log2ceil(17), 5);
}

BOOST_AUTO_TEST_CASE(log2floor) {
  BOOST_CHECK_EQUAL(kdtree::log2floor(1024), 10);
  BOOST_CHECK_EQUAL(kdtree::log2floor(1023), 9);
  BOOST_CHECK_EQUAL(kdtree::log2floor(1025), 10);
  BOOST_CHECK_EQUAL(kdtree::log2floor(14), 3);
  BOOST_CHECK_EQUAL(kdtree::log2floor(15), 3);
  BOOST_CHECK_EQUAL(kdtree::log2floor(16), 4);
  BOOST_CHECK_EQUAL(kdtree::log2floor(17), 4);
}

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

// in a full grid get the direct neighbours
std::vector<int> get_neighbours(int D, int n, int p) {
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


template<Size dims>
void run_knn(int pointsPerDim) {
  int k = 2 * dims + 1;
  auto points = gen_full_grid<dims>(pointsPerDim);
  std::cout << "input generated\n";
  auto tree = kdtree::buildKdTree(points);
  std::cout << "tree (depth:" << tree.depth << ") build done: \n";
  //for (int s = 1; s < tree.divisions.size() - 1; s *= 2) {
  //  for (int i = 0; i < s; ++i) {
  //    auto d = tree.divisions[s - 1 + i];
  //    std::cout << d.dim << '(' << d.p << ") | ";
  //  }
  //  std::cout << '\n';
  //}
  //std::cout << '\n';
  for (int i = 0; i < points.size(); ++i) {
    //if (i != 1) { continue; }
    std::cout << i << " ";
    if (i > 10) { break; }
    auto p = points[i];
    auto n1 = kdtree::knn(tree, points, k, p);
    auto n2 = simple_knn(points, k, p);
    auto n3 = get_neighbours(dims, pointsPerDim, i);
    std::sort(n1.begin(), n1.end());
    std::sort(n2.begin(), n2.end());
    std::sort(n3.begin(), n3.end());
    BOOST_CHECK(std::includes(n1.begin(), n1.end(), n3.begin(), n3.end()));
    if (!std::includes(n1.begin(), n1.end(), n3.begin(), n3.end())) {
      std::cout << "   " << n1 << " " << i << '\n';
      std::cout << "   " << n2 << " " << i << '\n';
      std::cout << "   " << n3 << " " << i << '\n';
    }
    if (n3.size() <  n1.size()) {
      continue; // we can't check equality b/c it is not unique
    }
    BOOST_CHECK(n1 == n2);
    if (n1 != n2) {
      std::cout << "   " << n1 << " " << i << '\n';
      std::cout << "   " << n2 << " " << i << '\n';
    }
  }
}

BOOST_AUTO_TEST_CASE(full_grid_var1) {
  run_knn<3>(4);
}

// 0  1  2  3
// 4  5  6  7
// 8  9 10 11
//12 13 14 15

BOOST_AUTO_TEST_CASE(full_grid_var) {
  for (int i = 1; i < 5; ++i) {
    run_knn<4>(i);
    run_knn<3>(i);
    run_knn<2>(i);
    run_knn<1>(i);
  }
}

BOOST_AUTO_TEST_CASE(full_grid) {
  run_knn<1>(5);
  run_knn<2>(5);
  run_knn<3>(5);
  run_knn<4>(5);
  run_knn<5>(5);
  run_knn<6>(5);
  run_knn<7>(5);
  run_knn<8>(5);
  run_knn<9>(5);
  run_knn<10>(5);
}

BOOST_AUTO_TEST_CASE(build_huge_tree) {
  auto points = gen_full_grid<9>(5);
  auto tree = kdtree::buildKdTree(points);
  kdtree::printTreeDivisions<9>(tree);
}

//BOOST_AUTO_TEST_SUITE_END()
