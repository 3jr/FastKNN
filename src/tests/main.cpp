#include "iostream"

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


template<std::array<double, 1>::size_type dims>
inline void gen_full_grid(std::vector<std::array<double, dims>> &ps,
    std::array<double, dims> &p, int d, int n) {
  if (d == dims) {
    ps.push_back(p);
  } else {
    for (int i = 0; i < n; ++i) {
      p[d] = static_cast<double>(i);
      gen_full_grid(ps, p, d + 1, n);
    }
  }
}


template<std::array<double, 1>::size_type dims>
void run_knn(int pointsPerDim, int k) {
  std::vector<std::array<double, dims>> points;
  std::array<double, dims> dummy;
  gen_full_grid(points, dummy, 0, pointsPerDim);
  std::cout << "input generated\n";
  kdtree::knn(points, k);
}


BOOST_AUTO_TEST_CASE(full_grid) {
  for (int i = 1; i < 5; ++i)
  for (int k = 1; k < 5; ++k) {
    run_knn<4>(i, 2 * 4);
    run_knn<3>(i, 2 * 3);
    run_knn<2>(i, 2 * 2);
    run_knn<1>(i, 2 * 1);
  }
}

//BOOST_AUTO_TEST_SUITE_END()
