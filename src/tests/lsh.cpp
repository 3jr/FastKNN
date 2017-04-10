#include <iostream>
#include <array>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <tuple>
#include <queue>

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "lsh.hpp"
#include "tests/common.hpp"

BOOST_AUTO_TEST_SUITE(lsh_tests)

template<Size dims, Size K>
void run_knn_minimal(int pointsPerDim, Real r, Size L) {
  int k = 2 * dims + 1;
  auto points = gen_full_grid<dims>(pointsPerDim);
  std::cout << "generate_hashes_start [Points: " << points.size() << "] ... ";
  auto hashes = lsh::generate_hashes<dims, K>(points, r, L);
  std::cout << "gen_done\n";
  int not_all_adjacents = 0;
  for (int i = 0; i < points.size(); ++i) {
    if (i > 10) { break; }
    auto p = points[i];
    auto n1 = lsh::knn<dims, K>(hashes, points, p, r, k);
    auto n3 = get_neighbours(dims, pointsPerDim, i);
    std::sort(n1.begin(), n1.end());
    std::sort(n3.begin(), n3.end());
    if (!std::includes(n1.begin(), n1.end(), n3.begin(), n3.end())) {
      ++not_all_adjacents;
    }
  }
  std::cout << "[Not all adjacents found: " << not_all_adjacents << " (tweak params for better results)]\n";
}

template<Size dims, Size K>
void run_knn(int pointsPerDim, Real r, Size L) {
  int k = 2 * dims + 1;
  auto points = gen_full_grid<dims>(pointsPerDim);
  auto hashes = lsh::generate_hashes<dims, K>(points, r, L);
  for (int i = 0; i < points.size(); ++i) {
    //if (i != 1) { continue; }
    std::cout << i << " ";
    if (i > 10) { break; }
    auto p = points[i];
    auto n1 = lsh::knn<dims, K>(hashes, points, p, r, k);
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
    //if (n3.size() <  n1.size()) {
    //  continue; // we can't check equality b/c it is not unique
    //}
    //BOOST_CHECK(n1 == n2);
    //if (n1 != n2) {
    //  std::cout << "   " << n1 << " " << i << '\n';
    //  std::cout << "   " << n2 << " " << i << '\n';
    //}
  }
}

BOOST_AUTO_TEST_CASE(demo) {
  run_knn_minimal<2, 2>(5, 1, 5);
}

BOOST_AUTO_TEST_CASE(simple) {
  run_knn_minimal<2, 2>(5, 1, 5);
  run_knn_minimal<2, 2>(50, 1, 50);
  run_knn_minimal<2, 2>(500, 1, 500);
  run_knn_minimal<2, 2>(5000, 1, 500);
}

BOOST_AUTO_TEST_SUITE_END()
