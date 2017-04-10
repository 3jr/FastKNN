#include <iostream>
#include <string>

#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "kdtree.hpp"
#include "tests/common.hpp"

BOOST_AUTO_TEST_SUITE(kdtree_tests)

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

BOOST_AUTO_TEST_CASE(demo) {
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

BOOST_AUTO_TEST_SUITE_END()
