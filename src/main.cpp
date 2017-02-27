#include <iostream>

#include "kdtree.hpp"

constexpr std::array<double, 1>::size_type dims = 5;

void gen_points(std::vector<std::array<double, dims>> &ps,
    std::array<double, dims> &p, int d, int n) {
  if (d == dims) {
    ps.push_back(p);
  } else {
    for (int i = 0; i < n; ++i) {
      p[d] = static_cast<double>(i);
      gen_points(ps, p, d + 1, n);
    }
  }
}

void f() {
  std::vector<std::array<double, dims>> points;
  std::array<double, dims> dummy;
  gen_points(points, dummy, 0, 5);
  std::cout << "input generated\n";
  kdtree::knn(points, 5);
}

#define assert(a) std::cout << (a) << '\n'

void testLog2() {
  assert(kdtree::log2ceil(1024) == 10);
  assert(kdtree::log2ceil(1023) == 10);
  assert(kdtree::log2ceil(1025) == 11);
  assert(kdtree::log2ceil(14) == 4);
  assert(kdtree::log2ceil(15) == 4);
  assert(kdtree::log2ceil(16) == 4);
  assert(kdtree::log2ceil(17) == 5);

  assert(kdtree::log2floor(1024) == 10);
  assert(kdtree::log2floor(1023) == 9);
  assert(kdtree::log2floor(1025) == 10);
  assert(kdtree::log2floor(14) == 3);
  assert(kdtree::log2floor(15) == 3);
  assert(kdtree::log2floor(16) == 4);
  assert(kdtree::log2floor(17) == 4);
}

int main() {
  f();
  //testLog2();
}
