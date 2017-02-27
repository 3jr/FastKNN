#include <array>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <tuple>
#include <queue>

namespace kdtree {

using std::vector;
using std::array;
using std::tuple;
using std::get;
using std::priority_queue;

using Real = double;
using Size = array<Real, 1>::size_type;

template<Size DIMS>
using Point = array<Real, DIMS>; // 160 byte

struct Division {
  int dim;
  Real p;
};

struct KdTree {
  KdTree(int size, int depth)
    : depth(depth),
      //divisions(2 * (1 << depth) - 1) { // {sum_{i=0}^{depth} i}
      divisions((1 << depth) - 1) {
    elems.reserve(size);
    for (auto i = 0; i < size; ++i) {
      elems.push_back(i);
    }
  }
  int depth;
  vector<Division> divisions;
  vector<int> elems;
};

using ElemIter = vector<int>::iterator;

Real square(Real v) { return v * v; }

template<Size DIMS>
void buildImpl(ElemIter begin, Size size, ElemIter lastElem, vector<Division> &divs, int mydiv,
    const vector<Point<DIMS>> &points, int depth, int maxDepth) {
  auto end = std::min(begin + size, lastElem);
  if (maxDepth <= depth) {
    return;
  }
  int currentDim = 0;
  Real currentVariance = 0;
  for (int d = 0; d < DIMS; d++) {
    Real average = std::accumulate(begin, end, 0, [&points, d](Real sum, int i){ return sum + points[i][d]; }) / size;
    Real variance = std::accumulate(begin, end, 0, [&points, d, average](Real sum, int i){ return sum + square(points[i][d] - average); }) / size;
    if (variance > currentVariance) {
      currentDim = d;
      currentVariance = variance;
    }
  }
  auto mid = std::min(begin + size / 2, lastElem - 1);
  std::nth_element(begin, mid, end,
      [&points, currentDim](int a, int b){ return points[a][currentDim] < points[b][currentDim]; });
  divs[mydiv] = Division{currentDim, points[*mid][currentDim]};
  buildImpl(begin, size / 2, lastElem, divs, 2 * mydiv + 1, points, depth + 1, maxDepth);
  buildImpl(mid, size / 2, lastElem, divs, 2 * mydiv + 2, points, depth + 1, maxDepth);
}

// round up
int log2ceil(int n) {
  n -= 1;
  int i = 0;
  for (; n != 0; i++) {
    n >>= 1;
  }
  return i;
}

// round up
int log2floor(int n) {
  int i = 0;
  for (; n != 0; i++) {
    n >>= 1;
  }
  return i - 1;
}

template<Size DIMS>
KdTree buildKdTree(const vector<Point<DIMS>> &points) {
  auto depth = static_cast<int>(log2ceil(points.size()));
  KdTree tree{static_cast<int>(points.size()), depth};
  buildImpl(tree.elems.begin(), 1 << depth, tree.elems.end(), tree.divisions, 0, points, 0, tree.depth);
  return tree;
}

template<Size DIMS>
Real distSquared(Point<DIMS> p1, Point<DIMS> p2) {
  Real d = 0;
  for (auto i = 0; i < DIMS; ++i) {
    d += square(p1[i] - p2[i]);
  }
  return d;
}

template<Size DIMS, typename Queue>
void searchNNDown(Size divI, ElemIter begin, Size size,
    Size largestSizeToMoveUpTo,
    Real minDistInTree, array<Real, DIMS> &minDistInTreePerDim,
    Queue &nearest,
    const Point<DIMS> p, Size secoundLastLevel, ElemIter totalEnd, const KdTree &tree, const vector<Point<DIMS>> &points, int k) {


  while (divI < secoundLastLevel) {
    auto div = tree.divisions[divI];
    auto left = p[div.dim] < div.p;
    //std::cout << "" << size << (left ? "left" : "right") << '\n';
    divI = 2 * divI + (left ? 1 : 2);
    size /= 2;
    begin = left ? begin : begin + size;
  }
  for (auto i = begin; i < std::min(begin + size, totalEnd); i++) {
    auto dist = distSquared(points[*i], p);
    if (nearest.size() < k) {
      nearest.push({dist, *i});
    } else if (dist < get<Real>(nearest.top())) {
      nearest.pop();
      nearest.push({dist, *i});
    }
  }
  searchNNUp(divI, begin, size,
    largestSizeToMoveUpTo,
    minDistInTree, minDistInTreePerDim,
    nearest,
    p, secoundLastLevel, totalEnd, tree, points, k);
}

template<Size DIMS, typename Queue>
void searchNNUp(Size divI, ElemIter begin, Size size,
    Size largestSizeToMoveUpTo,
    Real minDistInTree, array<Real, DIMS> minDistInTreePerDim /* optimize copy? */,
    Queue &nearest,
    const Point<DIMS> p, Size secoundLastLevel, ElemIter totalEnd, const KdTree &tree, const vector<Point<DIMS>> &points, int k) {

  while (size < largestSizeToMoveUpTo) {
    auto div = tree.divisions[divI];
    auto isRightChild = divI % 2 == 0;
    if (nearest.size() < k || minDistInTree < get<Real>(nearest.top())) {
      auto distInTreeForDim = square(p[div.dim] - div.p);
      minDistInTree += distInTreeForDim - minDistInTreePerDim[div.dim];
      minDistInTreePerDim[div.dim] = distInTreeForDim;
      auto beginOther = begin + (isRightChild ? -size : +size);
      auto sizeOther = size;
      auto divOther = divI + (isRightChild ? -1 : +1);
      //std::cout << "" << size << " " << (!isRightChild ? "left" : "right") << " other " << divI << " " << divOther << '\n';
      searchNNDown(divOther, beginOther, sizeOther,
        size,
        minDistInTree, minDistInTreePerDim,
        nearest,
        p, secoundLastLevel, totalEnd, tree, points, k);
    }
    auto beginUp = begin + (isRightChild ? -size : 0);
    auto sizeUp = size * 2;
    auto divUp = (divI - 1) / 2;
    begin = beginUp;
    size = sizeUp;
    divI = divUp;
    //std::cout << "" << size << " " << "up" << " " << minDistInTree << " " << divI << '\n';
  }
}

template<Size DIMS>
void knn(const vector<Point<DIMS>> &points, int k) {
  auto tree = buildKdTree(points);
  std::cout << "tree build done"  << '\n';
  auto secoundLastLevel = tree.divisions.size() / 2; // always floor b/c size is odd
  auto initSize = 1 << log2ceil(tree.elems.size());
  for (const auto p : points) {
    //if (p != Point<DIMS>{1,2,3,2,2}) { continue; }
    vector<tuple<Real, Size>> queueContainer{};
    queueContainer.reserve(k);
    auto compare = [](const tuple<Real, Size> &e1, const tuple<Real, Size> &e2){ return get<Real>(e1) > get<Real>(e2); };
    priority_queue<tuple<Real, Size>, decltype(queueContainer), decltype(compare)> nearest{compare, queueContainer};
    Size divI = 0; // index of division
    array<Real, DIMS> minDistInTreePerDim{}; // {} to zero initialize
    Real minDistInTree = 0;
    // search Down
    searchNNDown(divI, tree.elems.begin(), initSize,
      initSize,
      minDistInTree, minDistInTreePerDim,
      nearest,
      p, secoundLastLevel, tree.elems.end(), tree, points, k);
    //std::cout << "elements for this point: "  << nearest.size() << '\n';
    while (nearest.size()) {
      //std::cout << ", "  << get<Size>(nearest.top());
      nearest.pop();
    }
    //std::cout << '\n';
  }
}

}
