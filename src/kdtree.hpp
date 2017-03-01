#include <array>
#include <string>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <tuple>
#include <queue>

#include <iostream>

namespace kdtree {

using std::vector;
using std::array;
using std::tuple;
using std::get;
using std::priority_queue;
using std::to_string;

using Real = double;
using Size = array<Real, 1>::size_type;

template<typename T>
std::string to_string(vector<T> v) {
  std::string s{"["};
  for (auto e : v) {
    s += to_string(e) + " ";
  }
  return s + "]";
}

template<typename T, Size DIMS>
std::string to_string(array<T, DIMS> a) {
  std::string s{"["};
  for (auto e : a) {
    s += to_string(e) + " ";
  }
  return s + "]";
}

template<Size S>
std::string to_string(const char s[S]) {
  return s;
}

std::string to_string(const char c) {
  return {c, 0};
}

std::string to_string(const char* s) {
  return s;
}

std::string to_string(std::string s) {
  return s;
}

constexpr bool debug_output = false;

template<typename T>
void dbg(T v) {
  if (debug_output) {
    std::cout << to_string(v);
  }
}

template<typename T, typename... Ts>
void dbg(T v, Ts&&... vs) {
  if (debug_output) {
    std::cout << to_string(v);
    dbg(vs...);
  }
}

template<Size DIMS>
using Point = array<Real, DIMS>; // 160 byte

struct Division {
  int dim;
  Real p;
};

struct KdTree {
  KdTree(int size, int depth)
    : depth(depth),
      // divisions(2 * (1 << depth) - 1) { // {sum_{i=0}^{depth} i}
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

std::string to_string(ElemIter begin, Size size, ElemIter totalEnd) {
  std::string s{"["};
  for (auto i = begin; i < std::min(begin + size, totalEnd); ++i) {
    s += to_string(*i) + " ";
  }
  return s + "]";
}

inline Real square(Real v) { return v * v; }

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
    if (square(variance - currentVariance) < 0.01) {
      currentDim = d;
    }
  }
  dbg("split in dim ", currentDim, ": at ");
  auto mid = std::min(begin + size / 2, lastElem - 1);
  std::nth_element(begin, mid, end,
      [&points, currentDim](int a, int b){ return points[a][currentDim] < points[b][currentDim]; });
  dbg(points[*mid][currentDim], "=", points[*mid], "[", currentDim, "]\n");
  dbg("    left: ", to_string(begin, size / 2, lastElem), "\n");
  dbg("    right: ", to_string(mid, size / 2, lastElem), "\n");
  divs[mydiv] = Division{currentDim, points[*mid][currentDim]};
  buildImpl(begin, size / 2, lastElem, divs, 2 * mydiv + 1, points, depth + 1, maxDepth);
  buildImpl(mid, size / 2, lastElem, divs, 2 * mydiv + 2, points, depth + 1, maxDepth);
}

inline int log2ceil(int n) {
  n -= 1;
  int i = 0;
  for (; n != 0; i++) {
    n >>= 1;
  }
  return i;
}

inline int log2floor(int n) {
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

int countVisitedLeafes = 0;

template<Size DIMS, typename Queue>
void searchNNDown(Size divI, ElemIter begin, Size size,
    Size largestSizeToMoveUpTo,
    Real minDistInTree, array<Real, DIMS> &minDistInTreePerDim,
    Queue &nearest,
    const Point<DIMS> p, Size secoundLastLevel, ElemIter totalEnd, const KdTree &tree, const vector<Point<DIMS>> &points, int k) {

  ++countVisitedLeafes;

  while (divI < secoundLastLevel) {
    auto div = tree.divisions[divI];
    auto left = p[div.dim] < div.p;
    dbg(size, " ", left ? "left" : "right", to_string(begin, size, totalEnd), "\n");
    divI = 2 * divI + (left ? 1 : 2);
    size /= 2;
    begin = left ? begin : begin + size;
  }
  for (auto i = begin; i < std::min(begin + size, totalEnd); i++) {
    dbg("[", *i, "]");
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
    Real minDistInTree, array<Real, DIMS> &minDistInTreePerDim,
    Queue &nearest,
    const Point<DIMS> p, Size secoundLastLevel, ElemIter totalEnd, const KdTree &tree, const vector<Point<DIMS>> &points, int k) {

  while (size < largestSizeToMoveUpTo) {
    auto isRightChild = divI % 2 == 0;
    auto divUpI = (divI - 1) / 2;
    auto divUp = tree.divisions[divUpI];
    auto minDistInTreeOther = minDistInTree;
    auto minDistInTreePerDimOther = minDistInTreePerDim;
    auto distInTreeForDim = square(p[divUp.dim] - divUp.p);
    if (square(p[divUp.dim] - divUp.p) > 0.01) { // if this is the case we cannot assume to win anything
      minDistInTreeOther += distInTreeForDim - minDistInTreePerDimOther[divUp.dim];
      minDistInTreePerDimOther[divUp.dim] = distInTreeForDim;
    }
    dbg("", "eval other ", size, " ", minDistInTreeOther, "<"
      , (nearest.size() < k ? -1 : get<Real>(nearest.top())), " "
      , p[divUp.dim], "==", divUp.p, "="
      , "", (p[divUp.dim] == divUp.p ? "t" : "f"), " "
      , distInTreeForDim, "=", p, "[", divUp.dim, "]-", divUp.p
      , minDistInTreePerDimOther, "\n");
    if (nearest.size() < k
        || square(p[divUp.dim] - divUp.p) < 0.01
        || p[divUp.dim] == divUp.p // in case the decisions while going down were half wrong
        || minDistInTreeOther < get<Real>(nearest.top())) {
      auto beginOther = begin + (isRightChild ? -size : +size);
      auto sizeOther = size;
      auto divOther = divI + (isRightChild ? -1 : +1);
      dbg("", "", size, " ", (!isRightChild ? "left" : "right")
        , " other divI:", divI, " otherI:", divOther, " "
        , to_string(begin, size, totalEnd), "\n");
      searchNNDown(divOther, beginOther, sizeOther,
        size,
        minDistInTreeOther, minDistInTreePerDimOther,
        nearest,
        p, secoundLastLevel, totalEnd, tree, points, k);
    }
    auto beginUp = begin + (isRightChild ? -size : 0);
    auto sizeUp = size * 2;
    begin = beginUp;
    size = sizeUp;
    divI = divUpI;
    dbg("", "", size, " ", "up", " ", minDistInTree, " ", divI, " "
      , to_string(begin, size, totalEnd), "\n");
  }
}

template<Size DIMS>
void printTreeDivisions(const KdTree &tree) {
  std::cout << "tree (depth:" << tree.depth << ") build done: \n";
  for (int d = 0, s = 1; d < tree.depth; ++d, s *= 2) {
    std::vector<int> histogramm(DIMS, 0);
    for (int i = 0; i < s; ++i) {
      histogramm[tree.divisions[s + i].dim] += 1;
    }
    std::cout << "depth " << d << ":   ";
    for (int i = 0; i < DIMS; ++i) {
      if (histogramm[i] > 0) {
        std::cout << i << ":" << histogramm[i] << "  ";
      }
    }
    std::cout << "\n";
  }
}

template<Size DIMS>
vector<Size> knn(KdTree tree, const vector<Point<DIMS>> &points, int k, Point<DIMS> p) {
  dbg("", tree.elems, "\n\n");
  auto secoundLastLevel = tree.divisions.size() / 2; // always floor b/c size is odd
  auto initSize = 1 << log2ceil(tree.elems.size());
  vector<tuple<Real, Size>> queueContainer{};
  queueContainer.reserve(k);
  auto compare = [](const tuple<Real, Size> &e1, const tuple<Real, Size> &e2){ return get<Real>(e1) < get<Real>(e2); };
  priority_queue<tuple<Real, Size>, decltype(queueContainer), decltype(compare)> nearest{compare, queueContainer};
  Size divI = 0; // index of division
  array<Real, DIMS> minDistInTreePerDim{}; // {} to zero initialize
  Real minDistInTree = 0;
  // search Down
  countVisitedLeafes = 0;
  searchNNDown(divI, tree.elems.begin(), initSize,
    initSize,
    minDistInTree, minDistInTreePerDim,
    nearest,
    p, secoundLastLevel, tree.elems.end(), tree, points, k);
  std::cout << "Visited Leafes: " << countVisitedLeafes << "/" << tree.divisions.size() << "\n";
  vector<Size> result{};
  result.reserve(k);
  while (nearest.size() > 0) {
    result.push_back(get<Size>(nearest.top()));
    nearest.pop();
  }
  return result;
}

}
