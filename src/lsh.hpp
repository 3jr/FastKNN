#pragma once

#include <random>
#include <cmath>
#include <tuple>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>

namespace lsh {

using std::get;
using std::size_t;
using std::vector;
using std::array;
using std::priority_queue;
using std::tuple;
using std::make_tuple;
template<typename K, typename V>
using multimap = std::unordered_multimap<K, V>;

using Real = double;

template<size_t DIMS>
using Vec = array<Real, DIMS>; // 160 byte

using Maps = vector<multimap<size_t, size_t>>;

inline Real square(Real v) { return v * v; }

template<size_t DIMS>
Real distSquared(Vec<DIMS> p1, Vec<DIMS> p2) {
  Real d = 0;
  for (auto i = 0; i < DIMS; ++i) {
    d += square(p1[i] - p2[i]);
  }
  return d;
}

// represents a singular hash function
template<size_t DIMS>
struct h_t {
  Vec<DIMS> a;
  Real b;
};

// represents the combined hash function
template<size_t DIMS, size_t K>
using g_t = std::array<h_t<DIMS>, K>;

template<size_t DIMS, size_t K>
std::size_t eval_g(const g_t<DIMS, K> &g, const Vec<DIMS> &v, Real r) {
  std::hash<std::size_t> hasher;
  std::size_t hash = 0x4FEE0B91;
  for (int i = 0; i < K; ++i) {
    h_t<DIMS> h = g[i];
    Real dot_product = 0;
    for (int j = 0; j < DIMS; ++j) {
      dot_product += h.a[j] * v[j];
    }
    auto one_hash = static_cast<std::size_t>(std::floor((dot_product + h.b) / r));
    hash ^= hasher(one_hash) + 0x9e3779b9 + (hash<<6) + (hash>>2);
  }
  return hash;
}

template<size_t DIMS, size_t K>
vector<g_t<DIMS, K>> generate_hash_functions(Real r, size_t L) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> a_dist(0, 1);
  std::uniform_real_distribution<> b_dist(0, r);

  vector<g_t<DIMS, K>> res{};
  for (int i = 0; i < L; ++i) {
    g_t<DIMS, K> g;
    for (int j = 0; j < K; ++j) {
      Vec<DIMS> a{};
      for (int k = 0; k < DIMS; ++k) {
        a[k] = a_dist(gen);
      }
      g[j] = {a, b_dist(gen)};
    }
    res.push_back(g);
  }
  return res;
}

// to be used as the `HASH` argument for a std::unordered_map or std::unordered_multimap
template<typename T, std::size_t SIZE>
struct hash_array {
  inline std::size_t operator()(std::array<T, SIZE> a) {
    std::hash<T> hasher;
    auto hash = hasher(a[0]);
    for (int i = 1; i < SIZE; ++i) {
      hash ^= hasher(a[i]) + 0x9e3779b9 + (hash<<6) + (hash>>2);
    }
    return hash;
  }
};

template<size_t DIMS, size_t K>
auto generate_hashes(vector<Vec<DIMS>> points, Real r, size_t L) {
  auto gs = generate_hash_functions<DIMS, K>(r, L);
  Maps maps(L);
  for (int i = 0; i < L; ++i) {
    const auto &g = gs[i];
    auto &m = maps[i];
    for (int j = 0; j < points.size(); ++j) {
      const auto &p = points[j];
      m.insert(std::make_pair(eval_g(g, p, r), static_cast<size_t>(j)));
    }
  }
  return make_tuple(maps, gs);
}

template<size_t DIMS, size_t K>
vector<size_t> knn(tuple<Maps, vector<g_t<DIMS, K>>> maps_and_gs,
    vector<Vec<DIMS>> points, Vec<DIMS> p, Real r, size_t k) {
  auto maps = get<0>(maps_and_gs);
  auto gs = get<1>(maps_and_gs);
  using E = tuple<Real, size_t>;
  vector<E> queueContainer{};
  queueContainer.reserve(k);
  auto compare = [](const E &e1, const E &e2)
    { return get<Real>(e1) < get<Real>(e2); };
  priority_queue<tuple<Real, size_t>, decltype(queueContainer),
    decltype(compare)> nearest{compare, queueContainer};
  std::unordered_set<size_t> tested_points;
  for (int i = 0; i < maps.size(); ++i) {
    const auto &g = gs[i];
    const auto &m = maps[i];
    const auto range = m.equal_range(eval_g(g, p, r));
    for (auto e = range.first; e != range.second; ++e) {
      if (tested_points.end() != tested_points.find(e->second)) { continue; }
      tested_points.insert(e->second);
      auto d = distSquared(points[e->second], p);
      if (nearest.size() < k) {
        nearest.push({d, e->second});
      } else if (d < get<Real>(nearest.top())) {
        nearest.pop();
        nearest.push({d, e->second});
      }
    }
  }
  vector<size_t> result{};
  result.reserve(k);
  while (nearest.size() > 0) {
    result.push_back(get<size_t>(nearest.top()));
    nearest.pop();
  }
  return result;
}

}
