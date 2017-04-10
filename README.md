# FastKNN

Fast approximate `k`-nearest neighbor (KNN) implementation

This Repository explores 2 ideas to find the K nearest neighbors of points in d-dimensional space.
The first one used `k`-`d` trees and the second one uses locality-sensitive hashing (lsh).

KNN Parameter:
* `DIMS` and `d`: Number of dimensions of the euclidean space in which the points reside.
* `k`: Number of approximate neighbors that should be determined.
* `points`: Input points.
* `u`: Test point.


## LSH (Locality-sensitive hashing)

LSH is an approximate KNN algorithm.
It hashes every point and considers points that hash in the same bucket candidates for nearest neighbors.

### Parameter

LSH Parameter:
* `K`: Number of hashes that are combined to improve hash accuracy.
* `r`: Radius of hash domain.
* `L`: Number of hash maps used.

### Usage

```
  auto hashes = lsh::generate_hashes<dims, K>(points, r, L);
  auto approximate_knn = lsh::knn<dims, K>(hashes, points, u, r, k);
```


### Algorithm

_Singular hash funktion_ for point `v \in R^d`: `h(v,a,b) = floor((dot_product(a, v) + b) / r)`
* With `a \in R^d` with every element randomly chosen from a normal distribution.
* With `b \in R` randomly chosen from a uniform distribution.

_Combined hash function_ for point `v \in R^d`:\r
`g(v,a1...K,b1...K) = (h(v,a1,b1), h(v,a2,b2), ..., h(v,aK,bK))`

Now we insert every input point into `L` different hash maps.
Every hash map uses a combined hash function `g` with different random `a`s and `b`s.

To find the `k`-nearest neighbors of a point `u` it is hashed for every hash function
of the `L` different hash maps.
Every point that hashed to the same hash in at least one of the `L` hash Maps
is a candidate for one of the `k`-nearest neighbors.

## `k`-`d` trees

This algorithm builds a `k`-`d` tree from the input points and searches within this tree to find neighbors.

### Usage

```
  auto tree = kdtree::buildKdTree(points);
  auto knn = kdtree::knn(tree, points, k, u);
```

### Algorithm

First we construct the `k`-`d` tree.
The `k`-`d` tree only saves a permutation of the input points and the divisions of the space.
The permutation is in such a way that every node has a continuous range of elements.

* the root node has the elements in range `[0, totalElements)` and the division at index `0`
* the child nodes of a node with elements `[a, b)` and the division at index `t`.
  * has a left node with elements `[a, (b-a)/2)` and the division at index `2*t+1`
  * has a right node with elements `[(b-a)/2, b)` and the division at index `2*t+2`

If `totalElements` is not a power of 2 we pretend that it is the next power of two everywhere with two exceptions:
1. We never try to access the point beyond the real max element index.
2. We never try to descend into a subtree that has no elements that really exist.

With the constructed `k`-`d` tree the nearest neighbors of a point `u` are found recursively.
One recursive invocation has two phases.
One descend phase and one ascent phase.

On the descend phase the leaf closed to `u` is found.
This leaf is considered a candidate for one of the `k` nearest neighbors.
Now we ascend back to the rood node and do two actions per visited node:
1. We consider every node a candidate for the `k` nearest neighbors.
2. The branch not taken while descending could have possible candidates.
   Based on the currently collected candidates we calculate if the other subtree could have same necessary candidates.
   If so we recursively call the algorithm on the subtree again.

## Build

Prerequisites
* cmake
* boost test
* gcc, g++ (at least with C++14 support)
* make

```
$ git clone https://github.com/DavidPfander-UniStuttgart/FastKNN
$ cd FastKNN
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=release ../
$ make -j4
$ cd ..
```

## Run Some Tests

Some test take a long time to complete,
so take a look at them before running them and don't just run all tests.
There are two boost test suites:
one is for LSH (named `lsh_tests`) and one for k-d trees (named `kdtree_tests`).

### LSH

```
$ ./build/boost_tests -t lsh_tests/demo
Running 1 test case...
generate_hashes_start [Points: 25] ... gen_done
[Not all adjacents found: 6 (tweak params for better results)]
*** No errors detected
```

### `k`-`d` tree

```
$ ./build/boost_tests -t kdtree_tests/demo
tree (depth:6) build done:
0 Visited Leafes: 4/63
1 Visited Leafes: 6/63
2 Visited Leafes: 7/63
3 Visited Leafes: 5/63
4 Visited Leafes: 6/63
5 Visited Leafes: 9/63
6 Visited Leafes: 10/63
7 Visited Leafes: 8/63
8 Visited Leafes: 8/63
9 Visited Leafes: 12/63
10 Visited Leafes: 14/63
*** No errors detected
```

