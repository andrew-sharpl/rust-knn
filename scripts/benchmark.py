"""
Benchmark: sklearn (brute-force + KD-tree) vs. Rust KNN (brute-force + KD-tree).

Generates a synthetic classification dataset and times prediction for all four
combinations, asserting that results agree.

KD-tree should win at larger training sets with low-to-moderate dimensionality.
At high dimensions, the curse of dimensionality erodes the pruning advantage.
"""

import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

import knn

np.random.seed(42)
n_train = 50_000
n_test = 500
n_features = 5

X_train = np.random.rand(n_train, n_features)
y_train = np.random.randint(0, 3, size=n_train)
X_test = np.random.rand(n_test, n_features)


def time_predict(model, X_test):
    t0 = time.perf_counter()
    preds = model.predict(X_test)
    return time.perf_counter() - t0, preds


# --- sklearn brute-force ---
sk_brute = KNeighborsClassifier(n_neighbors=3, algorithm="brute")
sk_brute.fit(X_train, y_train)
sk_brute_time, sk_brute_preds = time_predict(sk_brute, X_test)

# --- sklearn KD-tree ---
sk_kdtree = KNeighborsClassifier(n_neighbors=3, algorithm="kd_tree")
sk_kdtree.fit(X_train, y_train)
sk_kdtree_time, sk_kdtree_preds = time_predict(sk_kdtree, X_test)

# --- Rust brute-force ---
rust_brute = knn.KnnClassifier(3)
rust_brute.fit(X_train, y_train)
rust_brute_time, rust_brute_preds = time_predict(rust_brute, X_test)

# --- Rust KD-tree ---
rust_kdtree = knn.KnnClassifier(3, algorithm=knn.Algorithm.KdTree)
rust_kdtree.fit(X_train, y_train)
rust_kdtree_time, rust_kdtree_preds = time_predict(rust_kdtree, X_test)

# --- verify results agree ---
assert list(sk_brute_preds) == rust_brute_preds, "sklearn brute != rust brute"
assert list(sk_brute_preds) == rust_kdtree_preds, "sklearn brute != rust kdtree"
assert list(sk_kdtree_preds) == rust_kdtree_preds, "sklearn kdtree != rust kdtree"

# --- print results ---
print(f"dataset: {n_train} train, {n_test} test, {n_features} features")
print()
print(f"sklearn brute:  {sk_brute_time:.4f}s")
print(f"sklearn kdtree: {sk_kdtree_time:.4f}s")
print(f"rust brute:     {rust_brute_time:.4f}s")
print(f"rust kdtree:    {rust_kdtree_time:.4f}s")
print()
print(f"rust kdtree vs sklearn brute:  {sk_brute_time / rust_kdtree_time:.2f}x")
print(f"rust kdtree vs sklearn kdtree: {sk_kdtree_time / rust_kdtree_time:.2f}x")
print(f"rust kdtree vs rust brute:     {rust_brute_time / rust_kdtree_time:.2f}x")
