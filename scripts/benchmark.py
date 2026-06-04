"""
Benchmark: sklearn brute-force KNN vs. Rust KNN (via PyO3).

Generates a synthetic classification dataset and times prediction for both
implementations, asserting that results agree.
"""

import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import knn

np.random.seed(42)
n_train = 5000
n_test = 500
n_features = 10

X_train = np.random.rand(n_train, n_features)
y_train = np.random.randint(0, 3, size=n_train)
X_test = np.random.rand(n_test, n_features)

# sklearn brute-force
sk_model = KNeighborsClassifier(n_neighbors=3, algorithm='brute')
sk_model.fit(X_train, y_train)
t0 = time.perf_counter()
sk_preds = sk_model.predict(X_test)
sk_time = time.perf_counter() - t0

# rust knn
rust_model = knn.KnnClassifier(3)
rust_model.fit(X_train, y_train)
t0 = time.perf_counter()
rust_preds = rust_model.predict(X_test)
rust_time = time.perf_counter() - t0

# verify and print
assert list(sk_preds) == rust_preds
print(f"sklearn: {sk_time:.4f}s")
print(f"rust:    {rust_time:.4f}s")
print(f"speedup: {sk_time/rust_time:.2f}x")
