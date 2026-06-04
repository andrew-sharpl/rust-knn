"""
End-to-end smoke tests for the Rust KNN Python bindings.
"""

import numpy as np
import knn

# Test 1: Basic 2D dataset
model = knn.KnnClassifier(1)
model.fit(
    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 10.0]]),
    np.array([0, 0, 1]),
)
predictions = model.predict(np.array([[0.1, 0.0], [0.0, 9.0]]))
assert predictions == [0, 1], f"Expected [0, 1], got {predictions}"
print("Test 1 passed: basic 2D prediction")

# Test 2: Multiple queries with k=2
model2 = knn.KnnClassifier(2)
model2.fit(
    np.array([[0.0], [1.0], [10.0], [11.0]]),
    np.array([0, 0, 1, 1]),
)
predictions2 = model2.predict(np.array([[0.4], [10.4]]))
assert predictions2 == [0, 1], f"Expected [0, 1], got {predictions2}"
print("Test 2 passed: multiple queries with k=2")

print("All Python tests passed!")
