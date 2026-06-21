# knn

A k-nearest neighbors classifier implemented in Rust with optional Python bindings.

This is a learning project for exploring Rust's ownership model, type system, and cross-language interoperability via PyO3. Two search algorithms are supported: brute-force (linear scan) and KD-tree (spatial partitioning).

## Features

- **Two search algorithms:** brute-force (O(n) per query) and KD-tree (O(log n) average per query), selectable via the `algorithm` parameter
- **Three distance metrics:** Euclidean (L2), Manhattan (L1), and Cosine
- **Four weighting schemes:** Uniform, InverseDistance, SmoothedInverse, Gaussian
- **Parallel prediction:** rayon multi-threading across query points (both algorithms)
- **Cache-friendly layout:** training data stored as a flat `Vec<f64>` in row-major order
- **Partial sort:** brute-force uses `select_nth_unstable_by` for O(n) average-time neighbor selection
- **Optional Python bindings:** enabled via the `python` Cargo feature; NumPy arrays go in, predictions come out

## Quick start

### Rust

The Cargo package is `rust-knn`; this dependency alias keeps the Rust import as `knn`.

```bash
cargo add rust-knn
```

```rust
use knn::{KnnClassifier, distance::Metric, algorithm::Algorithm};

// Brute-force (default)
let mut model = KnnClassifier::new(3);
model.fit(vec![0.0, 0.0, 1.0, 0.0, 0.0, 10.0], 2, vec![0, 0, 1]);
let predictions = model.predict(&[0.1, 0.0], 1, 2);
assert_eq!(predictions, vec![0]);

// KD-tree
let mut model = KnnClassifier::new(3);
model.with_algorithm(Algorithm::KdTree);
model.fit(vec![0.0, 0.0, 1.0, 0.0, 0.0, 10.0], 2, vec![0, 0, 1]);

// With a non-default metric
let mut model = KnnClassifier::new(3);
model.with_metric(Metric::Manhattan).with_algorithm(Algorithm::KdTree);
```

### Python

Python bindings are gated behind the `python` Cargo feature. Maturin enables this automatically via `pyproject.toml`, so the standard build commands just work.

**Prerequisites:** Rust toolchain, Python 3.9+, and `uv` for dependency management.

```bash
uv sync --group dev
uv tool run maturin develop
```

```python
import numpy as np
import knn

# Default: Euclidean metric, brute-force algorithm
model = knn.KnnClassifier(3)
model.fit(
    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 10.0]]),
    np.array([0, 0, 1]),
)
predictions = model.predict(np.array([[0.1, 0.0]]))
# predictions == [0]

# KD-tree (faster on moderate-size datasets)
model = knn.KnnClassifier(3, algorithm=knn.Algorithm.KdTree)

# With a different metric or weighting
model = knn.KnnClassifier(3, metric=knn.Metric.Manhattan, algorithm=knn.Algorithm.KdTree)
```

Invalid combinations and misuse raise `ValueError` (catchable in Python):
- `k == 0`
- `k` larger than the training set
- label count mismatch
- `predict()` before `fit()`
- query dimension mismatch
- cosine distance on a zero vector
- `Metric.Cosine` with `Algorithm.KdTree` (pruning is invalid for cosine)

## Algorithms

| Algorithm | Complexity | When to use |
|-----------|------------|-------------|
| BruteForce | O(n) per query | Small datasets, or when dimensionality is very high |
| KdTree | O(log n) average per query | Moderate-size datasets (up to ~100k points), low-to-moderate dimensionality |

The KD-tree supports Euclidean and Manhattan metrics. Cosine distance does not satisfy the triangle inequality in axis-aligned space, so KD-tree pruning is incorrect for cosine — using `Metric.Cosine` with `Algorithm.KdTree` raises a `ValueError` (Python) or panics (Rust) at `fit()` time.

## Distance metrics

| Metric | Formula | When to use |
|--------|---------|-------------|
| Euclidean | √Σ(xᵢ − yᵢ)² | General-purpose, continuous features |
| Manhattan | Σ\|xᵢ − yᵢ\| | Grid-like data, robust to outliers |
| Cosine | 1 − (A·B)/(‖A‖‖B‖) | Text/TF-IDF, direction matters more than magnitude |

Cosine distance is undefined for zero vectors (the denominator is zero). In Rust this panics; in Python it raises `ValueError`.

## Benchmarks

### Rust: brute-force vs KD-tree (500 queries, 3 neighbors)

| Dataset | Brute-force | KD-tree | Speedup |
|---------|-------------|---------|---------|
| 5k train, dim=10 | 2.98 ms | 128 µs | 23× |
| 50k train, dim=3 | 18.0 ms | 78 µs | 231× |
| 50k train, dim=10 | 29.2 ms | 131 µs | 223× |
| 50k train, dim=50 | 144 ms | 149 µs | 966× |

### Python: vs sklearn (500 queries, dim=10)

| Dataset | sklearn brute | sklearn kdtree | rust brute | rust kdtree |
|---------|---------------|----------------|------------|-------------|
| 50k train | 0.137s | 0.064s | 0.050s | **0.032s** |
| 500k train | **0.802s** | 1.213s | 5.190s | 1.085s |

At 50k training points, the Rust KD-tree is **4.3× faster than sklearn's brute-force** and **2× faster than sklearn's own KD-tree**. At 500k, sklearn's BLAS-backed brute-force wins because its SIMD constant-factor advantage overcomes the O(n) vs O(log n) gap. Closing that gap would require SIMD/BLAS support in the Rust distance functions.

## Running tests

```bash
# Rust unit + integration tests (no Python feature)
cargo test

# Rust tests with Python bindings enabled
cargo test --features python

# Python tests (requires building the extension first)
uv tool run maturin develop
uv run pytest tests/ -v

# Rust benchmarks
cargo bench

# Python benchmark against sklearn
uv tool run maturin develop --release
uv run python scripts/benchmark.py
```

Or use the justfile commands: `just test`, `just pytest`, `just bench`, `just bench-py`, `just build`, and `just dev`.

## Project structure

```
src/
├── lib.rs        # KnnClassifier (dispatch, label mapping, weighted_vote)
├── algorithm.rs  # Algorithm enum (BruteForce, KdTree)
├── brute.rs      # brute_force_search() — linear scan
├── kdtree.rs     # KdTree, build_kdtree(), k_nearest() — spatial partitioning
├── distance.rs   # Metric enum + Euclidean/Manhattan/Cosine functions
├── weights.rs    # Weighting enum + distance-to-weight conversion
└── python.rs     # PyO3 wrapper (NumPy → flat buffer → Rust) — requires `python` feature
benches/
└── bench_knn.rs   # Criterion benchmarks (brute vs KD-tree)
```

## License

MIT. See `LICENSE`.
