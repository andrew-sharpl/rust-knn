# knn

A k-nearest neighbors classifier implemented in Rust with Python bindings.

This is a learning project for exploring Rust's ownership model, type system, and cross-language interoperability via PyO3. The algorithm is brute-force KNN — the challenge was learning Rust, not beating sklearn on Euclidean distance (they call into BLAS for that).

## Features

- **Three distance metrics:** Euclidean (L2), Manhattan (L1), and Cosine
- **Parallel prediction:** rayon multi-threading across query points
- **Cache-friendly layout:** training data stored as a flat `Vec<f64>` in row-major order
- **Partial sort:** uses `select_nth_unstable_by` for O(n) average-time neighbor selection
- **Python bindings:** NumPy arrays go in, predictions come out — no manual serialization

## Quick start

### Rust

```toml
[dependencies]
knn = { git = "https://github.com/yourusername/knn" }
```

```rust
use knn::{KnnClassifier, distance::Metric};

let mut model = KnnClassifier::new(3);
model.fit(vec![0.0, 0.0, 1.0, 0.0, 0.0, 10.0], 2, vec![0, 0, 1]);
let predictions = model.predict(&[0.1, 0.0], 1, 2);
assert_eq!(predictions, vec![0]);
```

To use a non-default metric:

```rust
let mut model = KnnClassifier::with_metric(3, Metric::Manhattan);
```

### Python

```bash
uv run maturin develop
```

```python
import numpy as np
import knn

model = knn.KnnClassifier(3)
model.fit(
    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 10.0]]),
    np.array([0, 0, 1]),
)
predictions = model.predict(np.array([[0.1, 0.0]]))
# predictions == [0]
```

With a different metric:

```python
model = knn.KnnClassifier(3, metric=knn.Metric.Manhattan)
```

## Distance metrics

| Metric | Formula | When to use |
|--------|---------|-------------|
| Euclidean | √Σ(xᵢ − yᵢ)² | General-purpose, continuous features |
| Manhattan | Σ\|xᵢ − yᵢ\| | Grid-like data, robust to outliers |
| Cosine | 1 − (A·B)/(‖A‖‖B‖) | Text/TF-IDF, direction matters more than magnitude |

Cosine distance panics on zero vectors (the denominator is zero, making it undefined).

## Benchmarks

Against sklearn's `algorithm='brute'` on a synthetic dataset (5000 train × 500 test, 10 features, 3 classes):

| Implementation | Time | Notes |
|---------------|------|-------|
| sklearn (brute) | 1.0× baseline | Calls into Cython/Fortran/BLAS with SIMD |
| Rust (this crate) | ~3–4× slower | Pure Rust, no SIMD or BLAS |

sklearn's brute-force is heavily optimized with hand-tuned SIMD. Matching or beating it on raw Euclidean distance would require algorithmic improvements (KD-tree, LSH) or BLAS bindings. Where this crate wins: ~10× less memory, custom metrics, and a path to algorithmic improvements that sklearn doesn't optimize for.

## Running tests

```bash
# Rust unit + integration tests
cargo test

# Python smoke tests
uv run maturin develop
uv run python scripts/test_python.py

# Benchmarks (requires sklearn)
uv run maturin develop
uv run python scripts/benchmark.py
```

Or use the justfile commands: `just test`, `just bench`, `just dev`.

## Project structure

```
src/
├── lib.rs        # KnnClassifier core (flat buffer, rayon, majority vote)
├── distance.rs    # Metric enum + Euclidean/Manhattan/Cosine functions
└── python.rs     # PyO3 wrapper (NumPy → flat buffer → Rust)
tests/
└── test_knn.rs   # Integration tests (public API only)
scripts/
├── benchmark.py   # sklearn vs Rust timing comparison
└── test_python.py # Python smoke tests
knn.pyi           # Type stubs for Pyright/IDE autocomplete
```

## License

MIT