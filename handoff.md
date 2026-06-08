# Project Handoff: knn

## Purpose

This is a learning project for a CS graduate (University of Toronto, 2024) building a Rust-backed k-nearest neighbors library with Python bindings via PyO3 + maturin. The goal is to learn the Rust/Python FFI stack and demonstrate the ability to ship a published crate + pip-installable package.

The algorithm itself (brute-force KNN) is not novel — the challenge is learning Rust's ownership, borrowing, trait system, and cross-language interoperability.

## Current State

A working end-to-end pipeline exists:

- **Pure Rust core:** `KnnClassifier` with `new`, `fit`, `predict`, `fit_flat`, `predict_flat`
- **Memory layout:** Training data stored as a single flat `Vec<f64>` (row-major) for cache locality
- **Parallelism:** `predict_flat` uses `rayon` to parallelize across queries
- **Python bindings:** PyO3 wrapper accepts NumPy arrays and flattens them into Rust buffers
- **Tests:** 16 Rust unit tests + 3 integration tests + Python smoke tests, all passing
- **Benchmarks:** Benchmark harness exists but shows Rust is still slower than sklearn's brute-force (see Benchmark History)

## Repository Structure

```
knn/
├── Cargo.toml              # Rust deps: pyo3, numpy, ndarray, rayon
├── src/
│   ├── lib.rs              # Core KNN logic (flat buffer, rayon)
│   ├── python.rs           # PyO3 wrapper (KnnClassifierPy)
│   └── distance.rs         # Placeholder for distance metrics (currently empty)
├── tests/
│   └── test_knn.rs         # Integration tests (public API only)
├── scripts/
│   ├── benchmark.py        # Original benchmark (sklearn vs rust)
│   └── test_python.py      # Python smoke tests
├── docs/
│   └── week1.md            # Session-by-session roadmap for Week 1
└── knn.pyi                 # Type stubs for Pyright/IDE support
```

## Key Design Decisions

1. **Owned data, no lifetimes:** `KnnClassifier` owns its training data (`Vec<f64>`). It does not borrow with lifetimes (`KnnClassifier<'a>`) to avoid lifetime complexity in a first Rust project. This can be refactored later as an exercise (TRPL Ch. 10).

2. **Flat buffer layout:** Data is stored as `[p0f0, p0f1, p1f0, p1f1, ...]` rather than `Vec<Vec<f64>>`. This improves cache locality and prepares the code for NumPy interop. Points are accessed via slicing: `&self.data[i * dim .. (i+1) * dim]`.

3. **Dual API:** `fit` accepts `Vec<Vec<f64>>` (for Rust tests and backwards compatibility). `fit_flat` accepts an already-flat `Vec<f64>` (used by the Python wrapper to avoid double-copying).

4. **Partial sort:** `select_nth_unstable_by` finds the k smallest distances in O(n) average time instead of sorting the entire distance array.

5. **Thread safety:** `predict_flat` takes `&self` and uses `rayon::par_chunks`. Because the struct is only read, multiple threads can safely share the training data.

## Benchmark History (Git Tags)

| Tag | Description | Relative Speed vs sklearn |
|-----|-------------|---------------------------|
| `benchmark-baseline` | Vec<Vec<f64>>, full sort, single-threaded | ~0.04× |
| `benchmark-flat-buffer` | Flat Vec<f64>, select_nth_unstable_by | ~0.04× (small improvement) |
| `benchmark-rayon` | Flat buffer + rayon parallelism | ~0.25–0.29× |

**Finding:** Sklearn's `algorithm='brute'` calls into Cython/Fortran/BLAS with hand-optimized SIMD. Beating it on Euclidean distance requires either SIMD, BLAS bindings, or an algorithmic improvement (KD-tree, LSH). The Rust code is ~3–4× slower on brute-force Euclidean but uses ~10× less memory and enables custom metrics/algorithms that sklearn does not optimize.

## Python Usage

```python
import numpy as np
import knn

model = knn.KnnClassifier(3)
model.fit(
    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 10.0]]),
    np.array([0, 0, 1])
)
predictions = model.predict(np.array([[0.1, 0.0]]))
# predictions == [0]
```

Build and install:
```bash
uv run maturin develop
uv run python scripts/test_python.py
```

## Next Steps (Roadmap)

### Immediate (Week 3)
1. **Distance metrics:** Add Manhattan (L1) and Cosine to `src/distance.rs`. Expose a `metric` parameter to Python (likely an enum: `Metric::Euclidean`, `Metric::Manhattan`, `Metric::Cosine`).
2. **README:** Installation instructions, usage examples, benchmark table, and a note about when to use this library vs sklearn.
3. **knn.pyi:** Already exists; update it when adding the `metric` parameter.

### Medium-term (Week 4+)
4. **Algorithmic improvements:** KD-tree for exact search, then LSH for approximate search. This is where the library can legitimately outperform sklearn brute-force on large datasets.
5. **Weighted voting:** Currently uses uniform majority vote. Add distance-weighted voting.
6. **SIMD / BLAS:** For the truly performance-obsessed. Not necessary for the learning goal.

### Publishing
7. **Cargo.toml metadata:** Add author, description, license, repository URL before publishing to crates.io.
8. **PyPI:** `maturin publish` after verifying `maturin build` produces a clean wheel.

## Gotchas & Context

- **PyO3 version:** 0.23. Newer versions exist (0.28+) but upgrading may require API changes. The `Bound<'_, T>` pattern is the modern PyO3 style.
- **NumPy interop:** The `numpy` crate version (0.23) is tied to the PyO3 version. Do not upgrade one without checking compatibility.
- **Type stubs:** `knn.pyi` is picked up automatically by `maturin` during build. Keep it in the project root.
- **The `Vec<usize>` thing:** If you see `Vec<<usize>` anywhere, it's a typo from an earlier assistant. It should always be `Vec<usize>`.
- **Rust edition:** 2024. This is cutting-edge (released ~late 2024). If you downgrade edition for compatibility, check that syntax still compiles.

## Tests

Run Rust tests:
```bash
cargo test
```

Run Python tests:
```bash
uv run maturin develop
uv run python scripts/test_python.py
```

## Contact / Notes

This project is intentionally a learning exercise. Show the code that needs to be written but do not write it for the user. Treat it as a guided tutorial with code and in depth explanations, highlighting key concepts and design decisions. The code prioritizes clarity and correctness over micro-optimization. When in doubt, choose the simpler, more readable implementation and benchmark before optimizing.
