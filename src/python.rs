//! Python bindings for `knn` via PyO3.
//!
//! This module exposes `KnnClassifier` to Python as a `#[pyclass]`. NumPy arrays
//! are accepted as input and flattened into Rust-owned buffers before being
//! passed to the core classifier.

use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

/// Python-visible module entry point.
#[pymodule]
fn knn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KnnClassifierPy>()?;
    m.add_class::<MetricPy>()?;
    m.add_class::<WeightingPy>()?;
    m.add_class::<AlgorithmPy>()?;
    Ok(())
}

/// Python-visible distance metric enum.
#[pyclass(name = "Metric", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
enum MetricPy {
    Euclidean,
    Manhattan,
    Cosine,
}

/// Python-visible search algorithm enum.
#[pyclass(name = "Algorithm", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
enum AlgorithmPy {
    BruteForce,
    KdTree,
}

/// Python wrapper around Rust weighting functions for predict
#[pyclass(name = "Weighting", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
enum WeightingPy {
    Uniform,
    InverseDistance,
    SmoothedInverse,
    Gaussian,
}

/// Python wrapper around the Rust `KnnClassifier`.
#[pyclass(name = "KnnClassifier")]
struct KnnClassifierPy {
    inner: crate::KnnClassifier,
}

#[pymethods]
impl KnnClassifierPy {
    /// Create a new classifier with `k` neighbors.
    ///
    /// Raises `ValueError` if `k == 0`.
    #[new]
    #[pyo3(signature = (k, metric=None, weighting=None, algorithm=None))]
    fn new(
        k: usize,
        metric: Option<MetricPy>,
        weighting: Option<WeightingPy>,
        algorithm: Option<AlgorithmPy>,
    ) -> PyResult<Self> {
        if k == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "k must be positive",
            ));
        }

        let mut inner = crate::KnnClassifier::new(k);

        if let Some(m) = metric {
            let metric = match m {
                MetricPy::Euclidean => crate::distance::Metric::Euclidean,
                MetricPy::Manhattan => crate::distance::Metric::Manhattan,
                MetricPy::Cosine => crate::distance::Metric::Cosine,
            };
            inner.with_metric(metric);
        }

        if let Some(w) = weighting {
            let weighting = match w {
                WeightingPy::Uniform => crate::weights::Weighting::Uniform,
                WeightingPy::InverseDistance => crate::weights::Weighting::InverseDistance,
                WeightingPy::SmoothedInverse => crate::weights::Weighting::SmoothedInverse,
                WeightingPy::Gaussian => crate::weights::Weighting::Gaussian,
            };
            inner.with_weighting(weighting);
        }

        if let Some(a) = algorithm {
            let algorithm = match a {
                AlgorithmPy::BruteForce => crate::algorithm::Algorithm::BruteForce,
                AlgorithmPy::KdTree => crate::algorithm::Algorithm::KdTree,
            };
            inner.with_algorithm(algorithm);
        }

        Ok(Self { inner })
    }

    /// Fit on 2D training data and 1D labels (NumPy arrays).
    ///
    /// Data is copied into a flat `Vec<f64>` in row-major order.
    ///
    /// Raises `ValueError` if:
    /// - the algorithm is `KdTree` and the metric is `Cosine`
    /// - `k` is larger than the number of training points
    /// - the number of labels does not match the number of training points
    fn fit(&mut self, x: &Bound<'_, PyArray2<f64>>, y: &Bound<'_, PyArray1<i64>>) -> PyResult<()> {
        if self.inner.algorithm == crate::algorithm::Algorithm::KdTree
            && self.inner.metric == crate::distance::Metric::Cosine
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "KD-tree pruning is invalid for cosine distance; use Algorithm.BruteForce with Metric.Cosine",
            ));
        }

        let x_read = x.readonly();
        let x_view = x_read.as_array();
        let y_read = y.readonly();
        let y_view = y_read.as_array();

        let n_samples = x_view.nrows();
        let dim = x_view.ncols();

        if self.inner.k > n_samples {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "k ({}) cannot be larger than the number of training points ({})",
                self.inner.k, n_samples
            )));
        }

        if y_view.len() != n_samples {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} labels (one per training point), got {}",
                n_samples,
                y_view.len()
            )));
        }

        let mut data = Vec::with_capacity(n_samples * dim);
        for row in x_view.rows() {
            data.extend(row.iter().cloned());
        }

        let labels: Vec<usize> = y_view.iter().map(|&label| label as usize).collect();

        self.inner.fit(data, dim, labels);
        Ok(())
    }

    /// Predict labels for 2D query data (NumPy array).
    ///
    /// Returns a `list[int]` of predicted labels, one per query row.
    ///
    /// Raises `ValueError` if:
    /// - `fit` has not been called
    /// - the query dimension does not match the training dimension
    /// - cosine distance encounters a zero vector (in the training set or a query)
    fn predict(&self, x: &Bound<'_, PyArray2<f64>>) -> PyResult<Vec<usize>> {
        if self.inner.dim == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "model has not been fit; call fit() before predict()",
            ));
        }

        let x_read = x.readonly();
        let x_view = x_read.as_array();

        let n_queries = x_view.nrows();
        let dim = x_view.ncols();

        if dim != self.inner.dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "query dimension ({}) does not match training dimension ({})",
                dim, self.inner.dim
            )));
        }

        let mut queries = Vec::with_capacity(n_queries * dim);
        for row in x_view.rows() {
            queries.extend(row.iter().cloned());
        }

        // Check for zero vectors when using cosine distance (would panic in Rust).
        if self.inner.metric == crate::distance::Metric::Cosine {
            for i in 0..n_queries {
                let row = &queries[i * dim..(i + 1) * dim];
                let norm: f64 = row.iter().map(|&v| v * v).sum::<f64>().sqrt();
                if norm == 0.0 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "cosine distance is undefined for the zero vector (query row)",
                    ));
                }
            }
        }

        let predictions = self.inner.predict(&queries, n_queries, dim);
        Ok(predictions)
    }
}
