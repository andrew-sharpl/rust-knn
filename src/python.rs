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
    Ok(())
}

/// Python-visible distance metric enum.
#[pyclass(name = "Metric", eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
enum MetricPy {
    Euclidean,
    Manhattan,
    Cosine,
}

/// Python wrapper around the Rust `KnnClassifier`.
#[pyclass(name = "KnnClassifier")]
struct KnnClassifierPy {
    inner: crate::KnnClassifier,
}

#[pymethods]
impl KnnClassifierPy {
    /// Create a new classifier with `k` neighbors.
    #[new]
    #[pyo3(signature = (k, metric=None))]
    fn new(k: usize, metric: Option<MetricPy>) -> Self {
        let metric = match metric {
            Some(m) => match m {
                MetricPy::Euclidean => crate::distance::Metric::Euclidean,
                MetricPy::Manhattan => crate::distance::Metric::Manhattan,
                MetricPy::Cosine => crate::distance::Metric::Cosine,
            },
            None => crate::distance::Metric::Euclidean,
        };
        Self {
            inner: crate::KnnClassifier::with_metric(k, metric),
        }
    }

    /// Fit on 2D training data and 1D labels (NumPy arrays).
    ///
    /// Data is copied into a flat `Vec<f64>` in row-major order.
    fn fit(&mut self, x: &Bound<'_, PyArray2<f64>>, y: &Bound<'_, PyArray1<i64>>) -> PyResult<()> {
        let x_read = x.readonly();
        let x_view = x_read.as_array();
        let y_read = y.readonly();
        let y_view = y_read.as_array();

        let n_samples = x_view.nrows();
        let dim = x_view.ncols();
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
    fn predict(&self, x: &Bound<'_, PyArray2<f64>>) -> PyResult<Vec<usize>> {
        let x_read = x.readonly();
        let x_view = x_read.as_array();

        let n_queries = x_view.nrows();
        let dim = x_view.ncols();
        let mut queries = Vec::with_capacity(n_queries * dim);
        for row in x_view.rows() {
            queries.extend(row.iter().cloned());
        }

        let predictions = self.inner.predict(&queries, n_queries, dim);
        Ok(predictions)
    }
}
