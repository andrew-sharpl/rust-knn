use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyArrayMethods};

#[pymodule]
fn knn(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<KnnClassifierPy>()?;
    Ok(())
}

#[pyclass(name = "KnnClassifier")]
struct KnnClassifierPy {
    inner: crate::KnnClassifier,
}

#[pymethods]
impl KnnClassifierPy {
    #[new]
    fn new(k: usize) -> Self { 
        Self {
            inner: crate::KnnClassifier::new(k),
        }
    }

    fn fit(&mut self, x: &Bound<'_, PyArray2<f64>>, y: &Bound<'_, PyArray1<i64>>) -> PyResult<()> {
        // Get zero-copy views into NumPy arrays 
        let x_read = x.readonly();
        let x_view = x_read.as_array();

        let y_read = y.readonly();
        let y_view = y_read.as_array();

        // Convert ndarray views to Vec<Vec<f64>> and Vec<usize>
        let data: Vec<Vec<f64>> = x_view.rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();

        let labels: Vec<usize> = y_view.iter()
            .map(|&label| label as usize)
            .collect();

        self.inner.fit(data, labels);
        Ok(())
    }

    fn predict(&self, x: &Bound<'_, PyArray2<f64>>) -> PyResult<Vec<usize>> {
        let x_read = x.readonly();
        let x_view = x_read.as_array();

        let queries: Vec<Vec<f64>> = x_view.rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect();

        let predictions = self.inner.predict(&queries);

        Ok(predictions)

    }
}
