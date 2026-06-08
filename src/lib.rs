//! A pure-Rust brute-force k-nearest neighbors classifier.
//!
//! Training data is stored in a single flat `Vec<f64>` for cache-friendly
//! distance computation. Parallel prediction is supported via `rayon`.

use distance::Metric;
use rayon::prelude::*;
use std::collections::HashMap;

pub mod distance;
pub mod python;

/// Brute-force KNN classifier.
///
/// Data is stored in row-major layout: `[p0f0, p0f1, p1f0, p1f1, ...]`.
pub struct KnnClassifier {
    data: Vec<f64>,
    dim: usize,
    labels: Vec<usize>,
    pub k: usize,
    pub metric: Metric,
}

impl KnnClassifier {
    /// Create a new classifier with the given `k`.
    ///
    /// Panics if `k == 0`. Default distance metric is euclidean
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "k must be positive");
        Self {
            data: Vec::new(),
            dim: 0,
            labels: Vec::new(),
            k,
            metric: Metric::Euclidean,
        }
    }

    pub fn with_metric(k: usize, metric: Metric) -> Self {
        assert!(k > 0, "k must be positive");
        Self {
            data: Vec::new(),
            dim: 0,
            labels: Vec::new(),
            k,
            metric,
        }
    }

    /// Fit on data in flat row-major layout.
    ///
    /// `data.len()` must equal `n_points * dim`.
    pub fn fit(&mut self, data: Vec<f64>, dim: usize, labels: Vec<usize>) {
        assert_eq!(data.len() % dim, 0);
        let n_points = data.len() / dim;
        assert_eq!(
            n_points,
            labels.len(),
            "data and labels must have the same length"
        );
        assert!(
            self.k <= n_points,
            "k ({}) cannot be larger than training set size ({})",
            self.k,
            n_points
        );
        self.data = data;
        self.dim = dim;
        self.labels = labels;
    }

    /// Predict labels for queries in flat row-major layout.
    ///
    /// Uses `rayon` to parallelize across queries.
    pub fn predict(&self, queries: &[f64], n_queries: usize, dim: usize) -> Vec<usize> {
        assert_eq!(queries.len(), n_queries * dim);
        assert_eq!(
            dim, self.dim,
            "query dimension must match training data dimension"
        );

        queries
            .par_chunks(dim)
            .map(|query| {
                let mut distances: Vec<(f64, usize)> = Vec::with_capacity(self.labels.len());
                for i in 0..self.labels.len() {
                    let point = &self.data[i * self.dim..(i + 1) * self.dim];
                    distances.push((self.metric.distance(point, query), self.labels[i]));
                }

                distances.select_nth_unstable_by(self.k - 1, |a, b| a.0.total_cmp(&b.0));

                let top_labels: Vec<usize> = distances[..self.k]
                    .iter()
                    .map(|(_, label)| *label)
                    .collect();

                majority_vote(&top_labels)
            })
            .collect()
    }
}

/// Convert row-oriented test data into flat row-major layout.
#[cfg(test)]
fn rows_to_flat(rows: &[Vec<f64>]) -> (Vec<f64>, usize) {
    assert!(!rows.is_empty(), "rows must not be empty");
    let dim = rows[0].len();
    let mut flat = Vec::with_capacity(rows.len() * dim);
    for row in rows {
        assert_eq!(row.len(), dim, "all rows must have the same dimension");
        flat.extend(row);
    }
    (flat, dim)
}

/// Given the labels of the k nearest neighbors, return the most common one.
///
/// Ties are broken by choosing the smallest label.
fn majority_vote(neighbor_labels: &[usize]) -> usize {
    let mut counts = HashMap::new();

    for &label in neighbor_labels {
        *counts.entry(label).or_insert(0) += 1;
    }

    let mut best_label = neighbor_labels[0];
    let mut best_count = 0;

    for (&label, &count) in &counts {
        if count > best_count || (count == best_count && label < best_label) {
            best_count = count;
            best_label = label;
        }
    }

    best_label
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_successful_new_classifier() {
        let model = KnnClassifier::new(1);
        assert_eq!(model.k, 1);
        assert_eq!(model.data.len(), 0);
        assert_eq!(model.labels.len(), 0);
    }

    #[test]
    #[should_panic(expected = "k must be positive")]
    fn test_new_knn_zero_k() {
        KnnClassifier::new(0);
    }

    #[test]
    fn test_successful_fit() {
        let mut model = KnnClassifier::new(2);
        let (data, dim) = rows_to_flat(&[vec![0.0, 0.0], vec![1.0, 1.0]]);
        model.fit(data, dim, vec![0, 1]);
        assert_eq!(model.data, vec![0.0, 0.0, 1.0, 1.0]);
        assert_eq!(model.labels, vec![0, 1]);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_mismatched_lengths() {
        let mut model = KnnClassifier::new(1);
        model.fit(vec![0.0], 1, vec![0, 1]);
    }

    #[test]
    fn test_k_equals_data_len() {
        let mut model = KnnClassifier::new(2);
        model.fit(vec![0.0, 1.0], 1, vec![0, 1]);
    }

    #[test]
    #[should_panic(expected = "cannot be larger than training set size")]
    fn test_k_larger_than_data() {
        let mut model = KnnClassifier::new(5);
        model.fit(vec![0.0], 1, vec![0]);
    }

    #[test]
    #[should_panic(expected = "same dimension")]
    fn test_euclidean_dist_diff_dimensions() {
        let x = vec![1.0, 0.0];
        let y = vec![0.0];
        Metric::Euclidean.distance(&x, &y);
    }

    #[test]
    fn test_euclidean_dist_of_zero() {
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![1.0, 1.0, 1.0];
        assert_eq!(Metric::Euclidean.distance(&x, &y), 0.0);
    }

    #[test]
    fn test_euclidean_dist_3_4_5() {
        assert_eq!(Metric::Euclidean.distance(&[0.0, 0.0], &[3.0, 4.0]), 5.0);
    }

    #[test]
    fn test_euclidean_dist_1d() {
        assert_eq!(Metric::Euclidean.distance(&[0.0], &[5.0]), 5.0);
        assert_eq!(Metric::Euclidean.distance(&[3.0], &[8.0]), 5.0);
    }

    #[test]
    fn test_euclidean_dist_negative_coords() {
        assert_eq!(Metric::Euclidean.distance(&[-3.0, -4.0], &[0.0, 0.0]), 5.0);
        assert_eq!(Metric::Euclidean.distance(&[-1.0, 2.0], &[2.0, 6.0]), 5.0);
    }

    #[test]
    fn test_euclidean_dist_3d() {
        let expected = (14.0_f64).sqrt();
        let actual = Metric::Euclidean.distance(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]);
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {}, got {}",
            expected,
            actual
        );
    }

    #[test]
    fn test_majority_vote_simple() {
        assert_eq!(majority_vote(&[0, 0, 1]), 0);
    }

    #[test]
    fn test_majority_vote_unanimous() {
        assert_eq!(majority_vote(&[1, 1, 1]), 1);
    }

    #[test]
    fn test_majority_vote_single() {
        assert_eq!(majority_vote(&[5]), 5);
    }

    #[test]
    fn test_majority_vote_tie() {
        assert_eq!(majority_vote(&[1, 0, 2]), 0);
    }
}
