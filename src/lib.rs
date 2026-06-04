//! A pure-Rust brute-force k-nearest neighbors classifier.
//!
//! Training data is stored in a single flat `Vec<f64>` for cache-friendly
//! distance computation. Parallel prediction is supported via `rayon`.

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
    k: usize,
}

impl KnnClassifier {
    /// Create a new classifier with the given `k`.
    ///
    /// Panics if `k == 0`.
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "k must be positive");
        Self {
            data: Vec::new(),
            dim: 0,
            labels: Vec::new(),
            k,
        }
    }

    /// Fit on data provided as a vector of rows.
    ///
    /// Each inner `Vec` is one sample. This method flattens internally
    /// into a contiguous buffer.
    pub fn fit(&mut self, data: Vec<Vec<f64>>, labels: Vec<usize>) {
        assert_eq!(
            data.len(),
            labels.len(),
            "data and labels must have the same length"
        );
        assert!(
            self.k <= data.len(),
            "k ({}) cannot be larger than training set size ({ })",
            self.k,
            data.len()
        );

        self.dim = data[0].len();
        self.data = Vec::with_capacity(data.len() * self.dim);
        for point in data {
            assert_eq!(
                point.len(),
                self.dim,
                "all rows must have the same dimension"
            );
            self.data.extend(point);
        }
        self.labels = labels;
    }

    /// Fit on data already in flat row-major layout.
    ///
    /// `data.len()` must equal `n_points * dim`.
    pub fn fit_flat(&mut self, data: Vec<f64>, dim: usize, labels: Vec<usize>) {
        assert_eq!(data.len() % dim, 0);
        let n_points = data.len() / dim;
        assert_eq!(n_points, labels.len());
        assert!(self.k <= n_points);
        self.data = data;
        self.dim = dim;
        self.labels = labels;
    }

    /// Predict labels for queries in flat row-major layout.
    ///
    /// Uses `rayon` to parallelize across queries.
    pub fn predict_flat(&self, queries: &[f64], n_queries: usize, dim: usize) -> Vec<usize> {
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
                    distances.push((euclidean_distance(point, query), self.labels[i]));
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

    /// Predict labels for queries provided as a vector of rows.
    pub fn predict(&self, queries: &[Vec<f64>]) -> Vec<usize> {
        let mut predictions: Vec<usize> = Vec::with_capacity(queries.len());

        for query in queries {
            let mut distances: Vec<(f64, usize)> = Vec::with_capacity(self.labels.len());

            for i in 0..self.labels.len() {
                let point = &self.data[i * self.dim..(i + 1) * self.dim];
                let dist = euclidean_distance(point, query);
                distances.push((dist, self.labels[i]));
            }

            distances.select_nth_unstable_by(self.k - 1, |a, b| a.0.total_cmp(&b.0));

            let neighbour_labels: Vec<usize> = distances[..self.k]
                .iter()
                .map(|(_, label)| *label)
                .collect();

            predictions.push(majority_vote(&neighbour_labels));
        }
        predictions
    }
}

/// Compute Euclidean distance between two equal-length vectors.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "points must be of the same dimension");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
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
        model.fit(vec![vec![0.0, 0.0], vec![1.0, 1.0]], vec![0, 1]);
        assert_eq!(model.data, vec![0.0, 0.0, 1.0, 1.0]);
        assert_eq!(model.labels, vec![0, 1]);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn test_mismatched_lengths() {
        let mut model = KnnClassifier::new(1);
        model.fit(vec![vec![0.0]], vec![0, 1]);
    }

    #[test]
    fn test_k_equals_data_len() {
        let mut model = KnnClassifier::new(2);
        model.fit(vec![vec![0.0], vec![1.0]], vec![0, 1]);
    }

    #[test]
    #[should_panic(expected = "cannot be larger than training set size")]
    fn test_k_larger_than_data() {
        let mut model = KnnClassifier::new(5);
        model.fit(vec![vec![0.0]], vec![0]);
    }

    #[test]
    #[should_panic(expected = "same dimension")]
    fn test_euclidean_dist_diff_dimensions() {
        let x = vec![1.0, 0.0];
        let y = vec![0.0];
        euclidean_distance(&x, &y);
    }

    #[test]
    fn test_euclidean_dist_of_zero() {
        let x = vec![1.0, 1.0, 1.0];
        let y = vec![1.0, 1.0, 1.0];
        assert_eq!(euclidean_distance(&x, &y), 0.0);
    }

    #[test]
    fn test_euclidean_dist_3_4_5() {
        assert_eq!(euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]), 5.0);
    }

    #[test]
    fn test_euclidean_dist_1d() {
        assert_eq!(euclidean_distance(&[0.0], &[5.0]), 5.0);
        assert_eq!(euclidean_distance(&[3.0], &[8.0]), 5.0);
    }

    #[test]
    fn test_euclidean_dist_negative_coords() {
        assert_eq!(euclidean_distance(&[-3.0, -4.0], &[0.0, 0.0]), 5.0);
        assert_eq!(euclidean_distance(&[-1.0, 2.0], &[2.0, 6.0]), 5.0);
    }

    #[test]
    fn test_euclidean_dist_3d() {
        let expected = (14.0_f64).sqrt();
        let actual = euclidean_distance(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]);
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
