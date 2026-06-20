//! A k-nearest neighbors classifier with pluggable search algorithms.
//!
//! Training data is stored in a single flat `Vec<f64>` for cache-friendly
//! distance computation. Two search algorithms are supported: brute-force
//! (linear scan) and KD-tree (spatial partitioning). Parallel prediction is
//! supported via `rayon`.

use algorithm::Algorithm;
use distance::Metric;
use kdtree::{KdTree, build_kdtree};
use rayon::prelude::*;
use std::collections::HashMap;
use weights::Weighting;

pub mod algorithm;
pub mod brute;
pub mod distance;
pub mod kdtree;
pub mod weights;
#[cfg(feature = "python")]
pub mod python;

/// k-nearest neighbors classifier.
///
/// The classifier owns its training data. After [`fit`](Self::fit), points are
/// stored in a flat row-major buffer: `[p0f0, p0f1, p1f0, p1f1, ...]`.
///
/// Predictions find the `k` nearest training points to each query, then return
/// the majority (or distance-weighted) label. Ties are resolved by choosing
/// the smallest label.
///
/// Two search algorithms are available via [`with_algorithm`](Self::with_algorithm):
/// - [`Algorithm::BruteForce`] — linear scan over all training points
/// - [`Algorithm::KdTree`] — recursive spatial partitioning for O(log n) average search
pub struct KnnClassifier {
    data: Vec<f64>,
    dim: usize,
    labels: Vec<usize>,
    /// Number of neighbors used for each prediction.
    pub k: usize,
    /// Distance metric used to compare query points with training points.
    pub metric: Metric,
    /// Weighting function applied to distances.
    pub weighting: Weighting,
    /// Algorithm for finding neighbors
    pub algorithm: Algorithm,
    /// Tree used when algorithm == KdTree
    tree: Option<KdTree>,
}

impl KnnClassifier {
    /// Create a classifier that uses Euclidean distance.
    ///
    /// # Panics
    ///
    /// Panics if `k == 0`.
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "k must be positive");
        Self {
            data: Vec::new(),
            dim: 0,
            labels: Vec::new(),
            k,
            metric: Metric::Euclidean,
            weighting: Weighting::Uniform,
            algorithm: Algorithm::BruteForce,
            tree: None,
        }
    }

    pub fn with_metric(&mut self, metric: Metric) -> &mut Self {
        self.metric = metric;
        self
    }

    pub fn with_weighting(&mut self, weighting: Weighting) -> &mut Self {
        self.weighting = weighting;
        self
    }

    pub fn with_algorithm(&mut self, algorithm: Algorithm) -> &mut Self {
        self.algorithm = algorithm;
        self
    }

    /// Fit the classifier on training data in flat row-major layout.
    ///
    /// `data` must contain `n_points * dim` values, laid out as contiguous
    /// rows. For example, three two-dimensional points are represented as
    /// `[p0x, p0y, p1x, p1y, p2x, p2y]`.
    ///
    /// `labels` must contain one class label per training point.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `dim == 0`
    /// - `data.len()` is not divisible by `dim`
    /// - `labels.len()` does not match the number of training points
    /// - `self.k` is larger than the number of training points
    /// - the algorithm is `KdTree` and the metric is `Cosine` (KD-tree pruning
    ///   is invalid for cosine distance because it does not satisfy the
    ///   triangle inequality in axis-aligned space)
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
        assert!(
            !(self.algorithm == Algorithm::KdTree && self.metric == Metric::Cosine),
            "KD-tree pruning is invalid for cosine distance; use Algorithm::BruteForce with Metric::Cosine"
        );
        self.data = data;
        self.dim = dim;
        self.labels = labels;

        if self.algorithm == Algorithm::KdTree {
            self.tree = Some(build_kdtree(&self.data, dim));
        }
    }

    /// Predict labels for query points in flat row-major layout.
    ///
    /// `queries` must contain `n_queries * dim` values, using the same feature
    /// dimension passed to [`fit`](Self::fit). The returned vector contains one
    /// predicted label per query row.
    ///
    /// Uses `rayon` to parallelize across queries.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `queries.len() != n_queries * dim`
    /// - `dim` does not match the fitted training dimension
    /// - the selected [`Metric`] panics for any training/query pair
    /// - [`Algorithm::KdTree`] is selected but the tree was not built
    pub fn predict(&self, queries: &[f64], n_queries: usize, dim: usize) -> Vec<usize> {
        assert_eq!(queries.len(), n_queries * dim);
        assert_eq!(
            dim, self.dim,
            "query dimension must match training data dimension"
        );

        match self.algorithm {
            Algorithm::BruteForce => self.predict_brute(queries, dim),
            Algorithm::KdTree => self.predict_kdtree(queries, dim),
        }
    }

    /// Brute-force prediction: linear scan over all training points.
    fn predict_brute(&self, queries: &[f64], dim: usize) -> Vec<usize> {
        queries
            .par_chunks(dim)
            .map(|query| {
                let neighbors =
                    brute::brute_force_search(query, &self.data, self.dim, self.k, &self.metric);

                let labeled: Vec<(f64, usize)> = neighbors
                    .iter()
                    .map(|&(dist, idx)| (dist, self.labels[idx]))
                    .collect();

                weighted_vote(&labeled, &self.weighting)
            })
            .collect()
    }

    /// KD-tree prediction: recursive spatial search with pruning.
    fn predict_kdtree(&self, queries: &[f64], dim: usize) -> Vec<usize> {
        let tree = self.tree.as_ref().expect("KD-tree not built");

        queries
            .par_chunks(dim)
            .map(|query| {
                let neighbors = tree.k_nearest(query, &self.data, self.k, &self.metric);

                let labeled: Vec<(f64, usize)> = neighbors
                    .iter()
                    .map(|&(dist, idx)| (dist, self.labels[idx]))
                    .collect();

                weighted_vote(&labeled, &self.weighting)
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

fn weighted_vote(top_k: &[(f64, usize)], weighting: &Weighting) -> usize {
    let mut totals: HashMap<usize, f64> = HashMap::new();

    for &(distance, label) in top_k {
        let weight = weighting.weight(distance);

        if weight.is_infinite() {
            return label;
        }

        *totals.entry(label).or_insert(0.0) += weight;
    }

    let mut best_label = top_k[0].1;
    let mut best_total = 0.0;

    for (&label, &total) in &totals {
        if total > best_total || (total == best_total && label < best_label) {
            best_label = label;
            best_total = total;
        }
    }

    best_label
}

#[cfg(test)]
mod tests {
    use super::*;

    mod classifier {
        use super::*;

        #[test]
        fn new_default_metric_is_euclidean() {
            let model = KnnClassifier::new(3);
            assert_eq!(model.metric, Metric::Euclidean);
        }

        #[test]
        fn new_default_algorithm_is_bruteforce() {
            let model = KnnClassifier::new(3);
            assert_eq!(model.algorithm, Algorithm::BruteForce);
        }

        #[test]
        fn new_sets_k() {
            let model = KnnClassifier::new(1);
            assert_eq!(model.k, 1);
            assert_eq!(model.data.len(), 0);
            assert_eq!(model.labels.len(), 0);
        }

        #[test]
        #[should_panic(expected = "k must be positive")]
        fn new_zero_k_panics() {
            KnnClassifier::new(0);
        }

        #[test]
        #[should_panic(expected = "k must be positive")]
        fn with_metric_zero_k_panics() {
            KnnClassifier::new(0).with_metric(Metric::Manhattan);
        }

        #[test]
        fn with_algorithm_sets_algorithm() {
            let mut model = KnnClassifier::new(3);
            model.with_algorithm(Algorithm::KdTree);
            assert_eq!(model.algorithm, Algorithm::KdTree);
        }

        #[test]
        fn fit_stores_data() {
            let mut model = KnnClassifier::new(2);
            let (data, dim) = rows_to_flat(&[vec![0.0, 0.0], vec![1.0, 1.0]]);
            model.fit(data, dim, vec![0, 1]);
            assert_eq!(model.data, vec![0.0, 0.0, 1.0, 1.0]);
            assert_eq!(model.labels, vec![0, 1]);
        }

        #[test]
        #[should_panic(expected = "same length")]
        fn fit_mismatched_lengths_panics() {
            let mut model = KnnClassifier::new(1);
            model.fit(vec![0.0], 1, vec![0, 1]);
        }

        #[test]
        fn fit_k_equals_data_len() {
            let mut model = KnnClassifier::new(2);
            model.fit(vec![0.0, 1.0], 1, vec![0, 1]);
        }

        #[test]
        #[should_panic(expected = "cannot be larger than training set size")]
        fn fit_k_larger_than_data_panics() {
            let mut model = KnnClassifier::new(5);
            model.fit(vec![0.0], 1, vec![0]);
        }

        #[test]
        #[should_panic(expected = "KD-tree pruning is invalid for cosine")]
        fn fit_kdtree_with_cosine_panics() {
            let mut model = KnnClassifier::new(1);
            model.with_metric(Metric::Cosine);
            model.with_algorithm(Algorithm::KdTree);
            let (data, dim) = rows_to_flat(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
            model.fit(data, dim, vec![0, 1]);
        }

        #[test]
        fn fit_kdtree_with_euclidean_does_not_panic() {
            let mut model = KnnClassifier::new(1);
            model.with_algorithm(Algorithm::KdTree);
            let (data, dim) = rows_to_flat(&[vec![0.0, 0.0], vec![1.0, 1.0]]);
            model.fit(data, dim, vec![0, 1]);
        }

        #[test]
        fn fit_bruteforce_with_cosine_does_not_panic() {
            let mut model = KnnClassifier::new(1);
            model.with_metric(Metric::Cosine);
            let (data, dim) = rows_to_flat(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
            model.fit(data, dim, vec![0, 1]);
        }

        #[test]
        fn fit_does_not_build_tree_when_bruteforce() {
            let mut model = KnnClassifier::new(2);
            let (data, dim) = rows_to_flat(&[vec![0.0, 0.0], vec![1.0, 1.0]]);
            model.fit(data, dim, vec![0, 1]);
            assert!(model.tree.is_none());
        }

        #[test]
        fn fit_builds_tree_when_kdtree() {
            let mut model = KnnClassifier::new(2);
            model.with_algorithm(Algorithm::KdTree);
            let (data, dim) = rows_to_flat(&[vec![0.0, 0.0], vec![1.0, 1.0]]);
            model.fit(data, dim, vec![0, 1]);
            assert!(model.tree.is_some());
        }

        #[test]
        #[should_panic]
        fn predict_dimension_mismatch_panics() {
            let mut model = KnnClassifier::new(1);
            model.fit(vec![0.0, 0.0, 1.0, 0.0], 2, vec![0, 1]);
            let queries = vec![0.0, 0.0, 0.0];
            model.predict(&queries, 1, 3);
        }

        #[test]
        fn kdtree_predictions_match_bruteforce() {
            let (data, dim) = rows_to_flat(&[
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 10.0],
                vec![10.0, 10.0],
                vec![5.0, 5.0],
            ]);
            let labels = vec![0, 0, 1, 1, 2];
            let (queries, query_dim) = rows_to_flat(&[
                vec![0.1, 0.1],
                vec![9.0, 9.0],
                vec![5.0, 5.0],
                vec![0.0, 9.0],
            ]);

            let mut brute_model = KnnClassifier::new(3);
            brute_model.fit(data.clone(), dim, labels.clone());
            let brute_preds = brute_model.predict(&queries, 4, query_dim);

            let mut kdtree_model = KnnClassifier::new(3);
            kdtree_model.with_algorithm(Algorithm::KdTree);
            kdtree_model.fit(data, dim, labels);
            let kdtree_preds = kdtree_model.predict(&queries, 4, query_dim);

            assert_eq!(brute_preds, kdtree_preds);
        }

        #[test]
        fn kdtree_manhattan_matches_bruteforce() {
            let (data, dim) = rows_to_flat(&[
                vec![0.0, 0.0],
                vec![1.0, 0.0],
                vec![0.0, 10.0],
                vec![10.0, 10.0],
                vec![5.0, 5.0],
            ]);
            let labels = vec![0, 0, 1, 1, 2];
            let (queries, query_dim) = rows_to_flat(&[
                vec![0.1, 0.1],
                vec![9.0, 9.0],
                vec![5.0, 5.0],
                vec![0.0, 9.0],
            ]);

            let mut brute_model = KnnClassifier::new(3);
            brute_model.with_metric(Metric::Manhattan);
            brute_model.fit(data.clone(), dim, labels.clone());
            let brute_preds = brute_model.predict(&queries, 4, query_dim);

            let mut kdtree_model = KnnClassifier::new(3);
            kdtree_model.with_metric(Metric::Manhattan);
            kdtree_model.with_algorithm(Algorithm::KdTree);
            kdtree_model.fit(data, dim, labels);
            let kdtree_preds = kdtree_model.predict(&queries, 4, query_dim);

            assert_eq!(brute_preds, kdtree_preds);
        }
    }

    mod weighted_vote {
        use super::*;

        #[test]
        fn uniform_simple() {
            let top_k = [(0.1, 0), (0.5, 0), (0.9, 1)];
            assert_eq!(weighted_vote(&top_k, &Weighting::Uniform), 0);
        }

        #[test]
        fn uniform_unanimous() {
            let top_k = [(1.0, 1), (2.0, 1), (3.0, 1)];
            assert_eq!(weighted_vote(&top_k, &Weighting::Uniform), 1);
        }

        #[test]
        fn uniform_single() {
            let top_k = [(1.0, 5)];
            assert_eq!(weighted_vote(&top_k, &Weighting::Uniform), 5);
        }

        #[test]
        fn uniform_larger_k_multiple_classes() {
            let top_k = [(1.0, 0), (2.0, 0), (3.0, 0), (4.0, 1), (5.0, 2)];
            assert_eq!(weighted_vote(&top_k, &Weighting::Uniform), 0);
        }

        #[test]
        fn uniform_single_label() {
            let top_k = [(1.0, 5), (2.0, 5), (3.0, 5)];
            assert_eq!(weighted_vote(&top_k, &Weighting::Uniform), 5);
        }

        #[test]
        fn inverse_distance_closer_wins() {
            let top_k = [(0.1, 1), (2.0, 0)];
            assert_eq!(weighted_vote(&top_k, &Weighting::InverseDistance), 1);
        }

        #[test]
        fn inverse_distance_zero_short_circuits() {
            let top_k = [(0.0, 2), (0.1, 0), (0.1, 1)];
            assert_eq!(weighted_vote(&top_k, &Weighting::InverseDistance), 2);
        }

        #[test]
        fn smoothed_inverse_no_infinity() {
            let top_k = [(0.0, 1), (1.0, 0), (1.0, 0)];
            assert_eq!(weighted_vote(&top_k, &Weighting::SmoothedInverse), 0);
        }

        #[test]
        fn gaussian_closer_wins() {
            // Gaussian weight = exp(-d). Closer point has higher weight.
            // (0.1, label 1) -> exp(-0.1) ~ 0.905
            // (2.0, label 0) -> exp(-2.0) ~ 0.135
            // Label 1 wins.
            let top_k = [(0.1, 1), (2.0, 0)];
            assert_eq!(weighted_vote(&top_k, &Weighting::Gaussian), 1);
        }

        #[test]
        fn tie_breaks_by_smallest_label() {
            let top_k = [(1.0, 2), (1.0, 0), (1.0, 1)];
            assert_eq!(weighted_vote(&top_k, &Weighting::Uniform), 0);
        }
    }
}
