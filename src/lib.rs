//! A pure-Rust brute-force k-nearest neighbors classifier.
//!
//! This is intentionally simple: we own our data (no lifetimes yet),
//! use nested Vecs (easy to reason about, if not optimal), and defer
//! traits/parallelism/PyO3 to later weeks.

use std::collections::HashMap;
pub mod distance;
pub mod python;

/// Brute-force KNN classifier.
///
/// **Why `Vec<Vec<f64>>` for data?**
/// - *Pros*: Pure standard-library Rust. Each point is a self-contained `Vec<f64>`,
///   which mirrors how you might store a list of lists in Python.
/// - *Cons*: Each inner `Vec` is a separate heap allocation. During distance computation
///   the CPU chases pointers and gets poor cache locality. Also hard to SIMD.
/// - *The plan*: In Week 2–3 we’ll migrate to a flat `Vec<f64>` with a known `dim`,
///   then to `ndarray` or directly to NumPy-backed buffers via PyO3.
///
/// **Why owned data instead of borrowed slices?**
/// - If we borrowed, the struct would need a lifetime parameter: `KnnClassifier<'a>`.
///   That lifetime would infect every function signature and every call site.
/// - For a first Rust project, lifetimes-on-structs are a common stumbling block.
///   By owning the data (`Vec` moves into us), we side-step that entirely.
///   (TRPL Ch. 10 covers lifetimes; try applying them to this struct later as an exercise.)
///
/// **Why `usize` for labels?**
/// - We could make `KnnClassifier` generic over `Label: Clone + Eq + Hash`, but
///   trait bounds on structs add cognitive load in Week 1. `usize` covers
///   classification indices. We’ll generalize once the Rust basics feel natural.
pub struct KnnClassifier {
    // TODO (you will write this): training data and labels, plus k.
    data: Vec<Vec<f64>>,
    labels: Vec<usize>,
    k: usize,
}

impl KnnClassifier {
    /// Create a new classifier with the given `k`.
    ///
    /// No data is stored yet; call `fit` next.
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "k must be positive");
        Self {
            data: Vec::new(),
            labels: Vec::new(),
            k,
        }
    }

    /// Store training data and labels.
    ///
    /// **Ownership note:** `data` and `labels` are *moved* into this function.
    /// After `fit` returns, the caller no longer owns those Vecs — this struct does.
    /// That is Rust’s default: passing a Vec by value transfers ownership.
    /// (TRPL Ch. 4.1 — Ownership.)
    pub fn fit(&mut self, data: Vec<Vec<f64>>, labels: Vec<usize>) {
        // TODO: store the data.
        assert_eq!(data.len(), labels.len(), "data and labels must have the same length");
        assert!(self.k <= data.len(), "k ({}) cannot be larger than size of training set ({})", self.k, data.len());
        self.data = data;
        self.labels = labels;
    }

    /// Predict the class label for each query point.
    ///
    /// **Why `&[Vec<f64>]` for `queries`?**
    /// - We only need to *read* the query points, not own them.
    /// - `&[Vec<f64>]` is a borrowed slice of owned rows. This is slightly asymmetric
    ///   (we own training data but borrow queries), which is fine: queries come from
    ///   the caller and might be reused elsewhere.
    /// - Alternative: `&[&[f64]]` would borrow everything, but then callers must
    ///   construct slices-of-slices. We’ll revisit when we switch to flat arrays.
    ///
    /// Returns a `Vec<usize>` of predicted labels, one per query.
    pub fn predict(&self, queries: &[Vec<f64>]) -> Vec<usize> {
        let mut predictions: Vec<usize> = Vec::with_capacity(queries.len());
        
        for query in queries {
            let mut distances: Vec<(f64, usize)> = Vec::with_capacity(self.data.len());
            for (point, &label) in self.data.iter().zip(self.labels.iter()) {
                let dist = euclidean_distance(query, point);
                distances.push((dist, label));
            }

            let kth = self.k;
            distances.select_nth_unstable_by(kth -1, |a, b| a.0.total_cmp(&b.0));

            let neighbour_labels: Vec<usize> = distances[..self.k]
                .iter()
                .map(|(_, label)| *label)
                .collect();

            predictions.push(majority_vote(&neighbour_labels));
        }
        predictions
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Compute Euclidean distance between two equal-length vectors.
///
/// **Why slices (`&[f64]`) here?**
/// - We only need read access, and slices are the most flexible/readable
///   one-dimensional view in Rust. `&Vec<f64>` would also work, but `&[f64]`
///   is more idiomatic because it accepts `&Vec`, array references, and
///   any other contiguous data. (TRPL Ch. 4.3 — Slices.)
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "points must be of the same dimension");
    a.iter().zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Given the labels of the k nearest neighbors, return the most common one.
///
/// **Design choice:**
/// - This is a simple brute-force vote. In Week 4 we’ll switch to a weighted vote
///   (closer neighbors count more), which is why I’ve pulled it into its own helper
///   rather than inlining it inside `predict`.
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
        model.fit(
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            vec![0, 1],
        );
        assert_eq!(model.data.len(), 2);
        assert_eq!(model.data, vec![vec![0.0, 0.0], vec![1.0, 1.0]]);
        assert_eq!(model.labels.len(), 2);
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
        model.fit(
            vec![vec![0.0], vec![1.0]],
            vec![0, 1],
        ); // should not panic
    }

    #[test]
    #[should_panic(expected = "larger than size")]
    fn test_k_larger_than_data() {
        let mut model = KnnClassifier::new(5);
        model.fit(vec![vec![0.0]], vec![0]);
    }


    #[test]
    #[should_panic(expected="same dimension")]
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
        // Classic right-triangle test: sqrt(3^2 + 4^2) = 5
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
        assert!((actual - expected).abs() < 1e-12, "expected {}, got {}", expected, actual);
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
