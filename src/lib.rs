//! A pure-Rust brute-force k-nearest neighbors classifier.
//!
//! This is intentionally simple: we own our data (no lifetimes yet),
//! use nested Vecs (easy to reason about, if not optimal), and defer
//! traits/parallelism/PyO3 to later weeks.

pub mod distance;

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
    // Hint: `data: Vec<Vec<f64>>`, `labels: Vec<usize>`, `k: usize`
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
        // TODO: for each query:
        //   1. Compute Euclidean distance to every training point.
        //   2. Find the k smallest distances (naive sort or partial sort).
        //   3. Take a majority vote of their labels.
        // Consider: where does allocation happen? Can you reuse a buffer?
        todo!("Implement brute-force prediction")
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
    // TODO: count frequencies (HashMap<usize, usize>? or sort + run-length count?)
    // and return the label with the highest count.
    // Tie-breaking: what if two labels have the same count?
    todo!("Return the most common label")
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
        // (-3, -4) to (0, 0) should also be 5
        assert_eq!(euclidean_distance(&[-3.0, -4.0], &[0.0, 0.0]), 5.0);
        // (-1, 2) to (2, 6) -> sqrt(3^2 + 4^2) = 5
        assert_eq!(euclidean_distance(&[-1.0, 2.0], &[2.0, 6.0]), 5.0);
    }

    #[test]
    fn test_euclidean_dist_3d() {
        // 1-2-3 sqrt(14) triangle in 3D
        let expected = (14.0_f64).sqrt();
        let actual = euclidean_distance(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]);
        assert!((actual - expected).abs() < 1e-12, "expected {}, got {}", expected, actual);
    }

}
