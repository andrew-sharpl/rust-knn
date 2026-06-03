// Integration test for knn classifier.
//
// These tests import the crate as an external user would:
//   `use knn::KnnClassifier;`
//
// Because they’re in tests/, they cannot access private helpers like
// `euclidean_distance` or `majority_vote`. That’s intentional: it tests
// the public contract.
//
// **Rust testing basics (TRPL Ch. 11):**
// - `#[test]` marks a function as a test.
// - `assert_eq!(left, right)` panics on mismatch.
// - Run with: `cargo test`

use std::vec;
use knn::KnnClassifier;

#[test]
fn test_knn_basic() {
    let mut model = KnnClassifier::new(1);
    model.fit(
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 10.0]],
        vec![0, 0, 1],
    );

    let predictions = model.predict(&[vec![0.1, 0.0]]);
    assert_eq!(predictions, vec![0]);


    let predictions = model.predict(&[vec![0.0, 9.0]]);
    assert_eq!(predictions, vec![1]);
}

#[test]
fn test_knn_multiple_queries() {
    let mut model = KnnClassifier::new(2);
    model.fit(
        vec![vec![0.0], vec![1.0], vec![10.0], vec![11.0]],
        vec![0, 0, 1, 1],
    );

    let predictions = model.predict(&[vec![0.4], vec![10.4]]);
    assert_eq!(predictions, vec![0, 1]);
}

#[test]
fn test_knn_k3_ignores_far_points() {
    let mut model = KnnClassifier::new(4);
    model.fit(
        vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![100.0, 100.0], // class 0 but far away
            vec![0.0, 10.0],    // class 1, also far
        ],
        vec![0, 0, 0, 0, 1],
    );

    let predictions = model.predict(&[vec![0.05, 0.05]]);
    assert_eq!(predictions, vec![0]);
}
