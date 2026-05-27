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

use knn::KnnClassifier;

#[test]
fn test_knn_basic() {
    // TODO: Create a tiny 2D dataset where the answer is obvious.
    //
    // Example:
    //   Point (0.0, 0.0) -> class 0
    //   Point (1.0, 0.0) -> class 0
    //   Point (0.0, 10.0) -> class 1
    //
    // Then query (0.1, 0.0) with k=1 or k=2 and assert the predicted class.

    // let mut model = KnnClassifier::new(1);
    // model.fit(vec![ ... ], vec![ ... ]);
    // let predictions = model.predict(&[vec![0.1, 0.0]]);
    // assert_eq!(predictions, vec![0]);
}
