use knn::{KnnClassifier, distance::Metric};

fn rows_to_flat(rows: &[Vec<f64>]) -> (Vec<f64>, usize) {
    let dim = rows[0].len();
    let flat: Vec<f64> = rows.iter().flat_map(|row| row.iter().copied()).collect();
    (flat, dim)
}

#[test]
fn test_knn_basic() {
    let mut model = KnnClassifier::new(1);
    let (data, dim) = rows_to_flat(&[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 10.0]]);
    model.fit(data, dim, vec![0, 0, 1]);

    let (queries, query_dim) = rows_to_flat(&[vec![0.1, 0.0]]);
    let predictions = model.predict(&queries, 1, query_dim);
    assert_eq!(predictions, vec![0]);

    let (queries, query_dim) = rows_to_flat(&[vec![0.0, 9.0]]);
    let predictions = model.predict(&queries, 1, query_dim);
    assert_eq!(predictions, vec![1]);
}

#[test]
fn test_knn_multiple_queries() {
    let mut model = KnnClassifier::new(2);
    let (data, dim) = rows_to_flat(&[vec![0.0], vec![1.0], vec![10.0], vec![11.0]]);
    model.fit(data, dim, vec![0, 0, 1, 1]);

    let (queries, query_dim) = rows_to_flat(&[vec![0.4], vec![10.4]]);
    let predictions = model.predict(&queries, 2, query_dim);
    assert_eq!(predictions, vec![0, 1]);
}

#[test]
fn test_knn_k4_ignores_far_points() {
    let mut model = KnnClassifier::new(4);
    let (data, dim) = rows_to_flat(&[
        vec![0.0, 0.0],
        vec![0.1, 0.0],
        vec![0.0, 0.1],
        vec![100.0, 100.0], // class 0 but far away
        vec![0.0, 10.0],    // class 1, also far
    ]);
    model.fit(data, dim, vec![0, 0, 0, 0, 1]);

    let (queries, query_dim) = rows_to_flat(&[vec![0.05, 0.05]]);
    let predictions = model.predict(&queries, 1, query_dim);
    assert_eq!(predictions, vec![0]);
}

#[test]
fn test_knn_manhattan() {
    // With Manhattan distance, (0.5, 0) is closer to (0, 0) (dist 0.5)
    // than to (1, 1) (dist |0.5-1| + |0-1| = 1.5).
    let mut model = KnnClassifier::with_metric(1, Metric::Manhattan);
    let (data, dim) = rows_to_flat(&[vec![0.0, 0.0], vec![1.0, 1.0]]);
    model.fit(data, dim, vec![0, 1]);

    let (queries, query_dim) = rows_to_flat(&[vec![0.5, 0.0]]);
    let predictions = model.predict(&queries, 1, query_dim);
    assert_eq!(predictions, vec![0]);
}

#[test]
fn test_knn_cosine() {
    // Cosine distance cares about direction, not magnitude.
    // (1, 0) and (100, 0) point the same way → distance ≈ 0 → class 0.
    let mut model = KnnClassifier::with_metric(1, Metric::Cosine);
    let (data, dim) = rows_to_flat(&[vec![1.0, 0.0], vec![0.0, 1.0]]);
    model.fit(data, dim, vec![0, 1]);

    let (queries, query_dim) = rows_to_flat(&[vec![100.0, 0.0]]);
    let predictions = model.predict(&queries, 1, query_dim);
    assert_eq!(predictions, vec![0]);
}

#[test]
fn test_knn_default_metric_is_euclidean() {
    let model = KnnClassifier::new(1);
    assert_eq!(model.metric, Metric::Euclidean);
}

#[test]
fn test_knn_with_metric_manhattan() {
    let model = KnnClassifier::with_metric(3, Metric::Manhattan);
    assert_eq!(model.metric, Metric::Manhattan);
}
