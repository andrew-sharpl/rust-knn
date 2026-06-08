use knn::KnnClassifier;

fn rows_to_flat(rows: &[Vec<f64>]) -> (Vec<f64>, usize) {
    let dim = rows[0].len();
    let flat: Vec<f64> = rows.iter().flat_map(|row| row.iter().copied()).collect();
    (flat, dim)
}

#[test]
fn test_knn_basic() {
    let mut model = KnnClassifier::new(1);
    let (data, dim) = rows_to_flat(&[
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 10.0],
    ]);
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
    let (data, dim) = rows_to_flat(&[
        vec![0.0],
        vec![1.0],
        vec![10.0],
        vec![11.0],
    ]);
    model.fit(data, dim, vec![0, 0, 1, 1]);

    let (queries, query_dim) = rows_to_flat(&[vec![0.4], vec![10.4]]);
    let predictions = model.predict(&queries, 2, query_dim);
    assert_eq!(predictions, vec![0, 1]);
}

#[test]
fn test_knn_k3_ignores_far_points() {
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
