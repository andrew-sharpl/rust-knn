use criterion::{Criterion, black_box, criterion_group, criterion_main};
use knn::{KnnClassifier, algorithm::Algorithm, distance::Metric};

/// Build a deterministic synthetic dataset.
fn make_data(n_train: usize, dim: usize) -> (Vec<f64>, Vec<usize>) {
    let data: Vec<f64> = (0..n_train * dim).map(|i| i as f64 * 0.001).collect();
    let labels: Vec<usize> = (0..n_train).map(|i| i % 3).collect();
    (data, labels)
}

fn make_queries(n_queries: usize, dim: usize) -> (Vec<f64>, usize) {
    let queries: Vec<f64> = (0..n_queries * dim).map(|i| i as f64 * 0.0001).collect();
    (queries, n_queries)
}

fn bench_brute_vs_kdtree(c: &mut Criterion, n_train: usize, dim: usize, label: &str) {
    let (data, labels) = make_data(n_train, dim);
    let (queries, n_queries) = make_queries(500, dim);

    // Brute-force model
    let mut brute = KnnClassifier::new(3);
    brute.fit(data.clone(), dim, labels.clone());

    c.bench_function(&format!("brute_{label}"), |b| {
        b.iter(|| brute.predict(black_box(&queries), black_box(n_queries), black_box(dim)))
    });

    // KD-tree model
    let mut kdtree = KnnClassifier::new(3);
    kdtree.with_algorithm(Algorithm::KdTree);
    kdtree.fit(data.clone(), dim, labels.clone());

    c.bench_function(&format!("kdtree_{label}"), |b| {
        b.iter(|| kdtree.predict(black_box(&queries), black_box(n_queries), black_box(dim)))
    });
}

fn bench_euclidean_5k_dim10(c: &mut Criterion) {
    bench_brute_vs_kdtree(c, 5_000, 10, "euclidean_5k_dim10");
}

fn bench_euclidean_50k_dim10(c: &mut Criterion) {
    bench_brute_vs_kdtree(c, 50_000, 10, "euclidean_50k_dim10");
}

fn bench_euclidean_50k_dim3(c: &mut Criterion) {
    bench_brute_vs_kdtree(c, 50_000, 3, "euclidean_50k_dim3");
}

fn bench_euclidean_50k_dim50(c: &mut Criterion) {
    bench_brute_vs_kdtree(c, 50_000, 50, "euclidean_50k_dim50");
}

fn bench_manhattan_kdtree(c: &mut Criterion) {
    let (data, labels) = make_data(50_000, 10);
    let (queries, n_queries) = make_queries(500, 10);

    let mut brute = KnnClassifier::new(3);
    brute.with_metric(Metric::Manhattan);
    brute.fit(data.clone(), 10, labels.clone());

    c.bench_function("brute_manhattan_50k_dim10", |b| {
        b.iter(|| brute.predict(black_box(&queries), black_box(n_queries), black_box(10)))
    });

    let mut kdtree = KnnClassifier::new(3);
    kdtree.with_metric(Metric::Manhattan);
    kdtree.with_algorithm(Algorithm::KdTree);
    kdtree.fit(data, 10, labels);

    c.bench_function("kdtree_manhattan_50k_dim10", |b| {
        b.iter(|| kdtree.predict(black_box(&queries), black_box(n_queries), black_box(10)))
    });
}

criterion_group!(
    benches,
    bench_euclidean_5k_dim10,
    bench_euclidean_50k_dim10,
    bench_euclidean_50k_dim3,
    bench_euclidean_50k_dim50,
    bench_manhattan_kdtree,
);
criterion_main!(benches);
