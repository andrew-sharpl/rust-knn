use criterion::{Criterion, black_box, criterion_group, criterion_main};
use knn::KnnClassifier;

fn bench_predict_euclidean(c: &mut Criterion) {
    let n_train = 5000;
    let dim = 10;
    let data: Vec<f64> = (0..n_train * dim).map(|i| i as f64 * 0.001).collect();
    let labels: Vec<usize> = (0..n_train).map(|i| i % 3).collect();

    let mut model = KnnClassifier::new(3);
    model.fit(data, dim, labels);

    let n_queries = 500;
    let queries: Vec<f64> = (0..n_queries * dim).map(|i| i as f64 * 0.0001).collect();

    c.bench_function("predict_euclidean_5k_train_500_query", |b| {
        b.iter(|| model.predict(black_box(&queries), black_box(n_queries), black_box(dim)))
    });
}

criterion_group!(benches, bench_predict_euclidean);
criterion_main!(benches);
