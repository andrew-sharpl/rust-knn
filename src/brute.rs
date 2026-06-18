use crate::distance::Metric;

/// Brute-force k-nearest neighbors search.
///
/// Computes the distance from `query` to every point in `data`, then returns
/// the `k` closest as `(distance, point_index)` pairs, sorted by distance.
///
/// `data` is the flat row-major buffer. `dim` is the feature dimension.
pub fn brute_force_search(
    query: &[f64],
    data: &[f64],
    dim: usize,
    k: usize,
    metric: &Metric,
) -> Vec<(f64, usize)> {
    let n_points = data.len() / dim;
    let mut distances: Vec<(f64, usize)> = Vec::with_capacity(n_points);

    for i in 0..n_points {
        let point = &data[i * dim..(i + 1) * dim];
        distances.push((metric.distance(point, query), i));
    }

    distances.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));

    let mut result: Vec<(f64, usize)> = distances[..k].to_vec();
    result.sort_by(|a, b| a.0.total_cmp(&b.0));
    result
}
