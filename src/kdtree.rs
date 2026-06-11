// kdtree.rs

use crate::distance::Metric;

/// A single node in the KD-tree.
///
/// Stores the index (into the flat data buffer) of the training point
/// at this node, plus which axis we split on for the two subtrees.
struct KdNode {
    /// Index of this node's point in the flat `Vec<f64>` data buffer.
    point_index: usize,
    /// The axis (0..dim-1) this node splits on.
    split_dim: usize,
    /// Left subtree: points whose coordinate on `split_dim` is ≤ this node's.
    left: Option<Box<KdNode>>,
    /// Right subtree: points whose coordinate on `split_dim` is > this node's.
    right: Option<Box<KdNode>>,
}

/// A KD-tree for fast nearest-neighbor search.
///
/// Build from flat data buffer to mirror `numpy.ndarray`. Nodes store indices into that buffer.
pub struct KdTree {
    /// Root node of KD-tree.
    root: Option<Box<KdNode>>,
    /// Dimension of KD-tree.
    dim: usize,
}

/// Build a KD-tree from training data.
///
/// `data` is the flat row-major buffer. `dim` is the feature dimension.
pub fn build_kdtree(data: &[f64], dim: usize) -> KdTree {
    assert!(dim > 0, "dim must be positive");
    assert_eq!(data.len() % dim, 0, "data length must be divisible by dim");

    let n_points = data.len() / dim;
    let mut indices: Vec<usize> = (0..n_points).collect();
    let root = build_node(data, &mut indices, dim, 0);
    KdTree { root, dim }
}

/// Recursively builds a KD-tree node.
fn build_node(
    data: &[f64],
    indices: &mut [usize],
    dim: usize,
    depth: usize,
) -> Option<Box<KdNode>> {
    if indices.is_empty() {
        return None;
    }

    let axis = depth % dim;

    indices.sort_by(|&a, &b| data[a * dim + axis].total_cmp(&data[b * dim + axis]));

    let median = indices.len() / 2;
    let point_index = indices[median];

    let left = build_node(data, &mut indices[..median], dim, depth + 1);
    let right = build_node(data, &mut indices[median + 1..], dim, depth + 1);

    Some(Box::new(KdNode {
        point_index,
        split_dim: axis,
        left,
        right,
    }))
}

impl KdTree {
    /// Find the k nearest neighbors of `query`.
    ///
    /// Returns a vector of `(distance, point_index)` pairs, sorted by distance.
    pub fn k_nearest(
        &self,
        query: &[f64],
        data: &[f64],
        k: usize,
        metric: &Metric,
    ) -> Vec<(f64, usize)> {
        let mut best: Vec<(f64, usize)> = Vec::with_capacity(k);
        self.search_recursive(self.root.as_ref(), query, data, k, metric, &mut best);
        best
    }

    fn search_recursive(
        &self,
        node: Option<&Box<KdNode>>,
        query: &[f64],
        data: &[f64],
        k: usize,
        metric: &Metric,
        best: &mut Vec<(f64, usize)>,
    ) {
        let Some(node) = node else {
            return;
        };

        let point = &data[node.point_index * self.dim..(node.point_index + 1) * self.dim];
        let distance = metric.distance(point, query);

        if best.len() < k {
            best.push((distance, node.point_index));
            best.sort_by(|a, b| a.0.total_cmp(&b.0));
        } else if distance < best.last().expect("best should have at least k elements").0 {
            best.pop();
            best.push((distance, node.point_index));
            best.sort_by(|a, b| a.0.total_cmp(&b.0));
        }

        let query_coord = query[node.split_dim];
        let node_coord = data[node.point_index * self.dim + node.split_dim];

        let (near, far) = if query_coord <= node_coord {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };

        self.search_recursive(near.as_ref(), query, data, k, metric, best);

        let distance_to_plane = (query_coord - node_coord).abs();

        let should_search_far = if best.len() < k {
            true
        } else {
            distance_to_plane < best.last().expect("best should have at least k elements").0
        };

        if should_search_far {
            self.search_recursive(far.as_ref(), query, data, k, metric, best);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_1nn() {
        // Three 2D points: (0,0), (1,0), (0,10)
        let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 10.0];
        let dim = 2;
        let tree = build_kdtree(&data, dim);

        // Query: (0.1, 0.0) — closest to point 0 (0,0), dist ≈ 0.1
        let query = vec![0.1, 0.0];
        let neighbors = tree.k_nearest(&query, &data, 1, &Metric::Euclidean);

        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].1, 0);
        assert!((neighbors[0].0 - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_k3() {
        // Four 1D points: 0, 1, 10, 11
        let data = vec![0.0, 1.0, 10.0, 11.0];
        let dim = 1;
        let tree = build_kdtree(&data, dim);

        // Query: 0.4 — closest are 0 (dist 0.4), 1 (dist 0.6), 10 (dist 9.6)
        let query = vec![0.4];
        let neighbors = tree.k_nearest(&query, &data, 3, &Metric::Euclidean);

        assert_eq!(neighbors.len(), 3);
        assert_eq!(neighbors[0].1, 0);
        assert!((neighbors[0].0 - 0.4).abs() < 1e-10);
        assert_eq!(neighbors[1].1, 1);
        assert!((neighbors[1].0 - 0.6).abs() < 1e-10);
        assert_eq!(neighbors[2].1, 2);
        assert!((neighbors[2].0 - 9.6).abs() < 1e-10);
    }

    #[test]
    fn test_manhattan() {
        // With Manhattan, (0.5, 0) is closer to (0,0) (dist 0.5) than to (1,1) (dist 1.5)
        let data = vec![0.0, 0.0, 1.0, 1.0];
        let dim = 2;
        let tree = build_kdtree(&data, dim);

        let query = vec![0.5, 0.0];
        let neighbors = tree.k_nearest(&query, &data, 1, &Metric::Manhattan);

        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].1, 0);
        assert!((neighbors[0].0 - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_single_point() {
        let data = vec![5.0, 3.0];
        let dim = 2;
        let tree = build_kdtree(&data, dim);

        let query = vec![5.0, 3.0];
        let neighbors = tree.k_nearest(&query, &data, 1, &Metric::Euclidean);

        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].1, 0);
        assert_eq!(neighbors[0].0, 0.0);
    }

    #[test]
    fn test_results_sorted_by_distance() {
        let data = vec![0.0, 10.0, 5.0, 1.0, 2.0, 3.0, 8.0, 9.0];
        let dim = 2;
        let tree = build_kdtree(&data, dim);

        let query = vec![1.0, 2.0];
        let neighbors = tree.k_nearest(&query, &data, 3, &Metric::Euclidean);

        // Verify sorted ascending by distance
        for i in 1..neighbors.len() {
            assert!(neighbors[i - 1].0 <= neighbors[i].0);
        }
    }

    #[test]
    fn test_all_points_returned_when_k_exceeds_points() {
        // 2 points, ask for 3 neighbors — should return all 2
        let data = vec![0.0, 1.0];
        let dim = 1;
        let tree = build_kdtree(&data, dim);

        let query = vec![0.0];
        let neighbors = tree.k_nearest(&query, &data, 3, &Metric::Euclidean);

        assert_eq!(neighbors.len(), 2);
    }
}
