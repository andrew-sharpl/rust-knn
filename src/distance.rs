//! Distance metrics for k-nearest neighbors.
//!
//! This module will hold Euclidean, Manhattan, and Cosine distance functions.
//! Currently only Euclidean is implemented; additional metrics are planned.

use core::f64;
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Metric {
    Euclidean,
    Manhattan,
    Cosine,
}

impl Metric {
    /// Compute the distance between two points using this metric.
    ///
    /// # Panics
    ///
    /// Panics if `a` and `b` have different lengths.
    ///
    /// For `Metric::Cosine`, panics if either vector is the zero vector
    /// (cosine distance is undefined). 
    pub fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "points must have the same dimension");
        
        match self {
            Metric::Euclidean => euclidean(a, b),
            Metric::Manhattan => manhattan(a, b),
            Metric::Cosine => cosine(a, b),
        }
    }
}

/// √Σ(xᵢ - yᵢ)² — the straight-line distance.
fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Σ|xᵢ - yᵢ| — the "city block" distance.
fn manhattan(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .sum()

}

/// 1 - (A·B)/(‖A‖×‖B‖) — measures the angle between vectors.
///
/// Returns `0.0` for identical directions, `1.0` for orthogonal,
/// `2.0` for opposite directions.
///
/// # Panics
///
/// Panics if either vector is the zero vector (undefined cosine).
fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let dot_product: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();

    assert!(norm_a > 0.0, "cosine distance is undefined for the zero vector (a)");
    assert!(norm_b > 0.0, "cosine distance is undefined for the zero vector (b)");

    1.0 - dot_product / (norm_a * norm_b)
}

