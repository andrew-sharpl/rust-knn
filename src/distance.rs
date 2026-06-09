//! Distance metrics for k-nearest neighbors.
//!
//! Supports Euclidean (L2), Manhattan (L1), and Cosine distance via the
//! [`Metric`] enum.

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
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum()
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

    assert!(
        norm_a > 0.0,
        "cosine distance is undefined for the zero vector (a)"
    );
    assert!(
        norm_b > 0.0,
        "cosine distance is undefined for the zero vector (b)"
    );

    1.0 - dot_product / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    mod euclidean {
        use super::*;

        #[test]
        fn identical() {
            assert_eq!(Metric::Euclidean.distance(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
        }

        #[test]
        fn three_four_five() {
            assert_eq!(Metric::Euclidean.distance(&[0.0, 0.0], &[3.0, 4.0]), 5.0);
        }

        #[test]
        fn one_dimensional() {
            assert_eq!(Metric::Euclidean.distance(&[0.0], &[5.0]), 5.0);
            assert_eq!(Metric::Euclidean.distance(&[3.0], &[8.0]), 5.0);
        }

        #[test]
        fn negative_coords() {
            assert_eq!(Metric::Euclidean.distance(&[-3.0, -4.0], &[0.0, 0.0]), 5.0);
            assert_eq!(Metric::Euclidean.distance(&[-1.0, 2.0], &[2.0, 6.0]), 5.0);
        }

        #[test]
        fn three_d() {
            let expected = (14.0_f64).sqrt();
            let actual = Metric::Euclidean.distance(&[0.0, 0.0, 0.0], &[1.0, 2.0, 3.0]);
            assert!((actual - expected).abs() < 1e-12, "expected {}, got {}", expected, actual);
        }

        #[test]
        #[should_panic(expected = "same dimension")]
        fn mismatched_dimensions() {
            Metric::Euclidean.distance(&[1.0, 0.0], &[0.0]);
        }

        #[test]
        #[should_panic]
        fn empty_vector() {
            Metric::Euclidean.distance(&[], &[1.0, 2.0]);
        }
    }

    mod manhattan {
        use super::*;

        #[test]
        fn axes() {
            assert_eq!(Metric::Manhattan.distance(&[0.0, 0.0], &[3.0, 4.0]), 7.0);
        }

        #[test]
        fn negative() {
            assert_eq!(Metric::Manhattan.distance(&[-1.0], &[2.0]), 3.0);
        }

        #[test]
        fn identical() {
            assert_eq!(Metric::Manhattan.distance(&[5.0, 10.0], &[5.0, 10.0]), 0.0);
        }

        #[test]
        fn one_dimensional() {
            assert_eq!(Metric::Manhattan.distance(&[0.0], &[5.0]), 5.0);
        }

        #[test]
        fn three_d() {
            assert_eq!(Metric::Manhattan.distance(&[1.0, 2.0, 3.0], &[4.0, 6.0, 2.0]), 8.0);
        }

        #[test]
        #[should_panic(expected = "same dimension")]
        fn mismatched_dimensions() {
            Metric::Manhattan.distance(&[1.0, 0.0], &[0.0]);
        }
    }

    mod cosine {
        use super::*;

        #[test]
        fn orthogonal() {
            let d = Metric::Cosine.distance(&[1.0, 0.0], &[0.0, 1.0]);
            assert!((d - 1.0).abs() < 1e-12);
        }

        #[test]
        fn same_direction() {
            let d = Metric::Cosine.distance(&[1.0, 2.0], &[2.0, 4.0]);
            assert!(d.abs() < 1e-12);
        }

        #[test]
        fn opposite() {
            let d = Metric::Cosine.distance(&[1.0, 0.0], &[-1.0, 0.0]);
            assert!((d - 2.0).abs() < 1e-12);
        }

        #[test]
        fn identical() {
            let d = Metric::Cosine.distance(&[3.0, 4.0], &[3.0, 4.0]);
            assert!(d.abs() < 1e-12);
        }

        #[test]
        #[should_panic(expected = "undefined for the zero vector")]
        fn zero_vector_a_panics() {
            Metric::Cosine.distance(&[0.0, 0.0], &[1.0, 1.0]);
        }

        #[test]
        #[should_panic(expected = "undefined for the zero vector")]
        fn zero_vector_b_panics() {
            Metric::Cosine.distance(&[1.0, 1.0], &[0.0, 0.0]);
        }

        #[test]
        #[should_panic(expected = "same dimension")]
        fn mismatched_dimensions() {
            Metric::Cosine.distance(&[1.0, 0.0], &[0.0]);
        }
    }
}
