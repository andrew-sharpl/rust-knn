//! Distance-weighted voting for k-nearest neighbors.
//!
//! Each variant defines how to convert a neighbor's distance into a vote weight.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Weighting {
    /// Every neighbor counts equally (weight = 1).
    Uniform,
    /// weight = 1/d. Returns `f64::INFINITY` when d = 0; the voter short-circuits.
    InverseDistance,
    /// weight = 1/(1+d). Always finite, avoids division by zero.
    SmoothedInverse,
    /// weight = exp(-d). Smooth exponential falloff, always finite.
    Gaussian,
}

impl Weighting {
    /// Convert a distance to a weight
    ///
    /// For [`Weighting::InverseDistance`], returns `f64::INFINITY` when `distance == 0.0`.
    pub fn weight(&self, distance: f64) -> f64 {
        match self {
            Weighting::Uniform => 1.0,
            Weighting::InverseDistance => 1.0 / distance,
            Weighting::SmoothedInverse => 1.0 / (1.0 + distance), 
            Weighting::Gaussian => (-distance).exp(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_always_one() {
        assert_eq!(Weighting::Uniform.weight(0.0), 1.0);
        assert_eq!(Weighting::Uniform.weight(100.0), 1.0);
        assert_eq!(Weighting::Uniform.weight(0.23), 1.0);
    }

    #[test]
    fn inverse_distance_nonzero() {
        assert_eq!(Weighting::InverseDistance.weight(0.5), 2.0);
    }

    #[test]
    fn inverse_distance_one() {
        assert_eq!(Weighting::InverseDistance.weight(1.0), 1.0);
    }

    #[test]
    fn inverse_distance_zero_to_inf() {
        assert_eq!(Weighting::InverseDistance.weight(0.0), f64::INFINITY);
    }

    #[test]
    fn smoothed_inverse_at_d0() {
        assert_eq!(Weighting::SmoothedInverse.weight(0.0), 1.0);
    }

    #[test]
    fn smoothed_inverse_at_d1() {
        assert_eq!(Weighting::SmoothedInverse.weight(1.0), 0.5);
    }

    #[test]
    fn smoothed_inverse_at_d9() {
        assert_eq!(Weighting::SmoothedInverse.weight(9.0), 0.1);
    }

    #[test]
    fn gaussian_at_d0() {
        assert_eq!(Weighting::Gaussian.weight(0.0), 1.0);
    }

    #[test]
    fn gaussian_at_d1() {
        let w = Weighting::Gaussian.weight(1.0);
        assert!((w - 0.36787944117144233).abs() < 1e-12);
    }

    #[test]
    fn gaussian_decays() {
        let w1 = Weighting::Gaussian.weight(1.0);
        let w5 = Weighting::Gaussian.weight(5.0);
        assert!(w5 < w1);
    }
}
