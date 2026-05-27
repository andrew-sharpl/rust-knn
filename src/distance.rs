//! Distance metrics for k-nearest neighbors.
//!
//! Currently empty — this is a placeholder for Week 3, when we will add
//! Manhattan (L1), Cosine, and possibly Minkowski distances.
//!
//! **Design decision preview:**
//! We *could* represent a distance function as a trait:
//!
//! ```ignore
//! pub trait Distance {
//!     fn compute(a: &[f64], b: &[f64]) -> f64;
//! }
//! ```
//!
//! But trait objects (`Box<dyn Distance>`) or generics (`KnnClassifier<D: Distance>`)
//! add complexity in Week 1. Instead, we’ll likely start with an enum:
//!
//! ```ignore
//! pub enum Metric {
//!     Euclidean,
//!     Manhattan,
//!     Cosine,
//! }
//! ```
//!
//! An enum is simpler to serialize across the Rust/Python boundary (PyO3 loves enums),
//! and easier to pattern-match inside `predict`. Once the code works, we can refactor
//! to a trait if we want zero-cost abstraction and monomorphization.
//! (TRPL Ch. 10 — Generic Types, Traits, and Lifetimes.)
