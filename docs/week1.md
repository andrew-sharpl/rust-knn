# Week 1 — Pure Rust Foundations

## Goal
Build a working brute-force k-nearest neighbors classifier in pure Rust, using only the standard library. The focus is on ownership, borrowing, and getting a correct (if slow) end-to-end pipeline.

## Current Project State
- Library crate (`knn`) with `crate-type = ["rlib", "cdylib"]` (the `cdylib` is for Week 2).
- `src/lib.rs` contains the `KnnClassifier` struct and stubbed methods.
- `src/distance.rs` is a placeholder module for Week 3.
- `tests/test_knn.rs` is an integration-test stub.

---

## Session 1A — Implement `new` and `fit`

### Tasks
1. Add fields to the `KnnClassifier` struct:
   ```rust
   data: Vec<Vec<f64>>,
   labels: Vec<usize>,
   k: usize,
   ```
2. Fill in `new(k: usize) -> Self` to initialize the struct with empty vectors.
3. Fill in `fit(&mut self, data: Vec<Vec<f64>>, labels: Vec<usize>)`.
   - Validate that `data.len() == labels.len()`. For now use `assert_eq!`; later you can return a `Result`.
   - Consider: what should happen if `k > data.len()` after fitting? (Hint: `predict` will panic or behave badly. You may want to assert this in `fit` as well.)
4. Write a small `#[cfg(test)]` block inside `lib.rs` to assert that `fit` stores data correctly.

### Key Concepts to Internalize
- **Ownership:** `data` and `labels` are *moved* into `fit`. The caller gives up ownership; the struct now owns the data.
- **Mutable references (`&mut self`):** `fit` changes the struct’s state, so it takes `&mut self`. If you try to call `fit` through an immutable reference, the borrow checker will stop you. This is the core Rust contract for mutation.
- **Struct initialization:** `Self { data, labels, k }` (field-init shorthand) is idiomatic when variable names match field names.

### Verification
Run `cargo test`. The integration test in `tests/test_knn.rs` should still compile (it’s mostly empty), but your new unit test in `lib.rs` should pass.

---

## Session 1B — Implement `euclidean_distance`

### Tasks
1. Fill in the body of `euclidean_distance(a: &[f64], b: &[f64]) -> f64`.
   - Assert `a.len() == b.len()` (programmer error if not).
   - Compute `sqrt(sum((a_i - b_i)^2))`.
   - Use iterator methods: `a.iter().zip(b.iter()).map(...).sum::<f64>().sqrt()`.
2. Add a unit test in the `#[cfg(test)]` module:
   ```rust
   #[test]
   fn test_euclidean() {
       assert_eq!(euclidean_distance(&[0.0, 0.0], &[3.0, 4.0]), 5.0);
   }
   ```

### Key Concepts to Internalize
- **Slices (`&[f64]`):** The most universal read-only view of a contiguous array. Accepting `&[f64]` instead of `&Vec<f64>` makes the function usable with arrays, `Vec`s, and any other contiguous collection. (TRPL Ch. 4.3.)
- **Iterator adapters:** `zip`, `map`, and `sum` are lazy and often optimize well. Writing the loop manually is also fine if you find it clearer.
- **Floating-point equality:** `assert_eq!` with `f64` is exact. For computed values you may prefer `assert!((actual - expected).abs() < 1e-9)`.

### Verification
Run `cargo test`. Your unit test should pass.

---

## Session 1C — Implement `majority_vote`

### Tasks
1. Fill in `majority_vote(neighbor_labels: &[usize]) -> usize`.
   - Count frequencies of each label.
   - Return the label with the highest count.
   - Decide on a tie-breaking rule (e.g., smallest label wins) and document it in a comment.
2. Choose your counting strategy:
   - **Option A:** `HashMap<usize, usize>` (easy, but allocates and hashes).
   - **Option B:** Sort a copy of the slice and do a run-length count (no hashing, but `O(k log k)`).
   - **Option C:** If you know labels are dense and small, use a `Vec<usize>` as a frequency table.
3. Add unit tests:
   - Simple majority: `[0, 0, 1]` → `0`.
   - Tie: `[0, 1]` → your chosen tie-breaker.

### Key Concepts to Internalize
- **Helper purity:** `majority_vote` takes a borrowed slice and returns a value. It does not allocate anything that outlives the function (unless you choose `HashMap`), and it has no side effects. Pure functions are the easiest to test.
- **API evolvability:** We pulled voting into its own function so that in Week 4 we can swap it for weighted voting without touching `predict`.

### Verification
Run `cargo test`. Both unit tests should pass.

---

## Session 1D — Wire up `predict`

### Tasks
1. Fill in `predict(&self, queries: &[Vec<f64>]) -> Vec<usize>`.
   - For each query point:
     1. Compute Euclidean distance to **every** training point (using `euclidean_distance`).
     2. Collect `(distance, label)` pairs.
     3. Find the `k` smallest distances.
        - **Simplest:** sort the whole `Vec` of pairs by distance and take the first `k`.
        - **Slightly faster:** use `Vec::sort_by` and slice.
        - **Even faster (optional):** use `select_nth_unstable` to avoid a full sort.
     4. Extract the labels of those `k` neighbors.
     5. Call `majority_vote` on those labels.
     6. Push the result into the output `Vec`.
2. Fill in `tests/test_knn.rs` with a real integration test.
   - Create a small 2D dataset where the answer is obvious.
   - Example:
     ```rust
     let train = vec![
         vec![0.0, 0.0],
         vec![1.0, 0.0],
         vec![0.0, 10.0],
     ];
     let labels = vec![0, 0, 1];
     let mut model = KnnClassifier::new(1);
     model.fit(train, labels);
     let pred = model.predict(&[vec![0.1, 0.0]]);
     assert_eq!(pred, vec![0]);
     ```
3. Add edge-case tests:
   - Query exactly equal to a training point.
   - `k` equal to the full training set (all points vote).

### Key Concepts to Internalize
- **Borrowing vs. ownership in `predict`:**
  - `&self` means we promise not to mutate the classifier. This is why we can call `predict` multiple times without needing `mut`.
  - `queries: &[Vec<f64>]` means we borrow the queries. The caller keeps ownership and can reuse the query vector afterwards.
  - We return an *owned* `Vec<usize>` because the predictions are new data that the caller must own.
- **Allocation inside loops:** For each query, you will likely allocate a `Vec<(f64, usize)>` of distances. That’s fine for Week 1. In Week 4 we’ll look at reusing buffers (e.g., a single `Vec` cleared each iteration) to reduce allocator pressure.
- **Immutability discipline:** Notice that none of the public methods except `fit` need `&mut self`. This is a common Rust pattern: mutation is explicit and localized.

### Verification
Run `cargo test`. Both unit tests and integration tests should pass.

---

## Common Borrow-Checker Pitfalls to Watch For

1. **Moving out of a Vec while iterating:** If you try to `drain` or `pop` inside a loop without thinking about ownership, the compiler will complain. In `predict`, you are only *reading* `self.data` and `self.labels`, so `&self` is sufficient.
2. **Returning a reference to local data:** You cannot return `&usize` pointing into a local `HashMap` inside `majority_vote`; the map is dropped when the function returns. This is why `majority_vote` returns `usize` by value.
3. **Holding a reference across mutation:** If you ever try to sort `self.data` while holding a reference to one of its rows, the borrow checker will reject it. In our current design, `predict` does not mutate `self`, so this is not an issue.

---

## What You Should Have at the End of Week 1
- A compiling `KnnClassifier` with `new`, `fit`, and `predict`.
- `euclidean_distance` and `majority_vote` working and unit-tested.
- At least one integration test in `tests/test_knn.rs` passing.
- Comfort with `cargo test`, `cargo check`, and basic ownership/borrowing.

## Design Decisions to Keep in Mind for Future Weeks
- `Vec<Vec<f64>>` is a teaching aid. In Week 2/3 we will migrate to a layout compatible with NumPy (flat buffer + dimensions).
- Owned data avoids lifetimes for now. Once you are comfortable, try refactoring `KnnClassifier` to borrow training data (`&'a [Vec<f64>]`) as an exercise.
- `majority_vote` is a plain function today; weighted voting will replace it in Week 4.
- The `distance.rs` module is empty. In Week 3 we will add an enum or trait for distance metrics.
