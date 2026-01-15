//! # knit
//!
//! Consensus fusion primitives (rank fusion).
//!
//! This crate aims to hold small, well-specified fusion operators used across
//! retrieval, graphs, and clustering stacks.
#![warn(missing_docs)]

/// Reciprocal Rank Fusion (RRF).
///
/// Given multiple ranked lists, RRF assigns each item a score:
/// \[
/// \text{score}(d) = \sum_{l \in L} \frac{1}{k + \text{rank}_l(d)}
/// \]
///
/// where ranks are 1-indexed, and items not present in a list contribute 0.
pub fn rrf_score(ranks: &[usize], k: usize) -> f64 {
    if k == 0 {
        // k=0 is allowed mathematically; keep it explicit to avoid division surprises.
        return ranks.iter().filter(|&&r| r > 0).map(|&r| 1.0 / r as f64).sum();
    }
    ranks
        .iter()
        .filter(|&&r| r > 0)
        .map(|&r| 1.0 / (k as f64 + r as f64))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rrf_basic() {
        // ranks are 1-indexed; 0 means "missing"
        let s = rrf_score(&[1, 3, 0], 60);
        assert!(s > 0.0);
    }
}
