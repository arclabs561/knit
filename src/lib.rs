//! # clasp
//!
//! Consensus fusion primitives (rank fusion).
//!
//! This crate aims to hold small, well-specified fusion operators used across
//! retrieval, graphs, and clustering stacks.
//!
//! # Research Context
//!
//! - **Reciprocal Rank Fusion (RRF)**: A non-parametric method that is robust to
//!   outliers and requires no score normalization. Mathematically isomorphic to
//!   citation analysis (H-index).
//! - **Copula-based Fusion**: (Hermosillo-Valadez, 2022) theoretically grounds
//!   rank fusion by modeling dependencies between rankers using copulas.
//! - **Arithmetic Average (CombMNZ)**: Li et al. (2021) analysis shows AA fusion
//!   for mode preservation.
//!
//! # Methods
//!
//! - **Unsupervised**: RRF, CombMNZ, Borda.
//! - **Supervised**: Use `cerno-train` for learning-to-rank (LambdaMART, etc).
//!
#![warn(missing_docs)]

use std::collections::HashMap;
use std::hash::Hash;
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
        return ranks
            .iter()
            .filter(|&&r| r > 0)
            .map(|&r| 1.0 / r as f64)
            .sum();
    }
    ranks
        .iter()
        .filter(|&&r| r > 0)
        .map(|&r| 1.0 / (k as f64 + r as f64))
        .sum()
}

/// Fuse multiple ranked lists using RRF, returning per-item scores.
///
/// Returns `(item, score)` pairs sorted by score descending, with a deterministic tie-break
/// by item ascending.
pub fn rrf_fuse<T>(rankings: &[&[T]], k: usize) -> Vec<(T, f64)>
where
    T: Hash + Eq + Clone + Ord,
{
    let mut scores: HashMap<T, f64> = HashMap::new();

    for list in rankings {
        for (rank_0, item) in list.iter().enumerate() {
            let rank = rank_0 + 1;
            let score = if k == 0 {
                1.0 / rank as f64
            } else {
                1.0 / (k as f64 + rank as f64)
            };
            *scores.entry(item.clone()).or_insert(0.0) += score;
        }
    }

    let mut result: Vec<(T, f64)> = scores.into_iter().collect();
    result.sort_by(|(a_item, a_score), (b_item, b_score)| {
        b_score.total_cmp(a_score).then_with(|| a_item.cmp(b_item))
    });
    result
}

/// Fuse multiple ranked lists using RRF.
///
/// Returns a list of items sorted by RRF score (descending).
///
/// # Arguments
///
/// * `rankings` - A slice of ranked lists (e.g., from different retrievers).
/// * `k` - RRF constant (typically 60).
///
/// # Example
///
/// ```rust
/// use clasp::rrf_k_fuse;
///
/// let list1 = vec!["A", "B", "C"];
/// let list2 = vec!["B", "A", "D"];
///
/// // "B" is rank 2 in list1, rank 1 in list2 -> score = 1/(60+2) + 1/(60+1)
/// // "A" is rank 1 in list1, rank 2 in list2 -> score = 1/(60+1) + 1/(60+2)
/// // They should have equal score.
///
/// let fused = rrf_k_fuse(&[&list1, &list2], 60);
/// // Scores for A and B are identical. Tie-breaking is done by value (A < B).
/// assert_eq!(fused[0], "A");
/// assert_eq!(fused[1], "B");
/// ```
pub fn rrf_k_fuse<T>(rankings: &[&[T]], k: usize) -> Vec<T>
where
    T: Hash + Eq + Clone + Ord,
{
    rrf_fuse(rankings, k)
        .into_iter()
        .map(|(item, _score)| item)
        .collect()
}

/// Fuse multiple ranked lists using weighted RRF.
///
/// Each ranking list has an associated weight.
pub fn weighted_rrf_fuse<T>(rankings: &[(&[T], f64)], k: usize) -> Vec<(T, f64)>
where
    T: Hash + Eq + Clone + Ord,
{
    let mut scores: HashMap<T, f64> = HashMap::new();

    for (list, weight) in rankings {
        for (rank_0, item) in list.iter().enumerate() {
            let rank = rank_0 + 1;
            let score = if k == 0 {
                1.0 / rank as f64
            } else {
                1.0 / (k as f64 + rank as f64)
            };
            *scores.entry(item.clone()).or_insert(0.0) += score * weight;
        }
    }

    let mut result: Vec<(T, f64)> = scores.into_iter().collect();
    result.sort_by(|(a_item, a_score), (b_item, b_score)| {
        b_score.total_cmp(a_score).then_with(|| a_item.cmp(b_item))
    });
    result
}

/// Fuse multiple ranked lists using the Condorcet method (Copeland score).
///
/// The score for an item is the number of pairwise wins minus the number of pairwise losses
/// against all other items in the union of all rankings.
pub fn condorcet_fuse<T>(rankings: &[&[T]]) -> Vec<(T, isize)>
where
    T: Hash + Eq + Clone + Ord,
{
    let mut items = std::collections::HashSet::new();
    for list in rankings {
        for item in *list {
            items.insert(item.clone());
        }
    }

    let items_vec: Vec<T> = items.into_iter().collect();
    let n = items_vec.len();
    if n == 0 {
        return vec![];
    }

    // wins[i][j] = how many times item i ranked above item j
    let mut wins = vec![vec![0usize; n]; n];

    for list in rankings {
        // Map item to its rank in this list (missing items = infinity)
        let mut item_to_rank: HashMap<&T, usize> = HashMap::with_capacity(list.len());
        for (rank, item) in list.iter().enumerate() {
            // If a list contains duplicates, treat the earliest position as the rank.
            item_to_rank.entry(item).or_insert(rank);
        }

        for (i, row) in wins.iter_mut().enumerate().take(n) {
            for (j, win_count) in row.iter_mut().enumerate().take(n) {
                if i == j {
                    continue;
                }
                let rank_i = item_to_rank.get(&items_vec[i]);
                let rank_j = item_to_rank.get(&items_vec[j]);

                match (rank_i, rank_j) {
                    (Some(&ri), Some(&rj)) => {
                        if ri < rj {
                            *win_count += 1;
                        }
                    }
                    (Some(_), None) => {
                        *win_count += 1;
                    }
                    (None, Some(_)) => {
                        // j ranked above i
                    }
                    (None, None) => {
                        // both missing, no information
                    }
                }
            }
        }
    }

    let mut copeland_scores = Vec::with_capacity(n);
    for (i, row) in wins.iter().enumerate().take(n) {
        let mut score = 0isize;
        for (j, &win_count) in row.iter().enumerate().take(n) {
            if i == j {
                continue;
            }
            let loss_count = wins[j][i];
            if win_count > loss_count {
                score += 1;
            } else if win_count < loss_count {
                score -= 1;
            }
        }
        copeland_scores.push((items_vec[i].clone(), score));
    }

    copeland_scores.sort_by(|(a_item, a_score), (b_item, b_score)| {
        b_score.cmp(a_score).then_with(|| a_item.cmp(b_item))
    });

    copeland_scores
}

/// Fuse multiple ranked lists using Borda count.
///
/// For each list of length `n`, the top item receives `n` points, next `n-1`, etc.
/// Missing items contribute 0.
///
/// Returns `(item, score)` pairs sorted by score descending, with a deterministic tie-break
/// by item ascending.
pub fn borda_fuse<T>(rankings: &[&[T]]) -> Vec<(T, usize)>
where
    T: Hash + Eq + Clone + Ord,
{
    let mut scores: HashMap<T, usize> = HashMap::new();
    for list in rankings {
        let n = list.len();
        for (rank_0, item) in list.iter().enumerate() {
            let points = n.saturating_sub(rank_0);
            *scores.entry(item.clone()).or_insert(0) += points;
        }
    }

    let mut result: Vec<(T, usize)> = scores.into_iter().collect();
    result.sort_by(|(a_item, a_score), (b_item, b_score)| {
        b_score.cmp(a_score).then_with(|| a_item.cmp(b_item))
    });
    result
}

/// Fuse multiple ranked lists using CombMNZ (CombSUM * num_voters).
///
/// CombMNZ = sum(scores) * count(voters who retrieved the item).
/// This boosts items that appear in many lists (consensus) while respecting their scores.
///
/// Scores are assumed to be normalized (e.g. [0, 1]).
pub fn comb_mnz<T>(rankings: &[&[(T, f64)]]) -> Vec<(T, f64)>
where
    T: Hash + Eq + Clone + Ord,
{
    let mut sums: HashMap<T, f64> = HashMap::new();
    let mut counts: HashMap<T, usize> = HashMap::new();

    for list in rankings {
        for (item, score) in *list {
            *sums.entry(item.clone()).or_insert(0.0) += score;
            *counts.entry(item.clone()).or_insert(0) += 1;
        }
    }

    let mut result: Vec<(T, f64)> = sums
        .into_iter()
        .map(|(item, sum)| {
            let count = counts[&item];
            (item, sum * count as f64)
        })
        .collect();

    result.sort_by(|(a_item, a_score), (b_item, b_score)| {
        b_score.total_cmp(a_score).then_with(|| a_item.cmp(b_item))
    });

    result
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

    #[test]
    fn test_rrf_fuse() {
        let l1 = vec![1, 2, 3];
        let l2 = vec![3, 2, 1];

        // 2 is rank 2 in both. 1 is rank 1 and 3. 3 is rank 3 and 1.
        // All should have equal score = 1/(k+1) + 1/(k+3) vs 1/(k+2)*2
        // Let's use k=0 to make math easy

        let fused = rrf_k_fuse(&[&l1, &l2], 0);
        // Scores:
        // 1: 1/1 + 1/3 = 1.33
        // 2: 1/2 + 1/2 = 1.0
        // 3: 1/3 + 1/1 = 1.33

        assert_eq!(fused.len(), 3);
        // 1 and 3 are tied for first, 2 is last
        assert_eq!(fused[2], 2);
    }

    #[test]
    fn test_rrf_fuse_scores_sorted() {
        let l1 = vec![1, 2, 3];
        let l2 = vec![2, 1, 3];
        let fused = rrf_fuse(&[&l1, &l2], 60);
        assert_eq!(fused.len(), 3);
        for i in 1..fused.len() {
            assert!(fused[i - 1].1 >= fused[i].1);
        }
    }

    #[test]
    fn test_rrf_k_sensitivity_changes_top() {
        let l1 = vec!["B", "C", "A"];
        let l2 = vec!["E", "D", "A"];

        // Small k emphasizes top ranks in single lists.
        let fused_small = rrf_k_fuse(&[&l1, &l2], 0);
        // Large k makes "appears in multiple lists" dominate.
        let fused_large = rrf_k_fuse(&[&l1, &l2], 1_000);

        assert_ne!(fused_small[0], fused_large[0]);
    }

    #[test]
    fn test_borda_fuse_basic() {
        let list1 = vec!["A", "B", "C"];
        let list2 = vec!["B", "A", "D"];
        let fused = borda_fuse(&[&list1, &list2]);

        // "A" gets 3 + 2 = 5, "B" gets 2 + 3 = 5 -> tie broken by item (A < B).
        assert_eq!(fused[0], ("A", 5));
        assert_eq!(fused[1], ("B", 5));
    }

    #[test]
    fn test_comb_mnz() {
        // Items with scores
        let l1 = vec![("A", 0.9), ("B", 0.5)];
        let l2 = vec![("B", 0.6), ("C", 0.8)];

        let fused = comb_mnz(&[&l1, &l2]);

        // A: sum=0.9, count=1 -> mnz = 0.9 * 1 = 0.9
        // B: sum=0.5+0.6=1.1, count=2 -> mnz = 1.1 * 2 = 2.2
        // C: sum=0.8, count=1 -> mnz = 0.8 * 1 = 0.8

        assert_eq!(fused[0].0, "B");
        assert!((fused[0].1 - 2.2).abs() < 1e-6);
        assert_eq!(fused[1].0, "A");
        assert_eq!(fused[2].0, "C");
    }
}
