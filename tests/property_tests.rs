use knit::{condorcet_fuse, rrf_k_fuse, rrf_score, weighted_rrf_fuse};
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_rrf_monotonic(rank in 1usize..1000) {
        // Lower rank (better) should give higher score
        let s1 = rrf_score(&[rank], 60);
        let s2 = rrf_score(&[rank + 1], 60);
        prop_assert!(s1 > s2);
    }

    #[test]
    fn prop_rrf_fuse_subset(
        list1 in prop::collection::vec(0..100u32, 1..10),
        list2 in prop::collection::vec(0..100u32, 1..10)
    ) {
        let fused = rrf_k_fuse(&[&list1, &list2], 60);

        // Fused list should contain union of elements
        for x in &list1 {
            prop_assert!(fused.contains(x));
        }
        for x in &list2 {
            prop_assert!(fused.contains(x));
        }
    }

    #[test]
    fn prop_weighted_rrf_monotonic_in_weights(
        w1 in 0.1f64..10.0,
        w2 in 0.1f64..10.0
    ) {
        let list = vec![1];
        let res1 = weighted_rrf_fuse(&[(&list, w1)], 60);
        let res2 = weighted_rrf_fuse(&[(&list, w1 + w2)], 60);

        // Score should increase with weight
        prop_assert!(res2[0].1 > res1[0].1);
    }

    #[test]
    fn prop_rrf_permutation_invariance(
        list1 in prop::collection::vec(0..50u32, 1..10),
        list2 in prop::collection::vec(0..50u32, 1..10)
    ) {
        let fused1 = rrf_k_fuse(&[&list1, &list2], 60);
        let fused2 = rrf_k_fuse(&[&list2, &list1], 60);

        // Order of input lists should not matter for RRF
        prop_assert_eq!(fused1, fused2);
    }

    #[test]
    fn prop_condorcet_permutation_invariance(
        list1 in prop::collection::vec(0..50u32, 1..10),
        list2 in prop::collection::vec(0..50u32, 1..10)
    ) {
        let fused1 = condorcet_fuse(&[&list1, &list2]);
        let fused2 = condorcet_fuse(&[&list2, &list1]);

        // Order of input lists should not matter for Condorcet
        prop_assert_eq!(fused1, fused2);
    }

    #[test]
    fn prop_condorcet_monotonic_rank(
        item in 0..10u32,
        list1 in prop::collection::vec(0..10u32, 5..10)
    ) {
        // If we move an item up in one list, its score should not decrease
        let mut list2 = list1.clone();
        if let Some(pos) = list2.iter().position(|&x| x == item) {
            if pos > 0 {
                list2.swap(pos, pos - 1);
            }
        } else {
            list2.push(item);
        }

        let score1 = condorcet_fuse(&[&list1])
            .into_iter()
            .find(|(it, _)| *it == item)
            .map(|(_, s)| s)
            .unwrap_or(-100);

        let score2 = condorcet_fuse(&[&list2])
            .into_iter()
            .find(|(it, _)| *it == item)
            .map(|(_, s)| s)
            .unwrap_or(-100);

        prop_assert!(score2 >= score1);
    }
}
