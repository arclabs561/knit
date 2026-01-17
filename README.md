# clasp

Consensus fusion primitives: Reciprocal Rank Fusion (RRF), Borda count, and Condorcet voting.

Dual-licensed under MIT or Apache-2.0.

```rust
use clasp::rrf_k_fuse;

let list1 = vec!["A", "B", "C"]; // Rank 1, 2, 3
let list2 = vec!["B", "A", "D"]; // Rank 1, 2, 3

// B is rank 2 in list1, rank 1 in list2
// A is rank 1 in list1, rank 2 in list2
// Both get score 1/(k+1) + 1/(k+2). Tie-break by value ("A" < "B").
let fused = rrf_k_fuse(&[&list1, &list2], 60);
assert_eq!(fused[0], "A");
assert_eq!(fused[1], "B");
```
