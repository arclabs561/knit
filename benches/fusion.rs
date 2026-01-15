use criterion::{black_box, criterion_group, criterion_main, Criterion};
use knit::{rrf_k_fuse, rrf_score};
use std::collections::HashSet;

fn bench_rrf(c: &mut Criterion) {
    let mut group = c.benchmark_group("rrf");

    let k = 60;
    let list_len = 100;
    let ranks: Vec<usize> = (1..=list_len).collect();

    group.bench_function("rrf_score_100_items", |b| {
        b.iter(|| rrf_score(black_box(&ranks), black_box(k)))
    });

    let list1: Vec<usize> = (0..1000).collect();
    let list2: Vec<usize> = (0..1000).rev().collect(); // Reverse order

    group.bench_function("rrf_fuse_2_lists_1000", |b| {
        b.iter(|| rrf_k_fuse(black_box(&[&list1, &list2]), black_box(k)))
    });

    group.finish();
}

fn bench_rrf_k_sensitivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("rrf_k_sensitivity");

    let l1: Vec<usize> = (0..1000).collect();
    let l2: Vec<usize> = (500..1500).rev().collect();

    let ks = [0usize, 60, 1_000];
    for &k in &ks {
        group.bench_function(format!("top10_overlap_k{}", k), |b| {
            b.iter(|| {
                let fused = rrf_k_fuse(black_box(&[&l1, &l2]), black_box(k));
                let top: HashSet<usize> = fused.iter().take(10).cloned().collect();
                black_box(top.len())
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_rrf, bench_rrf_k_sensitivity);
criterion_main!(benches);
