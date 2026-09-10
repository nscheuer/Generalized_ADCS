# Task 7 -- LP/QP allocation solve timings (sec:V-B)

Machine: macOS-26.4-arm64-arm-64bit, Python single process/core. N=2000 solves, 3MTQ+1RW geometry, random request directions, fixed ~30 uT body field.

| Allocator | median | mean | p95 |
|---|---|---|---|
| LP (HiGHS `linprog`) | 530 us | 554 us | 689 us |
| QP (`lsq_linear` BVLS) | 62 us | 85 us | 129 us |

Paper sec:V-B currently cites LP ~350 us / QP ~40 us (single core). Measured here: LP 530 us / QP 62 us median. These are **desktop** numbers (this machine); the LP>QP ordering and ~10x ratio are the load-bearing qualitative claim. Embedded/flight-class timings would be larger (slower clock, no vectorized BLAS) but the relative ordering holds; the single-core desktop caveat in sec:V-B should stand as written unless flight-processor numbers are measured directly.
