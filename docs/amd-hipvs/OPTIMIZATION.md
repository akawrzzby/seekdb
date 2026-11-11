# CAGRA build tuning for the AMD hipVS index build

## Change

`tools/hipvs_bridge/seekdb_cuvs_bridge.c` previously used the hipVS defaults for the
NN-descent CAGRA build:

| parameter | before | after |
|---|---|---|
| `intermediate_graph_degree` | 256 | 128 |
| `nn_descent_niter` | 100 | 64 |
| `graph_degree` | 128 | 128 (unchanged) |

The intermediate graph is an internal artefact of the CAGRA build; shrinking it and
stopping NN-descent earlier removes redundant refinement work without changing the
final graph degree, the index semantics or the query path. `ef_construction` /
`ef_search` / `m` from the SQL DDL are untouched.

## Local measurements (AMD Radeon gfx1100, ROCm 7.2.4, hipVS release/rocmds-26.03)

250,000 x 128 FP32, nq=1000, search parameters identical to the baseline
(itopk >= 512, search_width = 4):

| variant | build time (s) | recall@10 | recall@100 |
|---|---|---|---|
| baseline (igd=256, niter=100) | 45 - 55 | 0.9955 | 0.9770 |
| **this change (igd=128, niter=64)** | **19.7 - 35.5** | **0.9935** | **0.9784** |
| igd=192, niter=75 | 27.7 | 0.9955 | 0.9785 |

Build speedup: ~1.8x on a busy GPU, ~2.5x when the GPU is idle. Recall@10 drops by
0.002 (threshold 0.005) and recall@100 slightly improves. Query throughput is
unchanged because the search path and the CAGRA search parameters are identical.

A more aggressive setting (niter=25) was rejected: recall dropped by 0.008, which is
outside the allowed quality threshold.

## Why this is safe

- only GPU build parameters change; the SQL surface, the index type, the metric,
  the vector precision and the query implementation stay exactly as the baseline;
- the exported index keeps the same graph degree, so query cost is unaffected
  (candidate query QPS must stay >= 90% of baseline);
- all results still come from the real AMD hipVS CAGRA build path.
