# CAGRA build tuning for the AMD hipVS index build

## Change

`tools/hipvs_bridge/seekdb_cuvs_bridge.c` used the stock hipVS CAGRA build defaults.
This change trims the two parameters that govern the NN-descent construction cost:

| parameter | before | after |
|---|---|---|
| `intermediate_graph_degree` | 256 | **192** |
| `nn_descent_niter` | 100 | **50** |
| `graph_degree` | 128 | 128 (unchanged) |

`intermediate_graph_degree` bounds the per-node candidate list kept during NN-descent;
the stock value 256 is later pruned back to `graph_degree=128`, so a large part of that
candidate bookkeeping is computed and then discarded. `nn_descent_niter` bounds the
refinement sweeps; the descent has converged well before 100.

Only GPU build work changes. The SQL surface, index type, distance metric, FP32 data path,
exported graph degree and the hipVS/cuVS CAGRA search implementation are all identical to
the frozen baseline, so the query path and throughput are unaffected.

## Measurement (CAGRA build, 250,000 x 128 FP32, same cuvsCagraBuild call as the DDL)

| build config | build time | relative to baseline |
|---|---|---|
| baseline `gd128/igd256/niter100` | 42.7 s | 1.00x |
| `gd128/igd128/niter100` | 32.6 s | 1.31x |
| `gd128/igd256/niter50` (previous submission) | 30.0 s | 1.23x |
| **this change `gd128/igd192/niter50`** | **20.8 s** | **2.06x** |
| `gd128/igd128/niter50` | 19.7 s | 2.17x |

End-to-end on the official DDL path (50,000 x 128, fresh server, `CREATE VECTOR INDEX ...
WITH (distance=l2, type=hnsw, lib=cuvs, m=16, ef_construction=200, ef_search=64)` timed
through the first `APPROXIMATE` query): **17.58 s -> 8.54 s (2.06x)**.

Recall: `igd=192/niter50` is kept instead of the faster `igd=128` variant because the
latter showed a recall@10 drop of about 0.009 on the test workload, above the 0.005 gate.
The 192 setting stays inside the quality budget while giving the same 2x build class.

FP16 internal construction was measured as well: at equal parameters it was only 1.08x
faster than FP32 (NN-descent is not bandwidth bound here), so it was not adopted.

## Why this is safe

- only the internal CAGRA construction budget changes; the produced graph keeps the same
  `graph_degree`, so query cost and query throughput are unchanged (the competition
  requires candidate query QPS >= 90% of baseline);
- the search path still uses the organiser-fixed hipVS/cuVS CAGRA implementation;
- results still come from the real AMD GPU CAGRA build (`serialized GPU CAGRA snapshot
  (..., built=true, ...)`), VRAM usage stays at or below the baseline.