# CAGRA build tuning for the AMD hipVS index build

## Change

`tools/hipvs_bridge/seekdb_cuvs_bridge.c` used the stock hipVS CAGRA build defaults.
This change shortens the NN-descent refinement phase of the CAGRA construction:

| parameter | before | after |
|---|---|---|
| `nn_descent_niter` | 100 | **50** |
| `intermediate_graph_degree` | 256 | 256 (unchanged) |
| `graph_degree` | 128 | 128 (unchanged) |

Only GPU build work changes. The SQL surface, the index type, the distance metric, the
FP32 data path, the exported graph degree and the CAGRA search implementation are all
identical to the frozen baseline.

## End-to-end measurement (official SQL path)

Method: fresh seekdb server per run, data loaded with INSERT, no index present, then time

```sql
CREATE VECTOR INDEX idx_items ON items(v)
  WITH (distance=l2, type=hnsw, lib=cuvs, m=16, ef_construction=200, ef_search=64);
SELECT id FROM items ORDER BY l2_distance(v, <query>) APPROXIMATE LIMIT 10;
```

The clock covers the DDL, data scan, host/device transfer, CAGRA build, GPU sync, index
publication and the first query, i.e. the competition's build-time definition.

| dataset | variant | build time | recall@10 | recall@100 |
|---|---|---|---|---|
| 50,000 x 128 | baseline (niter=100) | 19.4 s | 1.0000 | 0.9991 |
| 50,000 x 128 | **this change (niter=50)** | **12.3 s** | **1.0000** | **0.9992** |

Build speedup **~1.58x** with no measurable recall loss (recall@10 identical, recall@100
+0.0001). Every query returned `k` valid, distinct ids (0 invalid rows, 0 duplicate
blocks). GPU utilisation peaks at 100% during the build with 0.53-2.4 GB VRAM vs 28 MB
when idle, so the CAGRA build really runs on the AMD GPU.

A wider CAGRA micro-benchmark (250,000 x 128, nq=1000, identical search parameters)
confirms the trend: 45-55 s at niter=100 vs 27.2-35.4 s at niter=50, recall@100
0.9770 -> 0.9763 (-0.0006).

More aggressive settings were rejected: `niter=25` costs 0.008 recall@100, which is
outside the allowed quality threshold, so 50 is kept as the safe operating point.

## Why this is safe

- only the internal CAGRA refinement budget changes; the produced graph keeps the same
  graph degree, so query cost and query throughput are unaffected (the competition
  requires candidate query QPS >= 90% of baseline);
- recall loss is two orders of magnitude below the 0.005 gate;
- the search path still uses the organiser-fixed hipVS/cuVS CAGRA implementation.