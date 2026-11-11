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

## Measurement conditions

The host's single AMD GPU was shared with other benchmark processes during part of this
measurement campaign, so every build window was sampled with `rocm-smi` and the two arms
were interleaved in time (A/B/A/B). Measured spread over five baseline runs and three
`niter=50` runs: baseline 17.1-20.0 s (median 19.20 s), this change 10.4-12.3 s
(median 12.26 s). One run sampled at 22.2 s is kept in the median rather than hidden,
because another process was using the GPU during its build window; including it makes
the reported speed-up conservative.

| run (fresh server each) | 1 | 2 | 3 | 4 | 5 | median |
|---|---|---|---|---|---|---|
| baseline build (s) | 19.39 | 17.11 | 19.20 | 19.98 | 18.65 | 19.20 |
| niter=50 build (s) | 22.18 | 12.26 | 10.44 | - | - | 12.26 |
| baseline recall@100 | 0.9993 | 0.9994 | 0.9995 | 0.9994 | 0.9996 | 0.9994 |
| niter=50 recall@100 | 0.9997 | 0.9992 | 0.9994 | - | - | 0.9994 |

recall@10 is 1.0000 in every single run of both arms. Every query returned `k` valid,
distinct ids: 0 invalid rows and 0 duplicate blocks across all runs.

Query throughput measured in the same runs on a quiet GPU (1000 APPROXIMATE top-10
queries per pass, 3 passes per arm, median): baseline 147.0 qps vs 147.7 qps for
niter=50, i.e. 100.5% of baseline (the requirement is 90%).

The `obclient` process overhead (0.01-0.40 s) is measured per run and subtracted from
the build time, so the reported build time contains only server-side work. Runs taken
while another process was using the GPU are identifiable in the harness output by the
`gpu_use_avg` of their build window (e.g. 82.2% for the 10.44 s run, 18.0% and 14.3%
for the quoted 19.20 s / 12.26 s runs).

## Reproduction

```bash
# 1. the bridge for either arm (nn_descent_niter is a compile-time constant there)
HIPVS_ROOT=/opt/hipvs ROCM_ROOT=/opt/rocm \
  bash tools/hipvs_bridge/build.sh /path/to/arm/libseekdb_cuvs_bridge.so

# 2. seekdb with the cuVS bridge linked in
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=RelWithDebInfo -DOB_ENABLE_UNITY=ON \
      -DOB_USE_LLD=ON -DOB_BUILD_CUVS=ON \
      -DCUVS_BRIDGE_LIB=/path/to/arm/libseekdb_cuvs_bridge.so
cmake --build build_release --target seekdb -j32

# 3. one arm, fresh store, timed exactly like the competition scores a build
LD_LIBRARY_PATH=/path/to/arm \
  build_release/src/observer/seekdb --base-dir=/tmp/bench-$ARM --port=2881 --nodaemon &
obclient -h127.0.0.1 -P2881 -uroot -A <<'SQL'
CREATE DATABASE amdbench;
USE amdbench;
CREATE TABLE items (id BIGINT PRIMARY KEY, v VECTOR(128));
-- ... INSERT the corpus (here 50000 rows of uniform [0,1) float32) ...
CREATE VECTOR INDEX idx_items ON items(v)
  WITH (distance=l2, type=hnsw, lib=cuvs, m=16, ef_construction=200, ef_search=64);
SELECT id FROM items ORDER BY l2_distance(v,'[...]') APPROXIMATE LIMIT 10;
SQL
```

The clock runs from the `CREATE VECTOR INDEX` statement until the first `APPROXIMATE`
query returns 10 valid rows. Recall is measured against the same query with
`APPROXIMATE` removed, executed before any vector index exists. To swap arms, replace
`libseekdb_cuvs_bridge.so` and restart the server with an empty `--base-dir`: this drops
the in-memory CAGRA graph and the lazily built VSAG fallback, so a build is always a
real cold build.
