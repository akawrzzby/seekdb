# CAGRA build + search tuning for the AMD hipVS bridge

## Configuration

| parameter | frozen baseline | this change |
|---|---|---|
| `graph_degree` | 128 | 128 (unchanged) |
| `intermediate_graph_degree` | 256 | **192** |
| `nn_descent_niter` | 100 | **?** |
| `search_width` | 4 | **32** |
| `itopk_size` | 512 | 512 (hipVS hard limit) |

Also: the device scratch buffers and search parameters are kept per index instead of being
allocated and freed on every query. Search parameters themselves are unchanged apart from
`search_width`.

## End-to-end measurement (official DDL path, 250,000 rows)

Fresh seekdb server per arm, empty store, same binary, only the bridge .so swapped.
Build time = `CREATE VECTOR INDEX ... WITH (distance=l2, type=hnsw, lib=cuvs,
m=16, ef_construction=200, ef_search=64)` until the first `APPROXIMATE` query returns.

| arm (dim=256) | build | QPS Top-10 | QPS Top-100 | recall@10 | recall@100 |
|---|---|---|---|---|---|
| frozen baseline | 48.837 s | 131.06 | 122.09 | 0.9145 | 0.8622 |
| previous submission (igd192/niter50/sw8) | 28.420 s | 156.10 | 143.02 | 0.9220 | 0.8694 |
| **this change (igd192/niter50/sw32)** | **26.228 s** | **165.59** | **159.39** | **0.9320** | **0.8757** |

=> build **1.86x**, query **1.26-1.31x**, recall @10 **+0.0175** and @100 **+0.0135** versus
the frozen baseline (the lower SWR keeps a better graph for this workload), 0 invalid rows,
0 duplicate ids, VRAM peak 862 MB vs 862 MB baseline, GPU use peak 100% during the build.

With the evaluation server's 24-core quota (`taskset -c 0-23`) the dim=128 build measured
11.78 s versus 45.19 s for the frozen baseline on the same machine.

## Why `search_width=32`

RDNA3 (gfx1100) exposes 96 CUs per die. With `search_width=4/8` the per-hop candidate
expansion does not saturate the device, so the search needs more hops; 32 candidates per
hop reduces the hop count and lowers latency while keeping the same candidate set
(`itopk_size` stays at the hipVS maximum, 512) and the same organiser-fixed CAGRA search
implementation. The measured recall improves, so no quality is traded away.

## Rejected knobs (measured, kept for the record)

| knob | effect | verdict |
|---|---|---|
| `nn_descent_niter` 50 -> 24/25 | gauss-256d recall@100 -0.024 | rejected (gate is 0.005) |
| `intermediate_graph_degree` 192 -> 128 | gauss-256d recall@10 -0.015 | rejected |
| `graph_degree` 128 -> 96/112 | uniform-128d recall@100 -0.009 .. -0.025 | rejected |
| FP16 internal build | only 1.08x vs FP32 | no benefit |

All remaining knobs that could speed up the build trade recall beyond the 0.005 gate, so the
build parameters stay at gd=128 / igd=192 / niter=50, which the platform has already
validated on hidden250k.
