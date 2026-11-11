# hipVS bridge tuning (portable variant)

## Configuration

| parameter | frozen baseline | this change |
|---|---|---|
| `graph_degree` | 128 | 128 (unchanged) |
| `intermediate_graph_degree` | 256 | **192** |
| `nn_descent_niter` | 100 | **50** |
| `search_width` | 4 | **32** |
| `itopk_size` | 512 | 512 (hipVS hard limit) |
| OpenMP build team | visible CPUs | **min(16, visible CPUs)**, set via `setenv` |

Device scratch buffers and CAGRA search parameters are kept per index instead of being
allocated/freed on every query.

## Why the OpenMP clamp

hipVS builds the CAGRA graph with an OpenMP team sized from the visible CPUs. On a host with
many cores the NN-descent passes are spread across several NUMA/CCD domains and the build
becomes much slower than with a small, cache-local team. Measured with unchanged build
parameters (250k x 128, gd128/igd192/niter50):

| team size | CAGRA build |
|---|---|
| visible CPUs (128) | 20 784 ms |
| 32 | 10 462 ms |
| 24 | 9 580 ms |
| 16 | 8 143 ms |
| 1 | 31 863 ms |

The clamp is applied with `setenv("OMP_NUM_THREADS", ...)` before the first build, so it needs
no OpenMP runtime headers and no `-fopenmp` in the bridge build: the bridge simply exports the
value that libomp/libgomp read when their team is first sized. The value is
`min(SEEKDB_CUVS_OMP_THREADS, visible CPUs)` where the visible CPU count comes from
`sched_getaffinity`, so a machine that is already confined to a small CPU mask keeps its own
width and cannot be slowed down by the clamp.

## Measurements (250,000 rows, official DDL path)

| version | dim | build | QPS T10 | QPS T100 | recall@10 | recall@100 |
|---|---|---|---|---|---|---|
| frozen baseline | 128 | 45.19 s | 138.8 | 129.2 | 0.9900 | 0.9639 |
| frozen baseline | 256 | 48.84 s | 131.1 | 122.1 | 0.9145 | 0.8622 |
| **this change** | 128 | **11.36 s** | 182.7 | 163.0 | **0.9905** | **0.9689** |
| **this change** | 256 | **13.47 s** | 169.8 | 157.6 | **0.9325** | **0.8786** |

Build speedup 3.98x / 3.63x (weighted 3.76x), query speedup 1.289 / 1.293 (weighted 1.291x),
recall above the baseline on both data sets, 0 invalid and 0 duplicate ids, VRAM within the
baseline peak, GPU use peak 100% during the build.

## Rejected knobs (measured)

| knob | effect | verdict |
|---|---|---|
| `nn_descent_niter` 50 -> 25 | gauss-256d recall@10 -0.014 | rejected (gate 0.005) |
| `intermediate_graph_degree` 192 -> 128 | gauss-256d recall@10 -0.015 | rejected |
| `graph_degree` 128 -> 96/112 | uniform-128d recall@100 -0.009..-0.025 | rejected |
| FP16 internal build | 1.08x only | no benefit |
| `itopk_size` > 512 | hipVS hard limit | not possible |

The seekdb-side "store the snapshot vectors once" optimisation is deliberately **not** part of
this change: it touches `src/oblib/...`, so this submission keeps the frozen database sources
untouched and only tunes the bridge.