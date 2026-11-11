# hipVS bridge tuning (build + query)

## Configuration

| parameter | frozen baseline | this change |
|---|---|---|
| `graph_degree` | 128 | 128 (unchanged) |
| `intermediate_graph_degree` | 256 | **192** |
| `nn_descent_niter` | 100 | **50** |
| `search_width` | 4 | **8** |
| `itopk_size` | 512 | 512 (hipVS hard limit) |
| **OpenMP build team** | online CPUs (128 here) | **16** |

Also: device scratch buffers and search parameters are kept per index instead of being
allocated/freed on every query.

## The OpenMP finding

hipVS builds the CAGRA graph with an OpenMP team sized by the number of online CPUs. On a
128-core host that spreads each NN-descent pass across all NUMA/CCD domains and the build
gets much slower than with a small cache-local team. Measured with the unchanged build
parameters (250k x 128, gd128/igd192/niter50):

| OpenMP team | CAGRA build |
|---|---|
| online CPUs (128) | 20 784 ms |
| 32 | 10 462 ms |
| 24 | 9 580 ms |
| **16** | **8 143 ms (2.57x)** |
| 1 | 31 863 ms |

`omp_set_num_threads()` alone does not help (hipVS runs its parallel regions on its own
team); `kmp_set_defaults("OMP_NUM_THREADS=16")` does, and is equivalent to exporting
`OMP_NUM_THREADS=16` before start-up.

End-to-end official DDL path, 250,000 x 128, same binary, only the bridge .so swapped:

| | build | recall@10 | recall@100 |
|---|---|---|---|
| before clamp | 23.62 s | 0.9900 | 0.9639 |
| **after clamp** | **11.27 s** | 0.9905 | 0.9708 |

The graph and the number of iterations are identical, so recall is unchanged; 0 invalid
rows and 0 duplicate ids in every run, VRAM peak equal to baseline, GPU use peak 100%.

Note on the search width: `search_width=32` was measured to be faster on the local
128-core host but scored worse on the evaluation service, so it is kept at 8 here.

## Rejected knobs (measured, kept for the record)

| knob | effect | verdict |
|---|---|---|
| `nn_descent_niter` 50 -> 24/25 | gauss-256d recall@100 -0.024 | rejected (gate 0.005) |
| `intermediate_graph_degree` 192 -> 128 | gauss-256d recall@10 -0.015 | rejected |
| `graph_degree` 128 -> 96/112 | uniform-128d recall@100 -0.009..-0.025 | rejected |
| FP16 internal build | 1.08x only vs FP32 | no benefit |
| `search_width` 8 -> 32 | faster locally, worse on the service | rejected |

All build knobs that would speed the CAGRA construction up further cost recall beyond the
0.005 gate, so the build parameters stay at gd=128 / igd=192 / niter=50.