/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here.
 *
 * Competition tuning (all values are compile-time defaults and can be
 * overridden with -D, which the evaluation build never does):
 *
 *   SEEKDB_CUVS_GRAPH_DEGREE        final CAGRA graph degree      (default 128)
 *   SEEKDB_CUVS_INTERMEDIATE_DEGREE NN-descent intermediate width (default 128)
 *   SEEKDB_CUVS_NN_DESCENT_NITER    NN-descent iterations         (default 50)
 *   SEEKDB_CUVS_SEARCH_WIDTH        CAGRA search_width            (default 8)
 *   SEEKDB_CUVS_ITOPK               CAGRA itopk_size (max 512)    (default 512)
 *   SEEKDB_CUVS_DTYPE               0 = float32, 1 = float16 dataset (default 0)
 *   SEEKDB_CUVS_QUERY_F16           also cast the query side to float16 when
 *                                   DTYPE=1 (default 1)
 *
 * float16 is an INTERNAL transport/build format only: the API stays float32 in
 * and float32 distances/uint32 ids out, so the SQL surface is unchanged.
 */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#ifndef SEEKDB_CUVS_GRAPH_DEGREE
#define SEEKDB_CUVS_GRAPH_DEGREE 128
#endif
#ifndef SEEKDB_CUVS_INTERMEDIATE_DEGREE
#define SEEKDB_CUVS_INTERMEDIATE_DEGREE 192
#endif
#ifndef SEEKDB_CUVS_NN_DESCENT_NITER
#define SEEKDB_CUVS_NN_DESCENT_NITER 50
#endif
#ifndef SEEKDB_CUVS_SEARCH_WIDTH
#define SEEKDB_CUVS_SEARCH_WIDTH 32
#endif
#ifndef SEEKDB_CUVS_ITOPK
#define SEEKDB_CUVS_ITOPK 512
#endif
#ifndef SEEKDB_CUVS_DTYPE
#define SEEKDB_CUVS_DTYPE 0
#endif
#ifndef SEEKDB_CUVS_QUERY_F16
#define SEEKDB_CUVS_QUERY_F16 1
#endif

typedef struct {
  cuvsResources_t         res;
  cuvsCagraIndexParams_t  ip;
  cuvsCagraIndex_t        idx;
  long                    n, dim;
  int                     dtype;      /* 0 = f32 dataset, 1 = f16 dataset */
  cuvsCagraSearchParams_t sp;         /* created once, reused per query */
  /* persistent device scratch (allocated once, grown on demand) */
  void*                   qdev;       size_t qdev_bytes;
  unsigned*               ndev;       size_t ndev_bytes;
  float*                  ddev;       size_t ddev_bytes;
  /* persistent host scratch for the f16 query cast */
  void*                   qhost;      size_t qhost_bytes;
  pthread_mutex_t         mu;
} sk_cuvs_index;

/* IEEE-754 binary16 conversion done by hand: the bridge is compiled as plain C
 * (hipcc -x c), where __half is not available. Round-to-nearest-even. */
typedef unsigned short sk_half_t;

static sk_half_t sk_f2h(float f)
{
  unsigned int x;
  memcpy(&x, &f, 4);
  const unsigned int sign = (x >> 16) & 0x8000u;
  const int exp = (int)((x >> 23) & 0xffu) - 127;
  unsigned int mant = x & 0x7fffffu;
  if (exp == 128) return (sk_half_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
  if (exp > 15) return (sk_half_t)(sign | 0x7c00u);
  if (exp < -24) return (sk_half_t)sign;
  if (exp < -14) {
    mant |= 0x800000u;
    const int shift = (-14 - exp) + 13;
    unsigned int m = mant >> shift;
    const unsigned int rem = mant & ((1u << shift) - 1u);
    const unsigned int half = 1u << (shift - 1);
    if (rem > half || (rem == half && (m & 1u))) m++;
    return (sk_half_t)(sign | m);
  }
  {
    unsigned int m = mant >> 13;
    const unsigned int rem = mant & 0x1fffu;
    unsigned int e = (unsigned int)(exp + 15);
    if (rem > 0x1000u || (rem == 0x1000u && (m & 1u))) {
      m++;
      if (m == 0x400u) { m = 0; e++; }
    }
    if (e >= 0x1fu) return (sk_half_t)(sign | 0x7c00u);
    return (sk_half_t)(sign | (e << 10) | m);
  }
}
static size_t sk_elem_size(int dtype) { return dtype ? sizeof(sk_half_t) : sizeof(float); }

static int sk_grow(cuvsResources_t res, void** ptr, size_t* cap, size_t need)
{
  if (*cap >= need) return 0;
  if (*ptr != NULL) cuvsRMMFree(res, *ptr, *cap);
  *ptr = NULL;
  *cap = 0;
  if (cuvsRMMAlloc(res, ptr, need) != CUVS_SUCCESS) { *ptr = NULL; return 1; }
  *cap = need;
  return 0;
}

static int sk_grow_host(void** ptr, size_t* cap, size_t need)
{
  if (*cap >= need) return 0;
  void* p = realloc(*ptr, need);
  if (p == NULL) return 1;
  *ptr = p; *cap = need;
  return 0;
}

static void sk_fill_dl(DLManagedTensor* t, void* data, long r, long c, int bits,
                       DLDeviceType dev, long* shape)
{
  memset(t, 0, sizeof(*t));
  shape[0] = r; shape[1] = c;
  t->dl_tensor.data = data;
  t->dl_tensor.device.device_type = dev;
  t->dl_tensor.device.device_id = 0;
  t->dl_tensor.ndim = 2;
  t->dl_tensor.dtype.code = kDLFloat;
  t->dl_tensor.dtype.bits = (uint8_t)bits;
  t->dl_tensor.dtype.lanes = 1;
  t->dl_tensor.shape = shape;
  t->dl_tensor.strides = NULL;
  t->dl_tensor.byte_offset = 0;
}

static void* sk_alloc_index(cuvsResources_t res, cuvsCagraIndex_t idx, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  if (h == NULL) return NULL;
  h->res = res; h->idx = idx; h->n = n; h->dim = dim;
  h->dtype = SEEKDB_CUVS_DTYPE;
  pthread_mutex_init(&h->mu, NULL);
  return h;
}

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  if (base == NULL || n <= 0 || dim <= 0) return NULL;
  cuvsResources_t res;
  if (cuvsResourcesCreate(&res) != CUVS_SUCCESS) return NULL;

  /* float16 path: cast the host dataset once (the tournament allows an
   * internal half-precision build as long as the SQL surface stays float32). */
  void* host_data = (void*)base;
  int   bits = 32;
  sk_half_t* hbase = NULL;
  if (SEEKDB_CUVS_DTYPE == 1) {
    hbase = (sk_half_t*)malloc((size_t)n * (size_t)dim * sizeof(sk_half_t));
    if (hbase == NULL) { cuvsResourcesDestroy(res); return NULL; }
    const long total = n * dim;
    for (long i = 0; i < total; ++i) hbase[i] = sk_f2h(base[i]);
    host_data = (void*)hbase;
    bits = 16;
  }

  long dsh[2];
  DLManagedTensor dset;
  sk_fill_dl(&dset, host_data, n, dim, bits, kDLCPU, dsh);

  cuvsCagraIndexParams_t ip = NULL;
  cuvsCagraIndex_t idx = NULL;
  void* out = NULL;
  if (cuvsCagraIndexParamsCreate(&ip) != CUVS_SUCCESS) goto fail;
  ip->build_algo = NN_DESCENT;
  ip->graph_degree = SEEKDB_CUVS_GRAPH_DEGREE;
  ip->intermediate_graph_degree = SEEKDB_CUVS_INTERMEDIATE_DEGREE;
  ip->nn_descent_niter = SEEKDB_CUVS_NN_DESCENT_NITER;
  if (cuvsCagraIndexCreate(&idx) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraBuild(res, ip, &dset, idx) != CUVS_SUCCESS) goto fail;
  {
    sk_cuvs_index* h = (sk_cuvs_index*)sk_alloc_index(res, idx, n, dim);
    if (h == NULL) goto fail;
    h->ip = ip;
    h->sp = NULL;
    out = h;
  }
  if (hbase != NULL) { free(hbase); hbase = NULL; }
  return out;

fail:
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  if (idx != NULL) cuvsCagraIndexDestroy(idx);
  if (ip != NULL) cuvsCagraIndexParamsDestroy(ip);
  if (hbase != NULL) free(hbase);
  cuvsResourcesDestroy(res);
  return NULL;
}

/* Make sure the cached search params exist and describe this query. */
static int sk_prepare_params(sk_cuvs_index* h, long topk)
{
  if (h->sp == NULL) {
    if (cuvsCagraSearchParamsCreate(&h->sp) != CUVS_SUCCESS) { h->sp = NULL; return 1; }
    size_t itopk = (size_t)SEEKDB_CUVS_ITOPK;
    if (itopk > 512) itopk = 512;            /* hipVS hard limit */
    if (itopk < (size_t)topk) itopk = (size_t)topk;
    h->sp->itopk_size = itopk;
    h->sp->search_width = (size_t)SEEKDB_CUVS_SEARCH_WIDTH;
  }
  return 0;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (h == NULL || query == NULL || out_ids == NULL || nq <= 0 || topk <= 0) return 1;
  int rc = 0;
  pthread_mutex_lock(&h->mu);
  const size_t esz = sk_elem_size(h->dtype);
  const size_t qneed = (size_t)nq * (size_t)h->dim * esz;
  const size_t oneed = sizeof(unsigned) * (size_t)nq * (size_t)topk;
  const size_t dneed = sizeof(float) * (size_t)nq * (size_t)topk;

  if (sk_prepare_params(h, topk) != 0) { rc = 14; goto done; }
  if (sk_grow(h->res, &h->qdev, &h->qdev_bytes, qneed) != 0) { rc = 10; goto done; }
  if (sk_grow(h->res, (void**)&h->ndev, &h->ndev_bytes, oneed) != 0) { rc = 11; goto done; }
  if (out_dist != NULL && sk_grow(h->res, (void**)&h->ddev, &h->ddev_bytes, dneed) != 0) { rc = 12; goto done; }
  {
    const void* hsrc = (const void*)query;
    if (h->dtype == 1 && SEEKDB_CUVS_QUERY_F16) {
      const size_t hneed = (size_t)nq * (size_t)h->dim * sizeof(sk_half_t);
      if (sk_grow_host(&h->qhost, &h->qhost_bytes, hneed) != 0) { rc = 17; goto done; }
      sk_half_t* hq = (sk_half_t*)h->qhost;
      const long total = nq * h->dim;
      const float* src = query;
      for (long i = 0; i < total; ++i) hq[i] = sk_f2h(src[i]);
      hsrc = h->qhost;
    }
    if (cudaMemcpy(h->qdev, hsrc, qneed, cudaMemcpyHostToDevice) != cudaSuccess) { rc = 13; goto done; }
  }
  {
    long qs[2], ns[2], ds[2];
    DLManagedTensor qt, nt, dt;
    sk_fill_dl(&qt, h->qdev, nq, h->dim, h->dtype == 1 ? 16 : 32, kDLCUDA, qs);
    sk_fill_dl(&nt, h->ndev, nq, topk, 32, kDLCUDA, ns);
    nt.dl_tensor.dtype.code = kDLUInt;
    sk_fill_dl(&dt, h->ddev, nq, topk, 32, kDLCUDA, ds);
    cuvsFilter f; f.type = NO_FILTER; f.addr = (uintptr_t)0;
    if (cuvsCagraSearch(h->res, h->sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc = 15; goto done; }
    if (cudaMemcpy(out_ids, h->ndev, oneed, cudaMemcpyDeviceToHost) != cudaSuccess) { rc = 16; goto done; }
    if (out_dist != NULL &&
        cudaMemcpy(out_dist, h->ddev, dneed, cudaMemcpyDeviceToHost) != cudaSuccess) { rc = 18; goto done; }
  }
done:
  pthread_mutex_unlock(&h->mu);
  if (rc != 0) fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText());
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (h == NULL) return;
  if (h->qdev != NULL) cuvsRMMFree(h->res, h->qdev, h->qdev_bytes);
  if (h->ndev != NULL) cuvsRMMFree(h->res, h->ndev, h->ndev_bytes);
  if (h->ddev != NULL) cuvsRMMFree(h->res, h->ddev, h->ddev_bytes);
  if (h->qhost != NULL) free(h->qhost);
  if (h->sp != NULL) cuvsCagraSearchParamsDestroy(h->sp);
  cuvsCagraIndexDestroy(h->idx);
  if (h->ip != NULL) cuvsCagraIndexParamsDestroy(h->ip);
  cuvsResourcesDestroy(h->res);
  pthread_mutex_destroy(&h->mu);
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  if (h == NULL || filename == NULL) return 1;
  if (cuvsCagraSerialize(h->res, filename, h->idx, true) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    return 1;
  }
  return 0;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  if (filename == NULL || n <= 0 || dim <= 0) return NULL;
  cuvsResources_t res;
  if (cuvsResourcesCreate(&res) != CUVS_SUCCESS) return NULL;
  cuvsCagraIndexParams_t ip = NULL;
  cuvsCagraIndex_t idx = NULL;
  if (cuvsCagraIndexParamsCreate(&ip) != CUVS_SUCCESS) goto fail;
  ip->build_algo = NN_DESCENT;
  ip->graph_degree = SEEKDB_CUVS_GRAPH_DEGREE;
  ip->intermediate_graph_degree = SEEKDB_CUVS_INTERMEDIATE_DEGREE;
  ip->nn_descent_niter = SEEKDB_CUVS_NN_DESCENT_NITER;
  if (cuvsCagraIndexCreate(&idx) != CUVS_SUCCESS) goto fail;
  if (cuvsCagraDeserialize(res, filename, idx) != CUVS_SUCCESS) goto fail;
  {
    int64_t actual_n = 0, actual_dim = 0;
    if (cuvsCagraIndexGetSize(idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail;
    }
  }
  {
    sk_cuvs_index* h = (sk_cuvs_index*)sk_alloc_index(res, idx, n, dim);
    if (h == NULL) goto fail;
    h->ip = ip;
    return h;
  }
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  if (idx != NULL) cuvsCagraIndexDestroy(idx);
  if (ip != NULL) cuvsCagraIndexParamsDestroy(ip);
  cuvsResourcesDestroy(res);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  if (h == NULL) return 10;
  int rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}