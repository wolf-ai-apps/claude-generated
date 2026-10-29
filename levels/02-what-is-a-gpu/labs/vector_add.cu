// Lab 2a: your first CUDA kernel.
//
//   nvcc -O2 -o vector_add vector_add.cu && ./vector_add
//
// Mental model:
//   - A kernel is a function that runs once per thread.
//   - Threads are grouped into blocks (here 256 threads); blocks form a grid.
//   - The hardware runs threads in groups of 32 called warps, on SMs.
//   - Each thread computes its global index and handles one element.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t e = (call);                                                  \
    if (e != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e),     \
              __FILE__, __LINE__);                                           \
      exit(1);                                                               \
    }                                                                        \
  } while (0)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;  // global thread index
  if (i < n) {                                    // the last block may be partly outside the array
    c[i] = a[i] + b[i];
  }
}

int main() {
  const int n = 1 << 26;  // 67M floats = 256 MiB per array
  const size_t bytes = n * sizeof(float);

  float *ha = (float*)malloc(bytes), *hb = (float*)malloc(bytes), *hc = (float*)malloc(bytes);
  for (int i = 0; i < n; i++) { ha[i] = 1.0f; hb[i] = 2.0f; }

  float *da, *db, *dc;
  CHECK(cudaMalloc(&da, bytes));
  CHECK(cudaMalloc(&db, bytes));
  CHECK(cudaMalloc(&dc, bytes));

  cudaEvent_t t0, t1, t2, t3;
  cudaEventCreate(&t0); cudaEventCreate(&t1); cudaEventCreate(&t2); cudaEventCreate(&t3);

  cudaEventRecord(t0);
  CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));  // over PCIe
  CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));
  cudaEventRecord(t1);

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  vector_add<<<blocks, threads>>>(da, db, dc, n);  // launch: grid of blocks x threads
  CHECK(cudaGetLastError());
  cudaEventRecord(t2);

  CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
  cudaEventRecord(t3);
  cudaEventSynchronize(t3);

  float copy_in, kernel, copy_out;
  cudaEventElapsedTime(&copy_in, t0, t1);
  cudaEventElapsedTime(&kernel, t1, t2);
  cudaEventElapsedTime(&copy_out, t2, t3);

  for (int i = 0; i < n; i++) {
    if (hc[i] != 3.0f) { fprintf(stderr, "wrong result at %d\n", i); return 1; }
  }
  double gb = 3.0 * bytes / 1e9;  // kernel reads a, b and writes c
  printf("blocks=%d threads/block=%d\n", blocks, threads);
  printf("copy host->device: %7.2f ms\n", copy_in);
  printf("kernel:            %7.2f ms  (%.0f GB/s of HBM bandwidth)\n", kernel, gb / (kernel / 1e3));
  printf("copy device->host: %7.2f ms\n", copy_out);
  printf("The kernel is tiny compared with the copies: keep data on the GPU.\n");

  cudaFree(da); cudaFree(db); cudaFree(dc);
  free(ha); free(hb); free(hc);
  return 0;
}
