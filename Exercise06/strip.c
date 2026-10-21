#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1000000
#define STRIP 256   /* multiple of 8 doubles = 512-bit SIMD (and of 4 = AVX2); 2 KB per array per strip */
int main(void) {
  double *A = aligned_alloc(64, N * sizeof(double)), *B = aligned_alloc(64, N * sizeof(double)), *C = aligned_alloc(64, N * sizeof(double));
  for (int i = 0; i < N; i++) { A[i] = i * 0.5; B[i] = 2.0; }
  double t0 = omp_get_wtime();
  for (int i = 0; i < N; i++) C[i] = A[i] * B[i];
  double ts = omp_get_wtime() - t0;
  t0 = omp_get_wtime();
  #pragma omp parallel for schedule(static)
  for (int s = 0; s < N; s += STRIP) {
    int end = (s + STRIP < N) ? s + STRIP : N;
    #pragma omp simd
    for (int i = s; i < end; i++) C[i] = A[i] * B[i];
  }
  double tp = omp_get_wtime() - t0;
  int ok = 1; for (int i = 0; i < N; i++) if (C[i] != A[i] * 2.0) { ok = 0; break; }
  printf("threads=%d strip=%d correct=%d serial=%.6fs parallel=%.6fs\n", omp_get_max_threads(), STRIP, ok, ts, tp);
  return 0;
}
