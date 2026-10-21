#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int fib_serial(int n) { return n < 2 ? n : fib_serial(n - 1) + fib_serial(n - 2); }
int fib(int n) {
  int i, j;
  if (n < 2) return n;
  if (n < 20) return fib_serial(n);          /* cutoff: avoid creating tiny tasks */
  #pragma omp task shared(i)
  i = fib(n - 1);
  #pragma omp task shared(j)
  j = fib(n - 2);
  #pragma omp taskwait
  return i + j;
}
int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 40, r;
  double t0 = omp_get_wtime(); r = fib_serial(n); double ts = omp_get_wtime() - t0;
  t0 = omp_get_wtime();
  #pragma omp parallel
  {
    #pragma omp single
    r = fib(n);
  }
  double tp = omp_get_wtime() - t0;
  printf("n=%d threads=%d fib=%d serial=%.3fs parallel=%.3fs speedup=%.2f\n", n, omp_get_max_threads(), r, ts, tp, ts / tp);
  return 0;
}
