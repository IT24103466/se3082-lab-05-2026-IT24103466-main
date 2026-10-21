#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define NPOINTS 2000
#define MAXITER 2000
int main(int argc, char **argv) {
  const char *mode = argc > 1 ? argv[1] : "block";
  int numoutside = 0; const double eps = 1.0e-5; double t0 = omp_get_wtime();
  if (!strcmp(mode, "serial")) {
    for (int i = 0; i < NPOINTS; i++) for (int j = 0; j < NPOINTS; j++) {
      double cr = -2.0 + 2.5 * i / NPOINTS + eps, ci = 1.125 * j / NPOINTS + eps, zr = cr, zi = ci; int out = 0;
      for (int it = 0; it < MAXITER; it++) { double t = zr*zr - zi*zi + cr; zi = 2.0*zr*zi + ci; zr = t; if (zr*zr + zi*zi > 4.0) { out = 1; break; } }
      numoutside += out; }
  } else if (!strcmp(mode, "block")) {         /* manual equal block of rows per thread */
    #pragma omp parallel reduction(+:numoutside)
    { int nt = omp_get_num_threads(), id = omp_get_thread_num();
      int lo = id * NPOINTS / nt, hi = (id + 1) * NPOINTS / nt;
      for (int i = lo; i < hi; i++) for (int j = 0; j < NPOINTS; j++) {
        double cr = -2.0 + 2.5 * i / NPOINTS + eps, ci = 1.125 * j / NPOINTS + eps, zr = cr, zi = ci; int out = 0;
        for (int it = 0; it < MAXITER; it++) { double t = zr*zr - zi*zi + cr; zi = 2.0*zr*zi + ci; zr = t; if (zr*zr + zi*zi > 4.0) { out = 1; break; } }
        numoutside += out; } }
  } else {
    if (!strcmp(mode, "static")) omp_set_schedule(omp_sched_static, 0);
    else if (!strcmp(mode, "static1")) omp_set_schedule(omp_sched_static, 1);
    else if (!strcmp(mode, "dynamic")) omp_set_schedule(omp_sched_dynamic, 1);
    else omp_set_schedule(omp_sched_guided, 0);
    #pragma omp parallel for schedule(runtime) reduction(+:numoutside)
    for (int i = 0; i < NPOINTS; i++) for (int j = 0; j < NPOINTS; j++) {
      double cr = -2.0 + 2.5 * i / NPOINTS + eps, ci = 1.125 * j / NPOINTS + eps, zr = cr, zi = ci; int out = 0;
      for (int it = 0; it < MAXITER; it++) { double t = zr*zr - zi*zi + cr; zi = 2.0*zr*zi + ci; zr = t; if (zr*zr + zi*zi > 4.0) { out = 1; break; } }
      numoutside += out; }
  }
  double tt = omp_get_wtime() - t0; long total = (long)NPOINTS * NPOINTS, inside = total - numoutside;
  printf("mode=%s threads=%d inside=%ld area=%.6f time=%.3fs\n", mode, omp_get_max_threads(), inside, 2.0*2.5*1.125*inside/total, tt);
  return 0;
}
