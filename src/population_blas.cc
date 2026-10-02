// Copyright 2026 SiQAD contributors. Licensed under Apache-2.0.
#include "population_blas.h"
#ifdef SIMANNEAL_HAVE_OPENBLAS
#include <cblas.h>
#endif

namespace simanneal_blas {
bool openblasAvailable() {
#ifdef SIMANNEAL_HAVE_OPENBLAS
    return true;
#else
    return false;
#endif
}
void configureSingleThread() {
#ifdef SIMANNEAL_HAVE_OPENBLAS
    openblas_set_num_threads(1);
#endif
}
int threadCount() {
#ifdef SIMANNEAL_HAVE_OPENBLAS
    return openblas_get_num_threads();
#else
    return 0;
#endif
}
bool gemv(const double* matrix, const double* x, double* y, int n,
          double alpha, double beta) {
#ifdef SIMANNEAL_HAVE_OPENBLAS
    if (n<=0 || !matrix || !x || !y || x==y) return false;
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, alpha, matrix, n,
                x, 1, beta, y, 1);
    return true;
#else
    (void)matrix; (void)x; (void)y; (void)n; (void)alpha; (void)beta;
    return false;
#endif
}
bool symv(const double* matrix, const double* x, double* y, int n,
          double alpha, double beta) {
#ifdef SIMANNEAL_HAVE_OPENBLAS
    if (n<=0 || !matrix || !x || !y || x==y) return false;
    cblas_dsymv(CblasRowMajor, CblasUpper, n, alpha, matrix, n,
                x, 1, beta, y, 1);
    return true;
#else
    (void)matrix; (void)x; (void)y; (void)n; (void)alpha; (void)beta;
    return false;
#endif
}
}
