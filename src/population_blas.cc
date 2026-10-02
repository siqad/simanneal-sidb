// Copyright 2026 SiQAD contributors. Licensed under Apache-2.0.
#include "population_blas.h"
#include <stdexcept>
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
ScopedThreadCount::ScopedThreadCount(int threads, bool enabled) {
    if (!enabled) return;
    if (threads<1) throw std::invalid_argument("BLAS thread count must be positive");
#ifdef SIMANNEAL_HAVE_OPENBLAS
    previous_=openblas_get_num_threads();
    if (previous_<1) throw std::runtime_error("OpenBLAS returned an invalid thread count");
    if (previous_ == threads) {
        previous_ = 0;
        return;
    }
    openblas_set_num_threads(threads);
#endif
}
ScopedThreadCount::~ScopedThreadCount() noexcept {
#ifdef SIMANNEAL_HAVE_OPENBLAS
    if (previous_>0) openblas_set_num_threads(previous_);
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
