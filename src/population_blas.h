// Copyright 2026 SiQAD contributors. Licensed under Apache-2.0.
#ifndef SIMANNEAL_POPULATION_BLAS_H
#define SIMANNEAL_POPULATION_BLAS_H

// Supported OpenBLAS CBLAS API. Matrices are full row-major n by n arrays.
// symv consumes the upper triangle; callers must guarantee symmetry.
namespace simanneal_blas {
bool openblasAvailable();
// OpenBLAS thread control is process-global. The caller serializes scopes under
// the single-active-model contract. Unrelated concurrent BLAS users cannot be
// isolated: they see this temporary setting too.
class ScopedThreadCount {
public:
    explicit ScopedThreadCount(int threads=1, bool enabled=true);
    ~ScopedThreadCount() noexcept;
    ScopedThreadCount(const ScopedThreadCount&)=delete;
    ScopedThreadCount& operator=(const ScopedThreadCount&)=delete;
private:
    int previous_=0;
};
int threadCount();
// False means unavailable or invalid dimensions/pointers; y remains unchanged.
// Input x and output y must not alias. Nonfinite-matrix fallback belongs to caller.
bool gemv(const double* matrix, const double* x, double* y, int n,
          double alpha=1.0, double beta=0.0);
bool symv(const double* matrix, const double* x, double* y, int n,
          double alpha=1.0, double beta=0.0);
}
#endif
