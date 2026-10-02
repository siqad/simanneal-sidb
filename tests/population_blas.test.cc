#include "tests/catch2_wrapper.hpp"
#include "src/population_blas.h"
#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
std::vector<double> portable_matvec(const std::vector<double>& a,
    const std::vector<double>& x, const std::vector<double>& initial,
    double alpha, double beta) {
    const unsigned n=x.size();
    std::vector<double> result(n);
    for (unsigned i=0;i<n;++i) {
        double sum=0;
        for (unsigned j=0;j<n;++j) sum+=a[i*n+j]*x[j];
        result[i]=alpha*sum+beta*initial[i];
    }
    return result;
}
}

TEST_CASE("OpenBLAS wrapper reports availability without changing unavailable outputs") {
#ifdef SIMANNEAL_HAVE_OPENBLAS
    REQUIRE(simanneal_blas::openblasAvailable());
    const simanneal_blas::ScopedThreadCount threads;
    REQUIRE(simanneal_blas::threadCount()==1);
#else
    REQUIRE_FALSE(simanneal_blas::openblasAvailable());
    double a=2, x=3, y=17;
    REQUIRE_FALSE(simanneal_blas::gemv(&a,&x,&y,1));
    REQUIRE_FALSE(simanneal_blas::symv(&a,&x,&y,1));
    REQUIRE(y==17);
#endif
    double value=17;
    REQUIRE_FALSE(simanneal_blas::gemv(nullptr,&value,&value,0));
    REQUIRE_FALSE(simanneal_blas::symv(nullptr,&value,&value,-1));
    REQUIRE(value==17);
}

#ifdef SIMANNEAL_HAVE_OPENBLAS
TEST_CASE("OpenBLAS scoped thread control restores on return and exception") {
    const int original=simanneal_blas::threadCount();
    {
        const simanneal_blas::ScopedThreadCount caller_threads(2);
        REQUIRE(simanneal_blas::threadCount()==2);
        const auto early_return=[] {
            const simanneal_blas::ScopedThreadCount search_threads;
            REQUIRE(simanneal_blas::threadCount()==1);
            return;
        };
        early_return();
        REQUIRE(simanneal_blas::threadCount()==2);
        REQUIRE_THROWS_AS([] {
            const simanneal_blas::ScopedThreadCount search_threads;
            REQUIRE(simanneal_blas::threadCount()==1);
            throw std::runtime_error("scope exit");
        }(),std::runtime_error);
        REQUIRE(simanneal_blas::threadCount()==2);
        {
            const simanneal_blas::ScopedThreadCount disabled(1,false);
            REQUIRE(simanneal_blas::threadCount()==2);
        }
        REQUIRE(simanneal_blas::threadCount()==2);
    }
    REQUIRE(simanneal_blas::threadCount()==original);
}

TEST_CASE("OpenBLAS row-major DGEMV matches nonsymmetric portable products") {
    const simanneal_blas::ScopedThreadCount threads;
    for (unsigned n : {1u,7u,65u}) {
        std::vector<double> a(n*n),x(n),y(n);
        for (unsigned i=0;i<n;++i) {
            x[i]=static_cast<int>(i%3)-1;
            y[i]=i*.031;
            for (unsigned j=0;j<n;++j)
                a[i*n+j]=(static_cast<int>((i*13+j*7)%23)-11)*.027;
        }
        auto expected=portable_matvec(a,x,y,-.75,.25);
        REQUIRE(simanneal_blas::gemv(a.data(),x.data(),y.data(),n,-.75,.25));
        for (unsigned i=0;i<n;++i)
            REQUIRE(y[i]==Approx(expected[i]).margin(1e-11));
    }
}

TEST_CASE("OpenBLAS DSYMV reads upper triangle and supports independent workers") {
    const simanneal_blas::ScopedThreadCount threads;
    const unsigned n=65;
    std::vector<double> symmetric(n*n),upper(n*n),x(n),initial(n,.125);
    for (unsigned i=0;i<n;++i) {
        x[i]=static_cast<int>(i%3)-1;
        for (unsigned j=i;j<n;++j)
            symmetric[i*n+j]=symmetric[j*n+i]=(i==j?0:1.0/(1+j-i));
    }
    upper=symmetric;
    // Poison the ignored triangle to catch an ABI/orientation mistake.
    for (unsigned i=0;i<n;++i)
        for (unsigned j=0;j<i;++j) upper[i*n+j]=10000;
    auto expected=portable_matvec(symmetric,x,initial,1,.5);
    std::vector<std::vector<double>> results(4,initial);
    std::vector<int> completed(4,0);
    std::vector<std::thread> workers;
    for (unsigned w=0;w<results.size();++w)
        workers.emplace_back([&,w] {
            completed[w]=simanneal_blas::symv(upper.data(),x.data(),results[w].data(),n,1,.5);
        });
    for (auto& worker:workers) worker.join();
    for (unsigned w=0;w<results.size();++w) {
        REQUIRE(completed[w]);
        for (unsigned i=0;i<n;++i)
            REQUIRE(results[w][i]==Approx(expected[i]).margin(1e-11));
    }
    REQUIRE(simanneal_blas::threadCount()==1);
}
#endif
