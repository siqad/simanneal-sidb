#include "tests/catch2_wrapper.hpp"
#include "src/simanneal.h"
#include "src/population_blas.h"
#include <limits>

namespace {
phys::SimParams backend_fixture(phys::PopulationBackend backend) {
    phys::SimParams sp;
    std::vector<phys::EuclCoord> points;
    for (int i=0;i<65;++i) points.emplace_back((i%13)*7.68, (i/13)*7.68);
    sp.setDBLocs(points);
    sp.v_fc.clear();
    for (unsigned i=0;i<sp.v_ext.size();++i) sp.v_ext[i]=(static_cast<int>(i%7)-3)*.012;
    sp.mu=-.05;
    sp.num_instances=1; sp.num_workers=1;
    sp.anneal_cycles=128; sp.result_queue_factor=1; sp.record_history=true;
    sp.population_backend=backend;
    return sp;
}

#ifdef SIMANNEAL_HAVE_OPENBLAS
TEST_CASE("OpenBLAS construction preserves caller threads and invocation restores them") {
    const int original=simanneal_blas::threadCount();
    {
        const simanneal_blas::ScopedThreadCount caller_threads(2);
        for (auto backend : {phys::PopulationBackend::OpenBLAS,
                             phys::PopulationBackend::OpenBLASSymmetric}) {
            auto sp=backend_fixture(backend);
            sp.deterministic_seed=true;
            sp.random_seed=731;
            phys::SimAnneal master(sp);
            REQUIRE(simanneal_blas::threadCount()==2);
            master.invokeSimAnneal();
            REQUIRE(simanneal_blas::threadCount()==2);
            REQUIRE(master.searchStats().executed_restarts==1);
            REQUIRE(master.searchStats().population_blas_threads==1);
            // Deterministic pre-worker allocation failure exercises unwind.
            phys::SimAnneal::sim_params.num_instances=-1;
            REQUIRE_THROWS(master.invokeSimAnneal());
            REQUIRE(simanneal_blas::threadCount()==2);
            phys::SimAnneal::sim_params.num_instances=1;
            master.invokeSimAnneal();
            REQUIRE(simanneal_blas::threadCount()==2);
        }
    }
    REQUIRE(simanneal_blas::threadCount()==original);
}

TEST_CASE("OpenBLAS singleton early return restores caller thread count") {
    const int original=simanneal_blas::threadCount();
    {
        const simanneal_blas::ScopedThreadCount caller_threads(2);
        phys::SimParams sp;
        sp.setDBLocs(std::vector<phys::EuclCoord>{phys::EuclCoord(0,0)});
        sp.v_ext[0]=1;
        sp.singleton_shortcut=phys::FeatureSetting::Enabled;
        sp.population_backend=phys::PopulationBackend::OpenBLAS;
        sp.num_instances=1; sp.num_workers=1; sp.anneal_cycles=8;
        phys::SimAnneal master(sp);
        REQUIRE(simanneal_blas::threadCount()==2);
        master.invokeSimAnneal();
        REQUIRE(master.searchStats().singleton_used);
        REQUIRE(master.searchStats().population_blas_threads==1);
        REQUIRE(master.searchStats().executed_restarts==0);
        REQUIRE(simanneal_blas::threadCount()==2);
    }
    REQUIRE(simanneal_blas::threadCount()==original);
}
#endif
phys::ThreadChargeResults backend_history(phys::SimParams sp, std::uint64_t seed) {
    phys::SimAnneal master(sp);
    phys::SimAnnealThread worker(0,seed);
    worker.run();
    return master.chargeResults()[0];
}
}

TEST_CASE("Population backends preserve incremental energy and charge bounds") {
    std::vector<phys::PopulationBackend> backends{phys::PopulationBackend::Portable,
                                                 phys::PopulationBackend::Auto};
#ifdef SIMANNEAL_HAVE_ACCELERATE
    backends.push_back(phys::PopulationBackend::Accelerate);
#endif
#ifdef SIMANNEAL_HAVE_OPENBLAS
    backends.push_back(phys::PopulationBackend::OpenBLAS);
    backends.push_back(phys::PopulationBackend::OpenBLASSymmetric);
#endif
    for (auto backend : backends) {
        for (std::uint64_t seed : {731ULL,998ULL}) {
            auto sp=backend_fixture(backend);
            phys::SimAnneal master(sp);
            REQUIRE(phys::SimAnneal::sim_params.population_finite_matrix);
            phys::SimAnnealThread worker(0,seed);
            worker.run();
            REQUIRE(master.chargeResults()[0].size()==128);
            for (const auto &result : master.chargeResults()[0]) {
                REQUIRE(std::isfinite(result.system_energy));
                REQUIRE(result.system_energy==Approx(phys::SimAnneal::systemEnergy(result.config)).margin(1e-10));
                for (int charge : result.config) REQUIRE((charge>=-1 && charge<=1));
            }
        }
    }
}

#ifndef SIMANNEAL_HAVE_OPENBLAS
TEST_CASE("Explicit unavailable OpenBLAS backends fail before simulation") {
    for (auto backend : {phys::PopulationBackend::OpenBLAS,
                         phys::PopulationBackend::OpenBLASSymmetric}) {
        auto sp=backend_fixture(backend);
        REQUIRE_THROWS_AS(phys::SimAnneal(sp),std::invalid_argument);
    }
}
#endif

TEST_CASE("Auto selects the compiled population backend") {
    auto actual=backend_history(backend_fixture(phys::PopulationBackend::Auto),731);
#ifdef SIMANNEAL_HAVE_ACCELERATE
    auto expected=backend_history(backend_fixture(phys::PopulationBackend::Accelerate),731);
#else
    auto expected=backend_history(backend_fixture(phys::PopulationBackend::Portable),731);
    auto unavailable=backend_fixture(phys::PopulationBackend::Accelerate);
    REQUIRE_THROWS_AS(phys::SimAnneal(unavailable),std::invalid_argument);
#endif
    REQUIRE(actual.size()==expected.size());
    for (unsigned i=0;i<actual.size();++i) {
        REQUIRE(phys::SimAnneal::configToStr(actual[i].config)==phys::SimAnneal::configToStr(expected[i].config));
        REQUIRE(actual[i].system_energy==expected[i].system_energy);
    }
}

#ifdef SIMANNEAL_HAVE_ACCELERATE
TEST_CASE("Nonfinite geometry retains portable population behavior") {
    auto sp=backend_fixture(phys::PopulationBackend::Portable);
    // Duplicate coordinates yield an infinite coupling, exercising dense fallback.
    auto points=sp.db_locs; points[1]=points[0]; sp.setDBLocs(points);
    sp.v_ext.clear(); sp.v_fc.clear();
    auto portable=backend_history(sp,731);
    // backend_history destroys its model, so global model state is reset here.
    sp.population_backend=phys::PopulationBackend::Accelerate;
    auto accelerated=backend_history(sp,731);
    REQUIRE(portable.size()==accelerated.size());
    for (unsigned i=0;i<portable.size();++i) {
        REQUIRE(phys::SimAnneal::configToStr(portable[i].config)==phys::SimAnneal::configToStr(accelerated[i].config));
        REQUIRE((std::isnan(portable[i].system_energy) && std::isnan(accelerated[i].system_energy)));
    }
}
#endif
