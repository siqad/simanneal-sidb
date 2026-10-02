#include "tests/catch2_wrapper.hpp"
#include "src/simanneal.h"
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
    REQUIRE_FALSE(phys::SimAnneal::sim_params.population_finite_matrix);
    sp.population_backend=phys::PopulationBackend::Accelerate;
    auto accelerated=backend_history(sp,731);
    REQUIRE(portable.size()==accelerated.size());
    for (unsigned i=0;i<portable.size();++i) {
        REQUIRE(phys::SimAnneal::configToStr(portable[i].config)==phys::SimAnneal::configToStr(accelerated[i].config));
        REQUIRE((std::isnan(portable[i].system_energy) && std::isnan(accelerated[i].system_energy)));
    }
}
#endif
