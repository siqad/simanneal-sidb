#include "tests/catch2_wrapper.hpp"
#include "src/simanneal.h"
#include "src/interface.h"
#include <utility>

namespace {
phys::SimParams storageParams() {
    phys::SimParams sp;
    sp.setDBLocs(std::vector<phys::EuclCoord>{{0,0},{7.68,0},{0,7.68},
        {7.68,7.68},{15.36,0},{15.36,7.68}});
    sp.v_ext.clear(); sp.v_fc.clear();
    sp.num_instances=4; sp.num_workers=1; sp.anneal_cycles=64;
    sp.deterministic_seed=true; sp.random_seed=731;
    return sp;
}
}

TEST_CASE("Consumed parameter matrices retain storage and seeded results") {
    auto sp=storageParams();
    phys::SuggestedResults expected;
    {
        phys::SimAnneal copied(sp);
        REQUIRE(&copied.effectiveParams().db_r.data()[0] != &sp.db_r.data()[0]);
        REQUIRE(&copied.effectiveParams().v_ij.data()[0] != &sp.v_ij.data()[0]);
        copied.invokeSimAnneal();
        expected=copied.suggestedResults();
    }
    REQUIRE(sp.db_locs.size()==6);
    REQUIRE(sp.db_r.size1()==6);
    REQUIRE(sp.v_ij.size1()==6);
    const auto *distances=&sp.db_r.data()[0];
    const auto *potentials=&sp.v_ij.data()[0];
    {
        phys::SimAnneal consumed(std::move(sp));
        REQUIRE(&consumed.effectiveParams().db_r.data()[0]==distances);
        REQUIRE(&consumed.effectiveParams().v_ij.data()[0]==potentials);
        consumed.invokeSimAnneal();
        const auto &actual=consumed.suggestedResults();
        REQUIRE(actual.size()==expected.size());
        for (std::size_t i=0; i<actual.size(); ++i) {
            REQUIRE(phys::SimAnneal::configToStr(actual[i].config)==
                    phys::SimAnneal::configToStr(expected[i].config));
            REQUIRE(actual[i].system_energy==expected[i].system_energy);
            REQUIRE(actual[i].pop_likely_stable==expected[i].pop_likely_stable);
        }
    }
    sp=storageParams();
    phys::SimAnneal reused(sp);
    reused.invokeSimAnneal();
    REQUIRE(reused.suggestedResults().size()==expected.size());
}

TEST_CASE("Rejected consuming constructors preserve the active model lifecycle") {
    auto sp=storageParams();
    {
        phys::SimAnneal active(sp);
        auto rejected=storageParams();
        const auto *storage=&rejected.v_ij.data()[0];
        REQUIRE_THROWS_AS(phys::SimAnneal(std::move(rejected)),std::logic_error);
        REQUIRE(rejected.db_locs.size()==6);
        REQUIRE(&rejected.v_ij.data()[0]==storage);
        active.invokeSimAnneal();
        REQUIRE(active.suggestedResults().size()==4);
    }
    auto invalid=storageParams();
    invalid.num_instances=0;
    REQUIRE_THROWS_AS(phys::SimAnneal(std::move(invalid)),std::invalid_argument);
    phys::SimAnneal recovered(storageParams());
    recovered.invokeSimAnneal();
    REQUIRE(recovered.suggestedResults().size()==4);
}

TEST_CASE("Simulation interface consumes temporaries and preserves const parameters") {
    phys::SimAnnealInterface interface(
        std::string(SIMANNEAL_TEST_SOURCE_DIR)+"/sample_problems/or_00_problem.xml",
        "simanneal-parameter-storage-results.xml", "", 0);
    auto temporary=storageParams();
    const auto *distances=&temporary.db_r.data()[0];
    const auto *potentials=&temporary.v_ij.data()[0];
    REQUIRE(interface.runSimulation(std::move(temporary))==0);
    REQUIRE(&phys::SimAnneal::sim_params.db_r.data()[0]==distances);
    REQUIRE(&phys::SimAnneal::sim_params.v_ij.data()[0]==potentials);

    const auto parameters=storageParams();
    distances=&parameters.db_r.data()[0];
    potentials=&parameters.v_ij.data()[0];
    for (int run=0; run<2; ++run) {
        REQUIRE(interface.runSimulation(parameters)==0);
        REQUIRE(parameters.db_locs.size()==6);
        REQUIRE(parameters.db_r.size1()==6);
        REQUIRE(parameters.v_ij.size1()==6);
        REQUIRE(&parameters.db_r.data()[0]==distances);
        REQUIRE(&parameters.v_ij.data()[0]==potentials);
        REQUIRE(&phys::SimAnneal::sim_params.db_r.data()[0]!=distances);
        REQUIRE(&phys::SimAnneal::sim_params.v_ij.data()[0]!=potentials);
    }
}
