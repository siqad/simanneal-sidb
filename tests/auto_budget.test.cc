#include "tests/catch2_wrapper.hpp"
#include "src/simanneal.h"
#include <functional>

namespace {
phys::SimParams scoped(int count=6) {
    phys::SimParams p;
    std::vector<phys::EuclCoord> sites;
    for (int i=0; i<count; ++i) sites.emplace_back(i*7.68,0);
    p.setDBLocs(sites);
    p.mu=-.32;
    p.population_backend=phys::PopulationBackend::Portable;
    p.num_workers=1;
    return p;
}
void budget(phys::SimParams p, int cycles, int instances, int hops, bool selected) {
    phys::SimAnneal model(p);
    const auto &e=model.effectiveParams();
    REQUIRE(e.anneal_cycles==cycles);
    REQUIRE(e.num_instances==instances);
    REQUIRE(e.hop_attempt_factor==hops);
    REQUIRE(e.budget_auto_selected==selected);
    REQUIRE(e.requested_anneal_cycles==p.anneal_cycles);
    REQUIRE(e.requested_instances==p.num_instances);
    REQUIRE(e.requested_hop_attempt_factor==p.hop_attempt_factor);
}
}

TEST_CASE("Scoped Auto budgets cover size boundaries and preserve requests") {
    phys::SimParams defaults;
    REQUIRE(defaults.anneal_cycles==phys::AutoAnnealCycles);
    REQUIRE(defaults.num_instances==phys::AutoInstances);
    REQUIRE(defaults.hop_attempt_factor==phys::AutoHopAttempts);
    for (int count:{1,2,9,10,25,26,35,36}) {
        const int stock=count<=9 ? 16 : count<=25 ? 32 : 128;
        const bool eligible=count>=2 && count<=35;
        budget(scoped(count),eligible ? 256 : 10000,eligible ? stock/2 : stock,
               eligible ? 2 : 5,eligible);
    }
    auto p=scoped();
    p.anneal_cycles=10000;
    budget(p,10000,8,2,true);
    p.num_instances=16;
    budget(p,10000,16,2,true);
    p.hop_attempt_factor=5;
    budget(p,10000,16,5,false);
    p=scoped(); p.num_instances=16; p.hop_attempt_factor=5;
    budget(p,256,16,5,true);
    p=scoped(); p.anneal_cycles=10000; p.hop_attempt_factor=5;
    budget(p,10000,8,5,true);
    p=scoped(); p.hop_attempt_factor=0;
    budget(p,256,8,0,true);
    p=scoped(); p.num_instances=-1;
    budget(p,256,16,2,true);
}

TEST_CASE("Scoped Auto budgets reject invalid sentinels") {
    auto p=scoped(); p.anneal_cycles=-2;
    REQUIRE_THROWS_AS(phys::SimAnneal(p),std::invalid_argument);
    p=scoped(); p.num_instances=-3;
    REQUIRE_THROWS_AS(phys::SimAnneal(p),std::invalid_argument);
    p=scoped(); p.hop_attempt_factor=-2;
    REQUIRE_THROWS_AS(phys::SimAnneal(p),std::invalid_argument);
}

TEST_CASE("Every qualification condition gates scoped Auto budgets") {
    const std::vector<std::function<void(phys::SimParams &)>> exclusions={
        [](phys::SimParams &p){p.mu=-.25;},
        [](phys::SimParams &p){p.eps_r=5.7;},
        [](phys::SimParams &p){p.debye_length=4.9;},
        [](phys::SimParams &p){p.v_ext[0]=.01;},
        [](phys::SimParams &p){p.v_fc[0]=.01;},
        [](phys::SimParams &p){p.v_ext[0]=.01;p.v_fc[0]=-.01;},
        [](phys::SimParams &p){p.search_profile=phys::SearchProfile::Legacy;},
        [](phys::SimParams &p){p.random_backend=phys::RandomBackend::MT;},
        [](phys::SimParams &p){p.repair=phys::FeatureSetting::Disabled;},
        [](phys::SimParams &p){p.singleton_shortcut=phys::FeatureSetting::Disabled;},
        [](phys::SimParams &p){p.hop_selection=phys::LocalDistanceHop;},
        [](phys::SimParams &p){p.T_init=501;},
        [](phys::SimParams &p){p.T_min=3;},
        [](phys::SimParams &p){p.T_schedule=phys::LinearSchedule;},
        [](phys::SimParams &p){p.T_e_inv_point=.1;},
        [](phys::SimParams &p){p.v_freeze_end_point=.5;},
        [](phys::SimParams &p){p.v_freeze_init=0;},
        [](phys::SimParams &p){p.v_freeze_reset=0;},
        [](phys::SimParams &p){p.v_freeze_threshold=5;},
        [](phys::SimParams &p){p.phys_validity_check_cycles=11;},
        [](phys::SimParams &p){p.preanneal_cycles=1;},
        [](phys::SimParams &p){p.population_probability_cache=true;},
        [](phys::SimParams &p){p.transient_domain_mask=true;},
        [](phys::SimParams &p){p.record_history=true;},
        [](phys::SimParams &p){p.probability_shortcuts=false;},
        [](phys::SimParams &p){p.strategic_v_freeze_reset=true;},
        [](phys::SimParams &p){p.reset_T_during_v_freeze_reset=true;},
        [](phys::SimParams &p){p.refinement_options.mode=phys::refinement::Mode::K6;}
    };
    for (std::size_t i=0; i<exclusions.size(); ++i) {
        INFO("qualification condition=" << i);
        auto p=scoped(); exclusions[i](p);
        budget(p,10000,16,5,false);
    }
    auto p=scoped(); p.population_backend=phys::PopulationBackend::Auto;
#ifdef SIMANNEAL_HAVE_ACCELERATE
    budget(p,10000,16,5,false);
#else
    budget(p,256,8,2,true);
#endif
}

TEST_CASE("Auto and explicit scoped budgets produce identical seeded results") {
    auto p=scoped(); p.deterministic_seed=true; p.random_seed=731;
    phys::SuggestedResults expected;
    {
        phys::SimAnneal model(p); model.invokeSimAnneal();
        expected=model.suggestedResults();
        REQUIRE(model.searchStats().executed_restarts==8);
    }
    p.anneal_cycles=256; p.num_instances=8; p.hop_attempt_factor=2;
    phys::SimAnneal model(p); model.invokeSimAnneal();
    const auto &actual=model.suggestedResults();
    REQUIRE(actual.size()==expected.size());
    REQUIRE(model.searchStats().executed_restarts==8);
    for (std::size_t i=0; i<actual.size(); ++i) {
        REQUIRE(phys::SimAnneal::configToStr(actual[i].config)==
                phys::SimAnneal::configToStr(expected[i].config));
        REQUIRE(actual[i].system_energy==expected[i].system_energy);
    }
}
