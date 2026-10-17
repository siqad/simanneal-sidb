#include "tests/catch2_wrapper.hpp"
#include "src/simanneal.h"
#include <functional>
#include <cmath>

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
    for (int count:{1,2,9,10,25,26,35,36,62,63}) {
        const int stock=count<=9 ? 16 : count<=25 ? 32 : 128;
        const bool eligible=count>=2 && count<=62;
        budget(scoped(count),eligible ? (count<=35 ? 256 : 512) : 10000,eligible && count<=35 ? stock/2 : stock,
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

TEST_CASE("Scoped Auto budgets cover the qualified mu range and numerical backends") {
    for (double mu:{-.32,-.30,-.28,-.27,-.26,-.25,-.24,-.22,-.20}) {
        for (auto backend:{phys::PopulationBackend::Portable,phys::PopulationBackend::Auto}) {
            auto p=scoped(); p.mu=mu; p.population_backend=backend;
            budget(p,256,8,2,true);
        }
#ifdef SIMANNEAL_HAVE_ACCELERATE
        auto p=scoped(); p.mu=mu; p.population_backend=phys::PopulationBackend::Accelerate;
        budget(p,256,8,2,true);
#endif
    }
    for (double mu:{-.33,-.19,std::nextafter(-.32,-1.),std::nextafter(-.20,0.)}) {
        auto p=scoped(); p.mu=mu;
        budget(p,10000,16,5,false);
    }
}

TEST_CASE("Every qualification condition gates scoped Auto budgets") {
    const std::vector<std::function<void(phys::SimParams &)>> exclusions={
        [](phys::SimParams &p){p.mu=-.19;},
        [](phys::SimParams &p){p.eps_r=10.1;},
        [](phys::SimParams &p){p.debye_length=.9;},
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
    budget(p,256,8,2,true);
}

TEST_CASE("Auto and explicit scoped budgets produce identical seeded results") {
    std::vector<phys::PopulationBackend> backends={
        phys::PopulationBackend::Portable,phys::PopulationBackend::Auto};
#ifdef SIMANNEAL_HAVE_ACCELERATE
    backends.push_back(phys::PopulationBackend::Accelerate);
#endif
    for (double mu:{-.32,-.25,-.20}) {
        for (auto backend:backends) {
            auto p=scoped(); p.deterministic_seed=true; p.random_seed=731;
            p.mu=mu; p.population_backend=backend;
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
    }
}

TEST_CASE("Wide Auto physics requires positive exclusion outside nominal physics") {
    for (double eps:{1.,5.6,10.}) for (double screening:{1.,5.,10.}) {
        auto p=scoped(2); p.eps_r=eps; p.debye_length=screening;
        p.db_locs[1].first=10000;
        budget(p,256,8,2,true);
        p.db_locs[1].first=.1;
        budget(p,eps==5.6 && screening==5 ? 256 : 10000,
               eps==5.6 && screening==5 ? 8 : 16,
               eps==5.6 && screening==5 ? 2 : 5,
               eps==5.6 && screening==5);
    }
    for (double eps:{std::nextafter(1.,0.),std::nextafter(10.,11.)}) {
        auto p=scoped(2); p.eps_r=eps; p.db_locs[1].first=10000;
        budget(p,10000,16,5,false);
    }
    for (double screening:{std::nextafter(1.,0.),std::nextafter(10.,11.)}) {
        auto p=scoped(2); p.debye_length=screening; p.db_locs[1].first=10000;
        budget(p,10000,16,5,false);
    }
}

TEST_CASE("Wide Auto follows the conservative prepared domain proof near its threshold") {
    auto p=scoped(2); p.eps_r=1; p.debye_length=10;
    // Locate the proof boundary using the same prepared matrix consumed by Auto.
    double lo=1, hi=100;
    for (int i=0;i<50;++i) {
        const double mid=(lo+hi)/2; p.db_locs[1].first=mid;
        phys::SimAnneal model(p);
        if (model.effectiveParams().final_domains[0]&4) lo=mid; else hi=mid;
    }
    p.db_locs[1].first=lo;
    budget(p,10000,16,5,false);
    p.db_locs[1].first=hi;
    budget(p,256,8,2,true);
}

TEST_CASE("Larger and wider Auto budgets match explicit seeded execution") {
    for (int count:{6,35,36,62}) for (double eps:{5.6,10.}) {
        auto p=scoped(count); p.eps_r=eps;
        if (eps!=5.6 && count!=6) for (int i=0;i<count;++i) p.db_locs[i].first=i*10000.;
        p.deterministic_seed=true; p.random_seed=731;
        phys::SuggestedResults expected;
        phys::mv::Options cleanup;
        {
            phys::SimAnneal model(p); REQUIRE(model.effectiveParams().budget_auto_selected);
            cleanup=model.effectiveParams().mv;
            model.invokeSimAnneal(); expected=model.suggestedResults();
        }
        p.anneal_cycles=count<=35?256:512;
        p.num_instances=count<=9?8:count<=35?64:128; p.hop_attempt_factor=2;
        p.mv=cleanup; // Compare the complete resolved Auto configuration.
        phys::SimAnneal model(p); model.invokeSimAnneal();
        const auto &actual=model.suggestedResults();
        REQUIRE(actual.size()==expected.size());
        for (std::size_t i=0;i<actual.size();++i) {
            REQUIRE(phys::SimAnneal::configToStr(actual[i].config)==phys::SimAnneal::configToStr(expected[i].config));
            REQUIRE(actual[i].system_energy==expected[i].system_energy);
        }
    }
}

TEST_CASE("Positive-possible large Auto selects cleanup only with all budgets automatic") {
    for (int count:{35,36,62,63}) {
        auto p=scoped(count); p.eps_r=1; p.debye_length=10; p.mu=-.4;
        for (int i=0;i<count;++i) p.db_locs[i].first=i*3.84;
        phys::SimAnneal solver(p);
        const auto &e=solver.effectiveParams();
        const bool selected=count>=36 && count<=62;
        REQUIRE(e.pkp_large_auto_selected==selected);
        REQUIRE(e.anneal_cycles==(selected?512:10000));
        REQUIRE(e.num_instances==128);
        REQUIRE(e.hop_attempt_factor==(selected?2:5));
        REQUIRE(e.mv.polish_mode==(selected?3:0));
        REQUIRE(e.mv.polish_top==(selected?4:0));
        REQUIRE(e.mv.vblock_k==(selected?8:0));
        REQUIRE(e.mv.vblock_top==(selected?16:0));
        REQUIRE(e.mv.vblock_scope==1);
        REQUIRE(e.mv.vblock_cache);
    }
    for (int field:{0,1,2}) {
        auto p=scoped(36); p.eps_r=1; p.mu=-.4;
        if (field==0) p.anneal_cycles=512;
        if (field==1) p.num_instances=128;
        if (field==2) p.hop_attempt_factor=2;
        phys::SimAnneal solver(p); const auto &e=solver.effectiveParams();
        REQUIRE_FALSE(e.pkp_large_auto_selected);
        REQUIRE(e.anneal_cycles==(field==0?512:10000));
        REQUIRE(e.hop_attempt_factor==(field==2?2:5));
        REQUIRE(e.mv.polish_mode==0);
        REQUIRE(e.mv.vblock_k==0);
    }
}

TEST_CASE("Large cleanup honors explicit off and preserves unqualified search configurations") {
    auto p=scoped(36); p.eps_r=1; p.mu=-.4;
    p.mv.polish_mode=0; p.mv.vblock_k=0; p.mv.vblock_cache=false; p.mv.vblock_top=8;
    {
        phys::SimAnneal solver(p); const auto &e=solver.effectiveParams();
        REQUIRE(e.pkp_large_auto_selected);
        REQUIRE(e.mv.polish_mode==0); REQUIRE(e.mv.vblock_k==0);
        REQUIRE_FALSE(e.mv.vblock_cache); REQUIRE(e.mv.vblock_top==8);
    }
    p=scoped(36); p.eps_r=1; p.mu=-.4; p.T_init=501;
    {
        phys::SimAnneal solver(p);
        REQUIRE_FALSE(solver.effectiveParams().pkp_large_auto_selected);
        REQUIRE(solver.effectiveParams().anneal_cycles==10000);
    }
    p=scoped(36); p.eps_r=1; p.mu=-.4;
    for (int i=0;i<36;++i) p.db_locs[i].first=i*10000.;
    phys::SimAnneal solver(p);
    REQUIRE_FALSE(solver.effectiveParams().pkp_large_auto_selected);
    REQUIRE(solver.effectiveParams().mv.vblock_k==0);
}

TEST_CASE("Large positive-possible Auto stays within the measured physics box") {
    const std::vector<std::function<void(phys::SimParams &)>> outside={
        [](phys::SimParams &p){p.mu=std::nextafter(-.5,-1.);},
        [](phys::SimParams &p){p.mu=std::nextafter(-.1,0.);},
        [](phys::SimParams &p){p.eps_r=std::nextafter(1.,0.);},
        [](phys::SimParams &p){p.eps_r=std::nextafter(10.,11.);},
        [](phys::SimParams &p){p.debye_length=std::nextafter(1.,0.);},
        [](phys::SimParams &p){p.debye_length=std::nextafter(10.,11.);}
    };
    for (const auto &change:outside) {
        auto p=scoped(36); p.mu=-.4; p.eps_r=1; change(p);
        phys::SimAnneal solver(p);
        REQUIRE_FALSE(solver.effectiveParams().pkp_large_auto_selected);
        REQUIRE(solver.effectiveParams().mv.polish_mode==0);
        REQUIRE(solver.effectiveParams().mv.vblock_k==0);
    }
    for (double mu:{-.5,-.1}) {
        auto p=scoped(36); p.mu=mu; p.eps_r=1;
        phys::SimAnneal solver(p);
        REQUIRE(solver.effectiveParams().pkp_large_auto_selected);
    }
}
