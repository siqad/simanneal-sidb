#include "tests/catch2_wrapper.hpp"
#include "src/simanneal.h"
#include "src/affinity_workers.h"

TEST_CASE("Restart scheduling and optional history preserve seeded results") {
    phys::SimParams sp;
    sp.setDBLocs(std::vector<phys::EuclCoord>{{0,0},{3.84,0},{7.68,0},
        {0,7.68},{3.84,7.68},{7.68,7.68}});
    sp.v_ext.clear(); sp.v_fc.clear();
    sp.num_instances=12; sp.anneal_cycles=64; sp.result_queue_factor=.5;
    sp.hop_attempt_factor=5; // Hold the search budget fixed across history modes.
    sp.deterministic_seed=true; sp.random_seed=731;
    phys::SuggestedResults expected;
    for (bool history : {false,true}) {
        for (int workers : {1,4,12,0}) {
            sp.record_history=history; sp.num_workers=workers;
            phys::SimAnneal master(sp);
            REQUIRE(master.effectiveParams().num_workers ==
                simanneal_affinity::workerCount(workers, sp.num_instances));
            master.invokeSimAnneal();
            const auto &results=master.suggestedResults();
            REQUIRE(results.size()==12);
            if (expected.empty()) expected=results;
            for (std::size_t i=0;i<results.size();++i) {
                REQUIRE(results[i].initialized);
                REQUIRE(phys::SimAnneal::configToStr(results[i].config)==
                    phys::SimAnneal::configToStr(expected[i].config));
                REQUIRE(results[i].system_energy==expected[i].system_energy);
                REQUIRE(master.chargeResults()[i].size()==(history ? 32 : 0));
                REQUIRE(master.energyResults()[i].size()==(history ? 32 : 0));
            }
            master.invokeSimAnneal();
            REQUIRE(master.suggestedResults()[0].system_energy==expected[0].system_energy);
        }
    }
}

TEST_CASE("Explicit restart seeds have a strict MT19937 range") {
    REQUIRE(phys::SimParams::parseRandomSeed("0")==0);
    REQUIRE(phys::SimParams::parseRandomSeed("4294967295")==UINT32_MAX);
    for (const auto &value : {"", "-1", "+1", "1junk", " 1", "4294967296"})
        REQUIRE_THROWS_AS(phys::SimParams::parseRandomSeed(value),std::invalid_argument);
    phys::SimParams sp;
    sp.setDBLocs(std::vector<phys::EuclCoord>{{0,0}});
    sp.v_ext.clear(); sp.v_fc.clear();
    sp.deterministic_seed=true; sp.random_seed=UINT64_C(4294967296);
    REQUIRE_THROWS_AS(phys::SimAnneal(sp),std::invalid_argument);
}

namespace ublas = boost::numeric::ublas;

TEST_CASE( "OR_mod 00 test" ) {
    auto sim_params = phys::SimParams();
    // 100 lat vec
    sim_params.lat_vec.a1 = std::pair<FPType, FPType>(3.84, 0);
    sim_params.lat_vec.a2 = std::pair<FPType, FPType>(0, 7.68);
    sim_params.lat_vec.atoms = std::vector<std::pair<float, float>>({
        {0, 0},
        {0, 2.25}
    });
    sim_params.setDBLocs({
        {-4,-2, 0},
        {-2,-1, 0},
        { 2,-1, 0},
        { 4,-2, 0},
        { 0, 1, 0},
        { 0, 2, 1},
        { 0, 4, 1}
    }, sim_params.lat_vec);
    phys::SimAnneal annealer(sim_params);
    annealer.invokeSimAnneal();
    auto results = annealer.suggestedConfigResults(true);
    
    bool gs_result_found = false;
    std::vector<FPType> expected_result({-1, 0, 0, -1, -1, 0, -1});
    for (const auto &result : results) {
        gs_result_found = std::equal(
            result.config.cbegin(),
            result.config.cend(),
            expected_result.cbegin()
        );
        if (gs_result_found) {
            break;
        }
    }

    REQUIRE(gs_result_found);
}
TEST_CASE("Local target sampler conditions on neutral sites") {
    ublas::matrix<double> d(4,4);
    for (int i=0; i<4; ++i) for (int j=0; j<4; ++j) d(i,j)=std::abs(i-j)*1e-9;
    phys::HopNeighborhood hood;
    hood.build(d, 4, 2, 1.0, true);
    std::vector<int> charges{-1, 0, 0, 0};
    REQUIRE(hood.neighborsPerSite() == 2);
    int near = 0;
    for (int i=0; i<10000; ++i) {
        int target = hood.select(0, charges, (i+0.5)/10000.0);
        REQUIRE((target == 1 || target == 2));
        near += target == 1;
    }
    REQUIRE(near/10000.0 == Approx(1.0/(1.0+std::exp(-1.0))).margin(0.0001));
    charges[1] = 1; // positive charges are not neutral targets either
    REQUIRE(hood.select(0, charges, 0) == 2);
    charges[2] = -1;
    REQUIRE(hood.select(0, charges, 0.5) == -1); // global fallback even though site 3 is neutral
    hood.build(d, 4, 16, 1.0, false);
    REQUIRE(hood.neighborsPerSite() == 3);
    REQUIRE(hood.select(0, charges, 0.5) == 3);
    REQUIRE_THROWS_AS(hood.build(d,4,0,1,true), std::invalid_argument);
    REQUIRE_THROWS_AS(hood.build(d,4,1,0,true), std::invalid_argument);
    hood.build(d, 1, 16, 1.0, true);
    REQUIRE(hood.select(0, charges, 0.5) == -1);
}

TEST_CASE("Radius sampler tracks neutral targets without scanning the radius") {
    ublas::matrix<double> d(3,3);
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) d(i,j)=std::abs(i-j)*1e-9;
    phys::HopNeighborhood hood;
    hood.buildRadius(d, 3, 1.01);
    std::vector<int> charges{-1, 0, 0};
    phys::RadiusEligibleCache cache;
    REQUIRE(hood.selectRadius(0, charges, 0.5, cache) == 1);
    charges[1] = 1;
    hood.setNeutral(1, false, cache);
    REQUIRE(hood.selectRadius(0, charges, 0.5, cache) == -2);
    charges[1] = 0;
    hood.setNeutral(1, true, cache);
    REQUIRE(hood.selectRadius(0, charges, 0.5, cache) == 1);
}

TEST_CASE("Local hops preserve energy and occupation bookkeeping") {
    for (auto policy : {phys::UniformHop, phys::LocalUniformHop, phys::LocalDistanceHop, phys::LocalRadiusHop}) {
        for (double global : {0.0, 0.2, 1.0}) {
            phys::SimParams sp;
            sp.setDBLocs(std::vector<phys::EuclCoord>{{0,0},{3.84,0},{7.68,0},{0,7.68},{3.84,7.68},{7.68,7.68}});
            sp.v_ext.clear(); sp.v_fc.clear();
            sp.mu=-0.05; // exercise positive as well as negative charges
            sp.num_instances=1; sp.anneal_cycles=200; sp.result_queue_factor=1; sp.record_history=true;
            sp.hop_selection=policy; sp.hop_global_probability=global; sp.hop_neighbors=2;
            sp.hop_radius_nm=4.0;
            phys::SimAnneal master(sp);
            phys::SimAnnealThread worker(0,12345);
            worker.run();
            REQUIRE(master.chargeResults()[0].size()==200);
            for (const auto &result : master.chargeResults()[0]) {
                REQUIRE(result.system_energy == Approx(phys::SimAnneal::systemEnergy(result.config)).margin(1e-10));
                for (int charge : result.config) REQUIRE((charge>=-1 && charge<=1));
            }
        }
    }
}


TEST_CASE("An all-global local policy preserves the original random trajectory") {
    phys::SimParams sp;
    sp.setDBLocs(std::vector<phys::EuclCoord>{{0,0},{7.68,0},{15.36,0},{0,15.36},{7.68,15.36},{15.36,15.36}});
    sp.v_ext.clear(); sp.v_fc.clear();
    sp.num_instances=1; sp.anneal_cycles=100; sp.result_queue_factor=1; sp.record_history=true;
    phys::ThreadChargeResults original_history;
    {
        phys::SimAnneal original(sp);
        phys::SimAnnealThread a(0,998);
        a.run();
        original_history=original.chargeResults()[0];
    }
    sp.hop_selection=phys::LocalDistanceHop; sp.hop_global_probability=1;
    phys::ThreadChargeResults local_history;
    {
        phys::SimAnneal local(sp);
        phys::SimAnnealThread b(0,998);
        b.run();
        local_history=local.chargeResults()[0];
    }
    REQUIRE(original_history.size()==100);
    REQUIRE(original_history.size()==local_history.size());
    for (std::size_t i=0;i<original_history.size();++i) {
        REQUIRE(phys::SimAnneal::configToStr(original_history[i].config)==phys::SimAnneal::configToStr(local_history[i].config));
        REQUIRE(original_history[i].system_energy==local_history[i].system_energy);
    }
    sp.hop_global_probability=-0.1;
    REQUIRE_THROWS_AS(phys::SimAnneal(sp),std::invalid_argument);
}

TEST_CASE("Population validity reuse preserves baseline histories after changes",
          "[history][validity-cache]") {
    // These digests cover every charge and population-validity flag from the
    // uncached a1a3ff5 baseline. They exclude floating-point energy bits.
    const std::uint64_t expected[] = {
        UINT64_C(10700360247583351287), UINT64_C(13890666197318041969),
        UINT64_C(1710319441480308780), UINT64_C(7990926584911492685)};
    for (int variant=0; variant<4; ++variant) {
        phys::SimParams sp;
        sp.setDBLocs(std::vector<phys::EuclCoord>{{0,0},{3.84,0},{7.68,0},
            {0,7.68},{3.84,7.68},{7.68,7.68}});
        sp.v_ext.clear(); sp.v_fc.clear();
        sp.num_instances=1; sp.num_workers=1; sp.anneal_cycles=256;
        sp.result_queue_factor=1; sp.record_history=true;
        sp.population_backend=phys::PopulationBackend::Portable;
        sp.random_backend=phys::RandomBackend::MT;
        sp.mu=variant%2 ? -.05 : -.32;
        sp.hop_attempt_factor=variant<2 ? 0 : 5;
        sp.strategic_v_freeze_reset=true;
        sp.phys_validity_check_cycles=3;
        phys::SimAnneal model(sp);
        phys::SimAnnealThread worker(0,12345);
        worker.run();
        const auto &history=model.chargeResults()[0];
        REQUIRE(history.size()==256);
        std::uint64_t hash=UINT64_C(14695981039346656037);
        unsigned changed=0, unchanged=0;
        for (unsigned i=0; i<history.size(); ++i) {
            const auto &result=history[i];
            for (int charge : result.config) {
                hash ^= static_cast<unsigned char>(charge+1);
                hash *= UINT64_C(1099511628211);
            }
            hash ^= result.pop_likely_stable;
            hash *= UINT64_C(1099511628211);
            REQUIRE(result.system_energy ==
                Approx(phys::SimAnneal::systemEnergy(result.config)).margin(1e-10));
            REQUIRE(model.energyResults()[0][i]==result.system_energy);
            if (i) {
                bool same=true;
                for (unsigned j=0; j<result.config.size(); ++j)
                    same &= result.config[j]==history[i-1].config[j];
                if (same) ++unchanged;
                else ++changed;
            }
        }
        // Each arm exercises reuse and invalidation. The last two enable hops.
        REQUIRE(changed>0);
        REQUIRE(unchanged>0);
        REQUIRE(hash==expected[variant]);
    }
}

TEST_CASE("Radius MT histories refresh validity and repeat independently",
          "[history][validity-cache]") {
    phys::SimParams sp;
    sp.setDBLocs(std::vector<phys::EuclCoord>{{0,0},{3.84,0},{7.68,0},
        {0,7.68},{3.84,7.68},{7.68,7.68}});
    sp.v_ext.clear(); sp.v_fc.clear(); sp.mu=-.05;
    sp.num_instances=1; sp.num_workers=1; sp.anneal_cycles=256;
    sp.result_queue_factor=1; sp.record_history=true;
    sp.random_backend=phys::RandomBackend::MT;
    sp.population_backend=phys::PopulationBackend::Portable;
    sp.hop_selection=phys::LocalRadiusHop; sp.hop_radius_nm=4.;
    sp.hop_global_probability=0.; sp.strategic_v_freeze_reset=true;
    sp.phys_validity_check_cycles=3;
    phys::ThreadChargeResults first;
    for (int run=0; run<2; ++run) {
        phys::SimAnneal model(sp);
        phys::SimAnnealThread worker(0,12345);
        worker.run();
        const auto &history=model.chargeResults()[0];
        const auto &physical=model.effectiveParams();
        REQUIRE(history.size()==256);
        unsigned changes=0, unchanged=0;
        for (unsigned cycle=0; cycle<history.size(); ++cycle) {
            const auto &entry=history[cycle];
            bool valid=true;
            for (unsigned i=0; i<entry.config.size(); ++i) {
                double potential=-(physical.v_ext[i]+physical.v_fc[i]);
                for (unsigned j=0; j<entry.config.size(); ++j)
                    potential-=physical.v_ij(i,j)*entry.config[j];
                const double lower=potential+physical.mu;
                const double upper=potential+(physical.mu-constants::eta);
                const double eps=constants::POP_STABILITY_ERR;
                const int charge=entry.config[i];
                valid &= (charge==-1 && lower<eps) ||
                    (charge==1 && upper>-eps) ||
                    (charge==0 && lower>-eps && upper<eps);
            }
            REQUIRE(entry.pop_likely_stable==valid);
            REQUIRE(entry.system_energy==
                Approx(phys::SimAnneal::systemEnergy(entry.config)).margin(1e-10));
            if (cycle) {
                const bool same=phys::SimAnneal::configToStr(entry.config)==
                    phys::SimAnneal::configToStr(history[cycle-1].config);
                if (same) ++unchanged;
                else ++changes;
            }
            if (run) {
                REQUIRE(phys::SimAnneal::configToStr(entry.config)==
                    phys::SimAnneal::configToStr(first[cycle].config));
                REQUIRE(entry.pop_likely_stable==first[cycle].pop_likely_stable);
                REQUIRE(entry.system_energy==first[cycle].system_energy);
            }
        }
        REQUIRE(changes>0);
        REQUIRE(unchanged>0);
        if (!run) first=history;
    }
}
