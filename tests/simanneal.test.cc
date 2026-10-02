#include "tests/catch2_wrapper.hpp"
#include "src/simanneal.h"

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
    auto annealer = phys::SimAnneal(sim_params);
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
            sp.num_instances=1; sp.anneal_cycles=200; sp.result_queue_factor=1;
            sp.hop_selection=policy; sp.hop_global_probability=global; sp.hop_neighbors=2;
            sp.hop_radius_nm=4.0;
            phys::SimAnneal master(sp);
            phys::SimAnnealThread worker(0,12345);
            worker.run();
            for (const auto &result : worker.db_charges) {
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
    sp.num_instances=1; sp.anneal_cycles=100; sp.result_queue_factor=1;
    phys::SimAnneal original(sp);
    phys::SimAnnealThread a(0,998);
    a.run();
    sp.hop_selection=phys::LocalDistanceHop; sp.hop_global_probability=1;
    phys::SimAnneal local(sp);
    phys::SimAnnealThread b(0,998);
    b.run();
    REQUIRE(a.db_charges.size()==b.db_charges.size());
    for (std::size_t i=0;i<a.db_charges.size();++i) {
        REQUIRE(phys::SimAnneal::configToStr(a.db_charges[i].config)==phys::SimAnneal::configToStr(b.db_charges[i].config));
        REQUIRE(a.db_charges[i].system_energy==b.db_charges[i].system_energy);
    }
    sp.hop_global_probability=-0.1;
    REQUIRE_THROWS_AS(phys::SimAnneal(sp),std::invalid_argument);
}
