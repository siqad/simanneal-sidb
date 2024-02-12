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