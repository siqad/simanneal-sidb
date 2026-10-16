#include "tests/catch2_wrapper.hpp"
#include "src/simanneal.h"

TEST_CASE("End polish defaults off and rejects invalid options") {
    phys::SimParams p;
    p.setDBLocs(std::vector<phys::EuclCoord>{{0,0},{7.68,0},{15.36,0}});
    {
        phys::SimAnneal solver(p);
        REQUIRE(solver.effectiveParams().mv.polish_mode==0);
        REQUIRE(solver.effectiveParams().mv.polish_top==0);
    }
    for (int mode:{-2,4}) {
        p.mv.polish_mode=mode;
        REQUIRE_THROWS_AS(phys::SimAnneal(p),std::invalid_argument);
    }
}

TEST_CASE("End polish validates results and is independent of worker count") {
    phys::SimParams p;
    p.setDBLocs(std::vector<phys::EuclCoord>{{0,0},{7.68,0},{15.36,0},{3.84,7.68},{11.52,7.68}});
    p.num_instances=8; p.anneal_cycles=32; p.hop_attempt_factor=2;
    p.deterministic_seed=true; p.random_seed=731;
    p.population_backend=phys::PopulationBackend::Portable;
    p.mv.polish_mode=3;
    for (int top:{0,4}) {
        p.mv.polish_top=top;
        phys::SuggestedResults expected;
        for (int workers:{1,2}) {
            p.num_workers=workers;
            phys::SimAnneal solver(p); solver.invokeSimAnneal();
            const auto results=solver.suggestedConfigResults(true);
            REQUIRE_FALSE(results.empty());
            REQUIRE(solver.searchStats().mv.polish_restarts>0);
            for (const auto &r:results) {
                FPType energy;
                REQUIRE(phys::SimAnneal::validatedEnergy(r.config,energy));
            }
            if (workers==1) expected=results;
            else {
                REQUIRE(results.size()==expected.size());
                for (std::size_t i=0;i<results.size();++i) {
                    REQUIRE(phys::SimAnneal::configToStr(results[i].config)==phys::SimAnneal::configToStr(expected[i].config));
                    REQUIRE(results[i].system_energy==expected[i].system_energy);
                }
            }
        }
    }
}
