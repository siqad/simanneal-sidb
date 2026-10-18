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

TEST_CASE("Vblock validates options and defaults to an enabled suffix cache") {
    phys::SimParams p;
    p.setDBLocs(std::vector<phys::EuclCoord>{{0,0},{7.68,0},{15.36,0}});
    {
        phys::SimAnneal solver(p);
        REQUIRE(solver.effectiveParams().mv.vblock_k==0);
        REQUIRE(solver.effectiveParams().mv.vblock_top==0);
        REQUIRE(solver.effectiveParams().mv.vblock_dual==0);
        REQUIRE(solver.effectiveParams().mv.vblock_cache);
    }
    for (int k:{-2,5,12}) {
        p.mv.vblock_k=k;
        REQUIRE_THROWS_AS(phys::SimAnneal(p),std::invalid_argument);
    }
    p.mv.vblock_k=8; p.mv.vblock_scope=2;
    REQUIRE_THROWS_AS(phys::SimAnneal(p),std::invalid_argument);
    p.mv.vblock_scope=1; p.mv.vblock_top=-2;
    REQUIRE_THROWS_AS(phys::SimAnneal(p),std::invalid_argument);
}

TEST_CASE("Vblock top guard and suffix cache preserve validated results") {
    phys::SimParams p;
    std::vector<phys::EuclCoord> points;
    for (int i=0;i<10;++i) points.emplace_back((i%5)*3.84,(i/5)*7.68);
    p.setDBLocs(points);
    p.num_instances=16; p.anneal_cycles=32; p.hop_attempt_factor=2;
    p.deterministic_seed=true; p.random_seed=777; p.num_workers=2;
    p.population_backend=phys::PopulationBackend::Portable;
    p.mv.polish_mode=3; p.mv.polish_top=4;
    p.mv.vblock_k=8; p.mv.vblock_top=1;
    phys::SuggestedResults expected;
    for (bool cache:{false,true}) {
        p.mv.vblock_cache=cache;
        phys::SimAnneal solver(p); solver.invokeSimAnneal();
        REQUIRE(solver.searchStats().mv.vblock_endpoints<=1);
        REQUIRE(solver.searchStats().mv.vblock_endpoints>0);
        const auto results=solver.suggestedConfigResults(true);
        REQUIRE_FALSE(results.empty());
        for (const auto &r:results) {
            FPType energy;
            REQUIRE(phys::SimAnneal::validatedEnergy(r.config,energy));
        }
        if (cache) {
            const auto &e=solver.effectiveParams();
            phys::vblock::Geometry geometry(e.v_ij,e.v_ext,e.v_fc,e.mu,e.mv,e.mv_knn);
            phys::vblock::Charges q(results[0].config.begin(),results[0].config.end());
            phys::vblock::Memo memo;
            phys::vblock::Budget first(e.mv), second(e.mv);
            const auto a=geometry.descend(q,first,&memo);
            const auto b=geometry.descend(q,second,&memo);
            REQUIRE(first.blocks>0);
            REQUIRE(second.blocks==0);
            REQUIRE(a.q==b.q);
            REQUIRE(a.moves==b.moves);
        }
        if (!cache) expected=results;
        else {
            REQUIRE(results.size()==expected.size());
            for (std::size_t i=0;i<results.size();++i) {
                REQUIRE(phys::SimAnneal::configToStr(results[i].config)==phys::SimAnneal::configToStr(expected[i].config));
                REQUIRE(results[i].system_energy==expected[i].system_energy);
            }
        }
    }
    phys::vblock::Charges q{-1,0};
    REQUIRE(phys::vblock::stateKey(q,{0.,0.})!=phys::vblock::stateKey(q,{-0.,0.}));
}

TEST_CASE("Dual cleanup keeps valid results no worse than single k8 cleanup") {
    phys::SimParams p;
    std::vector<phys::EuclCoord> points;
    for (int i=0;i<14;++i) points.emplace_back((i%7)*3.84,(i/7)*7.68);
    p.setDBLocs(points);
    p.num_instances=16; p.anneal_cycles=32; p.hop_attempt_factor=2;
    p.deterministic_seed=true; p.random_seed=4242; p.num_workers=1;
    p.population_backend=phys::PopulationBackend::Portable;
    p.mv.polish_mode=3; p.mv.polish_top=4; p.mv.vblock_k=8; p.mv.vblock_top=0;
    phys::SuggestedResults single, dual;
    for (int mode:{0,1}) {
        p.mv.vblock_dual=mode;
        phys::SimAnneal solver(p); solver.invokeSimAnneal();
        REQUIRE(solver.effectiveParams().mv.vblock_dual==mode);
        REQUIRE(solver.searchStats().mv.vblock_endpoints>0);
        (mode ? dual : single)=solver.suggestedResults();
        for (const auto &r:solver.suggestedConfigResults(true)) {
            FPType energy;
            REQUIRE(phys::SimAnneal::validatedEnergy(r.config,energy));
        }
    }
    // Annealing is shared; each endpoint keeps the lower of its k8 and k12 descents.
    REQUIRE(single.size()==dual.size());
    for (std::size_t i=0;i<single.size();++i)
        if (single[i].initialized && dual[i].initialized)
            REQUIRE(dual[i].system_energy<=single[i].system_energy+1e-12);
    // A k12 geometry descends exactly with blocks larger than eight sites.
    p.mv.vblock_dual=1;
    phys::SimAnneal solver(p);
    const auto &e=solver.effectiveParams();
    auto twelve=e.mv; twelve.vblock_dual=0; twelve.vblock_k=12;
    phys::vblock::Geometry geometry(e.v_ij,e.v_ext,e.v_fc,e.mu,twelve,e.mv_knn);
    phys::vblock::Charges q(points.size(),-1);
    phys::vblock::Budget budget(twelve);
    const auto d=geometry.descend(q,budget);
    REQUIRE(d.q.size()==points.size());
}
