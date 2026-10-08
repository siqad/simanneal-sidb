#include "tests/catch2_wrapper.hpp"
#include "src/simanneal.h"
#include "src/repair_class_scan.h"
#include "src/repair_hop_bound.h"
#include <cstring>
#include <limits>

namespace {
struct OriginalMove { int from=-1, to=-1; double delta=-1e-6; };
OriginalMove original(const std::vector<int> &charges,
                      const std::vector<double> &potential,
                      const phys::ublas::matrix<double> &matrix) {
    OriginalMove best;
    for (int i=0; i<int(charges.size()); ++i)
        for (int j=0; j<int(charges.size()); ++j)
            if (charges[i]<charges[j]) {
                const double delta=-potential[i]+potential[j]-matrix(i,j);
                if (delta<best.delta) { best.from=i; best.to=j; best.delta=delta; }
            }
    return best;
}
phys::SimParams geometryParams(int count, int workers, bool coincident=false) {
    phys::SimParams params;
    std::vector<phys::EuclCoord> locations;
    for (int i=0; i<count; ++i)
        locations.emplace_back((i%31)*3.84, (i/31)*7.68);
    if (coincident) locations[1]=locations[0];
    params.setDBLocs(locations);
    params.setFixedCharges({phys::EuclCoord3d(-10,4,3)}, {1}, {5.6}, {5.0});
    for (int i=0; i<count; ++i) params.v_ext[i]=(i%7)*.001;
    params.num_workers=workers; params.num_instances=8;
    params.anneal_cycles=4;
    if (!coincident) params.hop_selection=phys::LocalDistanceHop;
    return params;
}
void requireMatrixBits(const phys::ublas::matrix<double> &a,
                       const phys::ublas::matrix<double> &b) {
    REQUIRE(a.size1()==b.size1()); REQUIRE(a.size2()==b.size2());
    REQUIRE(std::memcmp(&a.data()[0], &b.data()[0],
                        a.size1()*a.size2()*sizeof(double))==0);
}
}

TEST_CASE("Default repair target scans preserve the original exact move") {
    constexpr int count=5;
    phys::ublas::matrix<double> matrix(count,count);
    simanneal_class_scan::Workspace classWork;
    simanneal_pair_bound::Workspace boundWork;
    for (int fixture=0; fixture<5; ++fixture) {
        std::vector<double> potential(count);
        for (int i=0; i<count; ++i) {
            potential[i]=fixture==0 ? 0 : (i%3-1)*.125;
            for (int j=0; j<count; ++j)
                matrix(i,j)=fixture==0 ? .125 : ((i+j)%4)*.03125;
        }
        if (fixture==2) potential[1]=std::numeric_limits<double>::infinity();
        if (fixture==3) potential[3]=std::numeric_limits<double>::quiet_NaN();
        if (fixture==4) matrix(1,4)=std::numeric_limits<double>::infinity();
        simanneal_pair_bound::Geometry geometry(count,matrix);
        for (int encoded=0; encoded<243; ++encoded) {
            std::vector<int> charges(count);
            int state=encoded;
            for (auto &charge:charges) { charge=state%3-1; state/=3; }
            const auto expected=original(charges,potential,matrix);
            const auto scan=simanneal_class_scan::select(classWork,charges,potential,matrix,1e-6);
            const auto bound=simanneal_pair_bound::select(geometry,boundWork,charges,potential,matrix,1e-6);
            INFO("fixture=" << fixture << " state=" << encoded);
            REQUIRE(scan.from==expected.from); REQUIRE(scan.to==expected.to);
            REQUIRE(scan.delta==expected.delta);
            REQUIRE(bound.from==expected.from); REQUIRE(bound.to==expected.to);
            REQUIRE(bound.delta==expected.delta);
        }
    }
}

TEST_CASE("Automatic geometry preserves serial matrix bits and fixed fields") {
    for (int count:{511,512,651}) {
        auto input=geometryParams(count,1);
        phys::SimParams serial;
        { phys::SimAnneal solver(input); serial=solver.effectiveParams(); }
        input.num_workers=0;
        phys::SimAnneal automatic(input);
        const auto &actual=automatic.effectiveParams();
        requireMatrixBits(actual.db_r,serial.db_r);
        requireMatrixBits(actual.v_ij,serial.v_ij);
        REQUIRE(actual.population_finite_matrix==serial.population_finite_matrix);
        REQUIRE(actual.population_finite_matrix);
        for (int i=0; i<count; ++i) {
            REQUIRE(actual.v_ext[i]==serial.v_ext[i]);
            REQUIRE(actual.v_fc[i]==serial.v_fc[i]);
        }
        REQUIRE(actual.hop_neighborhood.neighborsPerSite()==serial.hop_neighborhood.neighborsPerSite());
        std::vector<int> charges(count);
        for (int i=0; i<count; ++i) charges[i]=i%3-1;
        for (int source=0; source<count; ++source)
            for (double random:{0.0,.125,.5,.999999})
                REQUIRE(actual.hop_neighborhood.select(source,charges,random)==
                        serial.hop_neighborhood.select(source,charges,random));
    }
}

TEST_CASE("Automatic geometry retains nonfinite matrix detection") {
    phys::SimParams serial;
    auto input=geometryParams(512,1,true);
    { phys::SimAnneal solver(input); serial=solver.effectiveParams(); }
    input.num_workers=0;
    phys::SimAnneal automatic(input);
    REQUIRE_FALSE(serial.population_finite_matrix);
    REQUIRE_FALSE(automatic.effectiveParams().population_finite_matrix);
    requireMatrixBits(automatic.effectiveParams().v_ij,serial.v_ij);
}
