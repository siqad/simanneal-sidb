// @editted:    2020-01-19
// @license:    Apache License 2.0
//
// @desc:       Python wrapper for SimAnneal

%module simanneal
%include <std_streambuf.i>
%include <std_sstream.i>
%include <std_iostream.i>
%include <std_deque.i>
%include <std_pair.i>
%include <std_vector.i>
%include <std_string.i>
%include <std_map.i>
%include <exception.i>

namespace boost {
    namespace numeric {
        namespace ublas {
        }
    }
}

namespace saglobal {
    extern int log_level;
}

extern int saglobal::log_level;

%{
#include "logger.h"
#include "global.h"
#include "simanneal.h"
%}

%include "logger.h"
%include "global.h"
%include "simanneal.h"

namespace std {
    %template(DoublePair) pair<double, double>;
    %template(DoublePairVector) vector< pair <double, double> >;
    %template(FloatPair) pair<float, float>;
    %template(FloatPairVector) vector< pair <float, float> >;
    %template(FloatVector) vector<float>;
    %template(IntVector) vector<int>;
    %template(IntVectorVector) vector< vector<int> >;
    %template(StringPair) pair<string, string>;
    %template(StringPairVector) vector< pair<string, string> >;
    %template(StringVector) vector<string>;
    %template(StringVector2D) vector< vector<string> >;
    %template(StringMap) map< string, string >;

    // Iterable container for suggested ground state results returned by SimAnneal
    %template(ConfigVector) vector< pair< vector<int>, float > >;
}

%{
#define SWIG_FILE_WITH_INIT
#define SWIG_PYTHON_2_UNICODE
%}

%extend phys::SimParams {
    
    void phys::SimParams::pySetVExt(std::vector<float> s_vec) {
        boost::numeric::ublas::vector<float> u_vec(s_vec.size());
        for (unsigned int i=0; i<s_vec.size(); i++)
            u_vec[i] = s_vec[i];
        $self->v_ext = u_vec;
    }
    
    %pythoncode{
        def set_db_locs(self, db_locs):
            dbs = FloatPairVector(db_locs)
            self.setDBLocs(dbs)

        def set_v_ext(self, v_ext):
            self.pySetVExt(FloatVector(v_ext))
    }
}

%extend phys::SimAnneal {

    // Convert vector of ChargeConfigResult to a vector of pair that has been 
    // specifically defined above as ConfigVector such that SWIG knows how to 
    // generate a suitable container for the results.
    std::vector<std::pair<std::vector<int>, float>> phys::SimAnneal::pySuggestedResults() {
        std::vector<std::pair<std::vector<int>, float>> out_results;
        for (auto result : self->suggestedConfigResults()) {
            std::vector<int> conf;
            for (int chg : result.config)
                conf.push_back(chg);
            out_results.push_back(std::make_pair(conf, result.system_energy));
        }
        return out_results;
    }

    %pythoncode{
        from collections import namedtuple
        ChargeResult = namedtuple('ChargeResult', ['config', 'energy'])

        def suggested_gs_results(self):
            configs = []
            for conf in self.pySuggestedResults():
                chg_cfg = [chg for chg in conf[0]]
                configs.append(self.ChargeResult(chg_cfg, conf[1]))
            return configs
    }
}
