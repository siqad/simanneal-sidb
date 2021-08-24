// @editted:    2020-01-19
// @license:    Apache License 2.0
//
// @desc:       Python wrapper for SimAnneal

%module simanneal

namespace boost {
    namespace numeric {
        namespace ublas {
        }
    }
}

%{
#include "simanneal_cuda.h"
%}

%include "simanneal_cuda.h"

%{
#define SWIG_FILE_WITH_INIT
#define SWIG_PYTHON_2_UNICODE
%}
