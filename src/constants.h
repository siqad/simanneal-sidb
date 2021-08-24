// @file:     constants.h
// @author:   Samuel
// @created:  2021.07.30
// @license:  Apache License 2.0
//
// @desc:     Constants for SimAnneal

#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

#include "global.h"

namespace constants{
  // lattice
  const FPType lat_a = 3.84;  // lattice vector in x, angstroms (intra dimer row)
  const FPType lat_b = 7.68;  // lattice vector in y, angstroms (inter dimer row)
  const FPType lat_c = 2.25;  // dimer pair separation, angstroms

  // energy band diagram
  const FPType eta = 0.59;    // TODO enter true value; energy difference between (0/-) and (+/0) levels

  // physics
  const FPType Q0 = 1.602e-19;
  const FPType PI = 3.14159;
  const FPType EPS0 = 8.854e-12;
  const FPType Kb = 8.617e-5;
  const FPType ERFDB = 5e-10;

  // simulation

  // Allowed headroom in eV for physically invalid configurations to still be 
  // considered "probably valid", the validity will be re-determined during export.
  // Typical error is 1E-4 or lower, so this should be plenty enough headroom.
  const FPType POP_STABILITY_ERR = 1e-3;
  const FPType RECALC_STABILITY_ERR = 1e-6;
}

#endif