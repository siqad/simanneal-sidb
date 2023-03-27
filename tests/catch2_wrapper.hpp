/**
 * @file catch2_wrapper.cpp
 *
 * @brief Includes the Catch2 unit testing framework plus some handy macros.
 *
 * @copyright Apache License 2.0
 */

#ifndef TESTS_CATCH2_WRAPPER_H_
#define TESTS_CATCH2_WRAPPER_H_

#include <iostream>

#include "libs/catch2/catch.hpp"

/**
 * Macro for 
 */
#define SQTEST_SUPPRESS_CERR(test)                                             \
  std::cerr.setstate(std::ios_base::failbit);                                  \
  test;                                                                        \
  std::cerr.clear();

#endif
