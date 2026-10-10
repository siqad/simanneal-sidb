// @file:     main.cc
// @author:   Samuel
// @created:  2017.08.28
// @editted:  2019.02.01
// @license:  Apache License 2.0
//
// @desc:     Main function for physics engine

#include "global.h"
#include "interface.h"
#include <unordered_map>
#include <iostream>
#include <string>

//saglobal::TimeKeeper *saglobal::TimeKeeper::time_keeper=nullptr;
//int saglobal::log_level = Logger::WRN;

using namespace phys;

static int run(int argc, char *argv[])
{
  std::string if_name, of_name, ext_pots_name;
  std::vector<std::string> cml_args;


  if (argc < 3) {
    throw "Less arguments than excepted.";
  } else {
    // argv[0] is the binary
    if_name = argv[1];
    of_name = argv[2];

    // store the rest of the arguments
    for (int i=3; i<argc; i++) {
      cml_args.push_back(argv[i]);
    }
  }

  // parse additional arguments
  int ext_pots_step=0;
  bool only_suggested_gs=false;
  bool qubo_energy=false;
  bool verbose=false;
  unsigned long cml_i=0;
  while (cml_i < cml_args.size()) {
    if (cml_args[cml_i] == "--ext-pots") {
      if (cml_i + 1 >= cml_args.size()) throw std::invalid_argument("--ext-pots requires a path");
      ext_pots_name = cml_args[++cml_i];
    } else if (cml_args[cml_i] == "--ext-pots-step") {
      if (cml_i + 1 >= cml_args.size()) throw std::invalid_argument("--ext-pots-step requires an integer");
      ext_pots_step = stoi(cml_args[++cml_i]);
    } else if (cml_args[cml_i] == "--only-suggested-gs") {
      // each SimAnneal instance only returns one configuration
      only_suggested_gs = true;
    } else if (cml_args[cml_i] == "--debug") {
      // show additional debug information
      std::cout << "--debug: Showing additional outputs." << std::endl;
      verbose = true;
      saglobal::log_level = Logger::DBG;
    /* NOTE: current QUBO implementation is incomplete
    } else if (cml_args[cml_i] == "--qubo") {
      // export energy value in QUBO formulation
      std::cout << "--qubo: Using QUBO energy equation for export." << std::endl;
      qubo_energy = true;
    */
    } else {
      throw "Unrecognized command-line argument: " + cml_args[cml_i];
    }
    cml_i++;
  }

  Logger log(saglobal::log_level);

  saglobal::TimeKeeper *tk = saglobal::TimeKeeper::instance();
  saglobal::Stopwatch *sw_simulation = tk->createStopwatch("Total Simulation");

  log.debug() << "In File: " << if_name << std::endl;
  log.debug() << "Out File: " << of_name << std::endl;
  log.debug() << "External Potentials File: " << ext_pots_name << std::endl;

  log.debug() << "\n*** Initiate SimAnneal interface ***" << std::endl;
  SimAnnealInterface interface(if_name, of_name, ext_pots_name, ext_pots_step, verbose);

  log.debug() << "\n*** Read Simulation parameters ***" << std::endl;
  SimParams sparams = interface.loadSimParams();

  log.debug() << "\n*** Invoke simulation ***" << std::endl;
  sw_simulation->start();
  interface.runSimulation(std::move(sparams));
  sw_simulation->end();

  log.debug() << "\n*** Write simulation results ***" << std::endl;
  interface.writeSimResults(only_suggested_gs, qubo_energy);

  log.debug() << "\n*** SimAnneal Complete ***" << std::endl;

  if (verbose) tk->printAllStopwatches();

  delete tk;
  return 0;
}

int main(int argc, char **argv) {
  try { return run(argc, argv); }
  catch (const std::exception &error) { std::cerr << error.what() << '\n'; }
  catch (const std::string &error) { std::cerr << error << '\n'; }
  catch (const char *error) { std::cerr << error << '\n'; }
  return 1;
}
