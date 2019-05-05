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

global::TimeKeeper *global::TimeKeeper::time_keeper=nullptr;
int global::log_level = Logger::WRN;

using namespace phys;

int main(int argc, char *argv[])
{
  std::cout << "Physeng invoked" << std::endl;
  std::string if_name, of_name;
  std::vector<std::string> cml_args;

  std::cout << "*** Argument Parsing ***" << std::endl;

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
  bool only_suggested_gs=false;
  unsigned long cml_i=0;
  while (cml_i < cml_args.size()) {
    if (cml_args[cml_i] == "--only-suggested-gs") {
      // each SimAnneal instance only returns one configuration
      std::cout << "--only-suggested-gs: Only returning suggested ground state from each instance." << std::endl;
      only_suggested_gs = true;
    } else if (cml_args[cml_i] == "--debug") {
      // show additional debug information
      std::cout << "--debug: Showing additional outputs." << std::endl;
      global::log_level = Logger::DBG;
    } else {
      throw "Unrecognized command-line argument: " + cml_args[cml_i];
    }
    cml_i++;
  }

  Logger log(global::log_level);

  global::TimeKeeper *tk = global::TimeKeeper::instance();
  global::Stopwatch *sw_simulation = tk->createStopwatch("Total Simulation");

  log.echo() << "In File: " << if_name << std::endl;
  log.echo() << "Out File: " << of_name << std::endl;

  log.echo() << "*** Initiate SimAnneal interface ***" << std::endl;
  SimAnnealInterface interface(if_name, of_name);

  log.echo() << "*** Invoke simulation ***" << std::endl;
  sw_simulation->start();
  interface.runSimulation();
  sw_simulation->end();

  log.echo() << "*** Write simulation results ***" << std::endl;
  interface.writeSimResults(only_suggested_gs);

  log.echo() << "*** SimAnneal Complete ***" << std::endl;

  tk->printAllStopwatches();
  delete tk;
}
