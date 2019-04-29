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

using namespace phys;

int main(int argc, char *argv[])
{
  std::cout << "Physeng invoked" << std::endl;
  std::string if_name, of_name;
  bool only_suggested_gs=false;

  std::cout << "*** Argument Parsing ***" << std::endl;

  if(argc == 1){
    if_name = std::string("cooldbdesign.xml");
    of_name = std::string("cooloutput.xml");
  } else if (argc == 2) {
    if_name = argv[1];
    of_name = std::string("cooloutput.xml");
  } else if (argc == 3) {
    if_name = argv[1];
    of_name = argv[2];
  } else if (argc == 4) {
    if_name = argv[1];
    of_name = argv[2];
    if (strcmp(argv[3], "--only-suggested-gs") == 0) {
      std::cout << "Only returning suggested ground state from each instance." << std::endl;
      only_suggested_gs = true;
    } else {
      std::cout << "Last argument not recognized, aborting." << std::endl;
      return 1;
    }
  } else {
    std::cout << "More arguments than expected are encountered, aborting." << std::endl;
    return 2;
  }

  global::TimeKeeper *tk = global::TimeKeeper::instance();
  global::Stopwatch *sw_simulation = tk->createStopwatch("Total Simulation");

  std::cout << "In File: " << if_name << std::endl;
  std::cout << "Out File: " << of_name << std::endl;

  std::cout << "Initiate SimAnneal interface" << std::endl;
  SimAnnealInterface interface(if_name, of_name);

  std::cout << "Invoke simulation" << std::endl;
  sw_simulation->start();
  interface.runSimulation();
  sw_simulation->end();

  std::cout << "Write simulation results" << std::endl;
  interface.writeSimResults(only_suggested_gs);

  std::cout << "SimAnneal Complete" << std::endl;

  tk->printAllStopwatches();
  delete tk;
}
