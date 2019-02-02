// @file:     main.cc
// @author:   Samuel
// @created:  2017.08.28
// @editted:  2018-06-13 - Robert
// @license:  Apache License 2.0
//
// @desc:     Main function for physics engine

#include "interface.h"
#include <unordered_map>
#include <iostream>
#include <string>

using namespace phys;

int main(int argc, char *argv[])
{
  std::cout << "Physeng invoked" << std::endl;
  std::string if_name, of_name;

  std::cout << "*** Argument Parsing ***" << std::endl;

  if(argc == 1){
    if_name = std::string("cooldbdesign.xml");
    of_name = std::string("cooloutput.xml");
  }
  else if(argc == 2){
    if_name = argv[1];
    of_name = std::string("cooloutput.xml");
  }
  else if(argc == 3){
    if_name = argv[1];
    of_name = argv[2];
  }
  else{
    std::cout << "More arguments than expected are encountered, aborting" << std::endl;
    return 0;
  }

  std::cout << "In File: " << if_name << std::endl;
  std::cout << "Out File: " << of_name << std::endl;

  std::cout << "Initiate SimAnneal interface" << std::endl;
  SimAnnealInterface interface(if_name, of_name);

  std::cout << "Invoke simulation" << std::endl;
  interface.runSimulation();

  std::cout << "Write simulation results" << std::endl;
  interface.writeSimResults();

  std::cout << "SimAnneal Complete" << std::endl;
}
