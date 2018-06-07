// @file:     main.cc
// @author:   Samuel
// @created:  2017.08.28
// @editted:  2017.08.28 - Samuel
// @license:  GNU LGPL v3
//
// @desc:     Main function for physics engine

#include "sim_anneal.h"
#include <iostream>
#include <string>

using namespace phys;

void exportData();
void getLocation();

// temporary main function for testing the xml parsing functionality
int main(int argc, char *argv[])
{
  std::cout << "Physeng invoked" << std::endl;
  std::string if_name, of_name;

  std::cout << "*** Argument Parsing ***" << std::endl;

  // for now, only support two arguments: input and output files
  // TODO flags: -i input_path -o output_path
  // maybe make a struct to contain program options in case of more input options
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

  std::cout << std::endl << "*** Constructing Problem ***" << std::endl;
  SimAnneal sim_anneal(if_name, of_name);

  std::cout << std::endl << "*** Run Simulation ***" << std::endl;

  sqconn = new SiQADConnector(std::string("SimAnneal"), i_path, o_path);

  if(!sim_anneal.runSim()) {
    std::cout << "Simulation failed, aborting" << std::endl;
    return 0;
  }

  std::cout << std::endl << "*** Write Result to Output ***" << std::endl;
  sim_anneal.exportData();
  // sim_anneal.writeResultsXml();
}

void exportData()
{
  // create the vector of strings for the db locations
  std::vector<std::pair<std::string, std::string>> dbl_data(db_locs.size());
  for (unsigned int i = 0; i < db_locs.size(); i++) { //need the index
  dbl_data[i].first = std::to_string(db_locs[i].first);
  dbl_data[i].second = std::to_string(db_locs[i].second);
  }
  sqconn->setExport("db_loc", dbl_data);

  std::vector<std::pair<std::string, std::string>> db_dist_data(db_charges.size());
  for (unsigned int i = 0; i < db_charges.size(); i++) {
    std::string dbc_link;
    for(auto chg : db_charges[i]){
      dbc_link.append(std::to_string(chg));
    }
    db_dist_data[i].first = dbc_link;
    db_dist_data[i].second = std::to_string(config_energies[i]);
  }

  sqconn->setExport("db_charge", db_dist_data);

  sqconn->writeResultsXml();
}

void getLocation()
{

}
