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

void initExpectedParams();
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

  phys_con = new PhysicsConnector(std::string("SimAnneal"), i_path, o_path);
  initExpectedParams();

  if(!sim_anneal.runSim()) {
    std::cout << "Simulation failed, aborting" << std::endl;
    return 0;
  }

  std::cout << std::endl << "*** Write Result to Output ***" << std::endl;
  sim_anneal.exportData();
  // sim_anneal.writeResultsXml();
}


void initExpectedParams()
{
  std::cout << "SimAnneal instantiated." << std::endl;
  phys_con->setRequiredSimParam("anneal_cycles");
  phys_con->setRequiredSimParam("global_v0");
  phys_con->setRequiredSimParam("debye_length");
  phys_con->setRequiredSimParam("result_queue_size");
  phys_con->setExpectDB(true);
  phys_con->readProblem();
  for (auto& iter : phys_con->getRequiredSimParam()) {
    if(!phys_con->parameterExists(iter)){
      std::cout << "Parameter " << iter << " not found." << std::endl;
    }
  }
}

void exportData()
{
  // create the vector of strings for the db locations
  std::vector<std::vector<std::string>> dbl_data(SimAnneal::db_locs.size());
  for (unsigned int i = 0; i < SimAnneal::db_locs.size(); i++) { //need the index
    dbl_data[i].resize(2);
    dbl_data[i][0] = std::to_string(SimAnneal::db_locs[i].first);
    dbl_data[i][1] = std::to_string(SimAnneal::db_locs[i].second);
  }
  phys_con->setExportDBLoc(true);
  phys_con->setDBLocData(dbl_data);

  std::vector<std::vector<std::string>> db_dist_data(SimAnneal::db_charges.size());
  //unsigned int i = 0;
  for (unsigned int i = 0; i < SimAnneal::db_charges.size(); i++) {
  //for (auto db_charge : db_charges) {
    db_dist_data[i].resize(2);
    std::string dbc_link;
    for(auto chg : SimAnneal::db_charges[i]){
      dbc_link.append(std::to_string(chg));
    }
    db_dist_data[i][0] = dbc_link;
    db_dist_data[i][1] = std::to_string(SimAnneal::config_energies[i]);
    // std::cout << db_dist_data[i][0] << std::endl;
  }

  phys_con->setExportDBElecConfig(true);
  phys_con->setDBElecData(db_dist_data);

  phys_con->writeResultsXml();
}

void getLocation()
{

}
