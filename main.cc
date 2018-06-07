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

int SimAnneal::n_dbs = -1;
int SimAnneal::t_max = 0;
float SimAnneal::Kc = 0;
float SimAnneal::debye_length = 0;
float SimAnneal::v_0 = 0;
float SimAnneal::kT0 = 0, SimAnneal::kT_step = 0, SimAnneal::v_freeze_step = 0;
int SimAnneal::result_queue_size = 0;
std::vector<std::pair<float,float>> SimAnneal::db_locs = { };
ublas::matrix<float> SimAnneal::db_r = { };  // distance between all dbs
ublas::vector<float> SimAnneal::v_ext = { }; // keep track of voltages at each DB
ublas::matrix<float> SimAnneal::v_ij = { };     // coulombic repulsion

void exportData();

// temporary main function for testing the xml parsing functionality
int main(int argc, char *argv[])
{
  SiQADConnector* sqconn;

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

  sqconn = new SiQADConnector(std::string("SimAnneal"), if_name, of_name);


  // grab all physical locations (in original distance unit) (Used to be part of runSim)
  std::cout << "Grab all physical locations..." << std::endl;
  sim_anneal.n_dbs = 0;
  for(auto db : *(sqconn->dbCollection())) {
    sim_anneal.db_locs.push_back(std::make_pair(db->x, db->y));
    sim_anneal.n_dbs++;
    std::cout << "DB loc: x=" << sim_anneal.db_locs.back().first
        << ", y=" << sim_anneal.db_locs.back().second << std::endl;
  }
  std::cout << "Free dbs, n_dbs=" << sim_anneal.n_dbs << std::endl << std::endl;

  // exit if no dbs
  if(sim_anneal.n_dbs == 0) {
    std::cout << "No dbs found, nothing to simulate. Exiting." << std::endl;
    std::cout << "Simulation failed, aborting" << std::endl;
    return 0;
  }


  //Variable initialization. (Used to be part of initVars())

  std::cout << "Initializing variables..." << std::endl;
  sim_anneal.t_max = std::stoi(sqconn->getParameter("anneal_cycles"));
  sim_anneal.v_0 = std::stof(sqconn->getParameter("global_v0"));
  sim_anneal.debye_length = std::stof(sqconn->getParameter("debye_length"));
  sim_anneal.debye_length *= 1E-9; // TODO change the rest of the code to use nm / angstrom
                        //      instead of doing a conversion here.

  sim_anneal.kT0 = constants::Kb;
  sim_anneal.kT0 *= std::stof(sqconn->getParameter("min_T"));
  std::cout << "kT0 retrieved: " << std::stof(sqconn->getParameter("min_T"));

  sim_anneal.result_queue_size = std::stoi(sqconn->getParameter("result_queue_size"));
  sim_anneal.result_queue_size = sim_anneal.t_max < sim_anneal.result_queue_size ? sim_anneal.t_max : sim_anneal.result_queue_size;

  sim_anneal.Kc = 1/(4 * constants::PI * constants::EPS_SURFACE * constants::EPS0);
  sim_anneal.kT_step = 0.999;    // kT = Boltzmann constant (eV/K) * 298 K, NOTE kT_step arbitrary
  sim_anneal.v_freeze_step = 0.001;  // NOTE v_freeze_step arbitrary

  std::cout << "Variable initialization complete" << std::endl;

  sim_anneal.db_r.resize(sim_anneal.n_dbs,sim_anneal.n_dbs);
  sim_anneal.v_ext.resize(sim_anneal.n_dbs);
  sim_anneal.v_ij.resize(sim_anneal.n_dbs,sim_anneal.n_dbs);



  std::cout << "Performing pre-calculation..." << std::endl;

  for (int i=0; i<sim_anneal.n_dbs; i++) {
    for (int j=i; j<sim_anneal.n_dbs; j++) {
      if (j==i) {
        sim_anneal.db_r(i,j) = 0;
        sim_anneal.v_ij(i,j) = 0;
      } else {
        sim_anneal.db_r(i,j) = sim_anneal.distance(sim_anneal.db_locs[i].first, sim_anneal.db_locs[i].second, sim_anneal.db_locs[j].first, sim_anneal.db_locs[j].second)*sim_anneal.db_distance_scale;
        sim_anneal.v_ij(i,j) = sim_anneal.interElecPotential(sim_anneal.db_r(i,j));
        sim_anneal.db_r(j,i) = sim_anneal.db_r(i,j);
        sim_anneal.v_ij(j,i) = sim_anneal.v_ij(i,j);
      }
      std::cout << "db_r[" << i << "][" << j << "]=" << sim_anneal.db_r(i,j) << ", v_ij["
          << i << "][" << j << "]=" << sim_anneal.v_ij(i,j) << std::endl;
    }

    // TODO add electrode effect to v_ext

    sim_anneal.v_ext[i] = sim_anneal.v_0;
  }
  std::cout << "Pre-calculation complete" << std::endl << std::endl;



  SimAnneal sim_1(if_name, of_name);
  std::thread t1(&SimAnneal::runSim, sim_1);
  t1.join();


  std::cout << std::endl << "*** Write Result to Output ***" << std::endl;
  //sim_anneal.exportData();
  // sim_anneal.writeResultsXml(); (Commented by Sam)



  //exportData();
}

/*
void exportData()
{
  // create the vector of strings for the db locations
  std::vector<std::pair<std::string, std::string>> dbl_data(sim_anneal.db_locs.size());
  for (unsigned int i = 0; i < sim_anneal.db_locs.size(); i++) { //need the index
  dbl_data[i].first = std::to_string(sim_anneal.db_locs[i].first);
  dbl_data[i].second = std::to_string(sim_anneal.db_locs[i].second);
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
*/
