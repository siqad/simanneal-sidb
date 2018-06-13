// @file:     main.cc
// @author:   Samuel
// @created:  2017.08.28
// @editted:  2018-06-13 - Robert
// @license:  GNU LGPL v3
//
// @desc:     Main function for physics engine

#include "sim_anneal.h"
#include <iostream>
#include <string>

using namespace phys;

//initialization of static variables used in the SimAnneal class
int SimAnneal::n_dbs = -1;
int SimAnneal::t_max = 0;
int SimAnneal::num_threads = 0;
float SimAnneal::Kc = 0;
float SimAnneal::debye_length = 0;
float SimAnneal::v_0 = 0;
float SimAnneal::kT0 = 0, SimAnneal::kT_step = 0, SimAnneal::v_freeze_step = 0;
int SimAnneal::result_queue_size = 0;
std::vector<std::pair<float,float>> SimAnneal::db_locs = { };
ublas::matrix<float> SimAnneal::db_r = { };  // distance between all dbs
ublas::vector<float> SimAnneal::v_ext = { }; // keep track of voltages at each DB
ublas::matrix<float> SimAnneal::v_ij = { };     // coulombic repulsion

std::vector< boost::circular_buffer<ublas::vector<int>> > SimAnneal::chargeStore = {};
std::vector< boost::circular_buffer<float> > SimAnneal::energyStore = {};


// temporary main function for testing the xml parsing functionality
int main(int argc, char *argv[])
{
  SiQADConnector* sqconn;

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

  std::cout << std::endl << "*** Constructing Problem ***" << std::endl;
  SimAnneal sim_accessor(-1);

  std::cout << std::endl << "*** Run Simulation ***" << std::endl;

  sqconn = new SiQADConnector(std::string("SimAnneal"), if_name, of_name);


  // grab all physical locations (in original distance unit) (Used to be part of runSim)
  std::cout << "Grab all physical locations..." << std::endl;
  sim_accessor.n_dbs = 0;
  for(auto db : *(sqconn->dbCollection())) {
    sim_accessor.db_locs.push_back(std::make_pair(db->x, db->y));
    sim_accessor.n_dbs++;
    std::cout << "DB loc: x=" << sim_accessor.db_locs.back().first
        << ", y=" << sim_accessor.db_locs.back().second << std::endl;
  }
  std::cout << "Free dbs, n_dbs=" << sim_accessor.n_dbs << std::endl << std::endl;

  // exit if no dbs
  if(sim_accessor.n_dbs == 0) {
    std::cout << "No dbs found, nothing to simulate. Exiting." << std::endl;
    std::cout << "Simulation failed, aborting" << std::endl;
    return 0;
  }


  //Variable initialization
  std::cout << "Initializing variables..." << std::endl;
  sim_accessor.num_threads = std::stoi(sqconn->getParameter("num_threads"));
  sim_accessor.t_max = std::stoi(sqconn->getParameter("anneal_cycles"));
  sim_accessor.v_0 = std::stof(sqconn->getParameter("global_v0"));
  sim_accessor.debye_length = std::stof(sqconn->getParameter("debye_length"));
  sim_accessor.debye_length *= 1E-9; // TODO change the rest of the code to use nm / angstrom
                        //      instead of doing a conversion here.

  sim_accessor.kT0 = constants::Kb;
  sim_accessor.kT0 *= std::stof(sqconn->getParameter("min_T"));
  std::cout << "kT0 retrieved: " << std::stof(sqconn->getParameter("min_T"));

  sim_accessor.result_queue_size = std::stoi(sqconn->getParameter("result_queue_size"));
  sim_accessor.result_queue_size = sim_accessor.t_max < sim_accessor.result_queue_size ? sim_accessor.t_max : sim_accessor.result_queue_size;

  sim_accessor.Kc = 1/(4 * constants::PI * constants::EPS_SURFACE * constants::EPS0);
  sim_accessor.kT_step = 0.999;    // kT = Boltzmann constant (eV/K) * 298 K, NOTE kT_step arbitrary
  sim_accessor.v_freeze_step = 0.001;  // NOTE v_freeze_step arbitrary

  std::cout << "Variable initialization complete" << std::endl;

  sim_accessor.db_r.resize(sim_accessor.n_dbs,sim_accessor.n_dbs);
  sim_accessor.v_ext.resize(sim_accessor.n_dbs);
  sim_accessor.v_ij.resize(sim_accessor.n_dbs,sim_accessor.n_dbs);

  std::cout << "Performing pre-calculation..." << std::endl;

  for (int i=0; i<sim_accessor.n_dbs; i++) {
    for (int j=i; j<sim_accessor.n_dbs; j++) {
      if (j==i) {
        sim_accessor.db_r(i,j) = 0;
        sim_accessor.v_ij(i,j) = 0;
      } else {
        sim_accessor.db_r(i,j) = sim_accessor.distance(sim_accessor.db_locs[i].first, sim_accessor.db_locs[i].second, sim_accessor.db_locs[j].first, sim_accessor.db_locs[j].second)*sim_accessor.db_distance_scale;
        sim_accessor.v_ij(i,j) = sim_accessor.interElecPotential(sim_accessor.db_r(i,j));
        sim_accessor.db_r(j,i) = sim_accessor.db_r(i,j);
        sim_accessor.v_ij(j,i) = sim_accessor.v_ij(i,j);
      }
      std::cout << "db_r[" << i << "][" << j << "]=" << sim_accessor.db_r(i,j) << ", v_ij["
          << i << "][" << j << "]=" << sim_accessor.v_ij(i,j) << std::endl;
    }

    // TODO add electrode effect to v_ext

    sim_accessor.v_ext[i] = sim_accessor.v_0;
  }

  if(sim_accessor.num_threads == -1){
    if(sim_accessor.n_dbs <= 9){
      sim_accessor.num_threads = 8;
    }
    else if(sim_accessor.n_dbs <= 25){
      sim_accessor.num_threads = 16;
    }
    else{
      sim_accessor.num_threads = 100;
    }
  }

  boost::circular_buffer<boost::numeric::ublas::vector<int>> placeVec;

  for (int i = 0; i < sim_accessor.num_threads; ++ i){
    sim_accessor.chargeStore.push_back(placeVec);
  }

  sim_accessor.energyStore.resize(sim_accessor.num_threads);


  std::cout << "Pre-calculation complete" << std::endl << std::endl;


  std::vector<std::thread> threads;
  for (int i=0; i<sim_accessor.num_threads; i++) {
    SimAnneal sim(i);
    std::thread th(&SimAnneal::runSim, sim);
    threads.push_back(std::move(th));
  }

  for (auto &th : threads) {
    th.join();
  }

  std::cout << std::endl << "*** Write Result to Output ***" << std::endl;

  //Selecting the best simmulated annealing calculation if more threads were run in parallel.
  float bestThread = 0;
  if (sim_accessor.num_threads > 1){
    for(int i = 1; i < sim_accessor.num_threads; i++){
      if(sim_accessor.energyStore[i][sim_accessor.energyStore[i].size() - 1] < sim_accessor.energyStore[bestThread][sim_accessor.energyStore[bestThread].size() - 1]){
        bestThread = i;
      }
    }
  }

  std::cout << " LOWEST ENERGY FOUND: " << sim_accessor.energyStore[bestThread][sim_accessor.energyStore[bestThread].size() - 1] << std::endl;
  std::cout << " BEST THREAD ID: " << bestThread << std::endl;
  std::cout << " NUM THREADS USED: " << sim_accessor.num_threads << std::endl;


  // create the vector of strings for the db locations
  std::vector<std::pair<std::string, std::string>> dbl_data(sim_accessor.db_locs.size());
  for (unsigned int i = 0; i < sim_accessor.db_locs.size(); i++) { //need the index
  dbl_data[i].first = std::to_string(sim_accessor.db_locs[i].first);
  dbl_data[i].second = std::to_string(sim_accessor.db_locs[i].second);
  }
  sqconn->setExport("db_loc", dbl_data);

  std::vector<std::pair<std::string, std::string>> db_dist_data(sim_accessor.chargeStore[bestThread].size());
  for (unsigned int i = 0; i < sim_accessor.chargeStore[bestThread].size(); i++) {
    std::string dbc_link;
    for(auto chg : sim_accessor.chargeStore[bestThread][i]){
      dbc_link.append(std::to_string(chg));
    }
    db_dist_data[i].first = dbc_link;
    db_dist_data[i].second = std::to_string(sim_accessor.energyStore[bestThread][i]);
  }

  sqconn->setExport("db_charge", db_dist_data);

  sqconn->writeResultsXml();
}
