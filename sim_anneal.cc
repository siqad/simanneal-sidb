// @file:     sim_anneal.cc
// @author:   Samuel
// @created:  2017.08.23
// @editted:  2017.08.23 - Samuel
// @license:  GNU LGPL v3
//
// @desc:     Simulated annealing physics engine

#include "sim_anneal.h"
#include <ctime>

using namespace phys;

SimAnneal::SimAnneal(const std::string& i_path, const std::string& o_path)
{
  phys_con = new PhysicsConnector(std::string("SimAnneal"), i_path, o_path);
  rng.seed(std::time(NULL));
  initExpectedParams();
}


void SimAnneal::initExpectedParams()
{
  std::cout << "SimAnneal instantiated." << std::endl;
  phys_con->setRequiredSimParam("preanneal_cycles");
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


void SimAnneal::exportData()
{
  // create the vector of strings for the db locations
  std::vector<std::vector<std::string>> dbl_data(db_locs.size());
  for (unsigned int i = 0; i < db_locs.size(); i++) { //need the index
    dbl_data[i].resize(2);
    dbl_data[i][0] = std::to_string(db_locs[i].first);
    dbl_data[i][1] = std::to_string(db_locs[i].second);
  }
  phys_con->setExportDBLoc(true);
  phys_con->setDBLocData(dbl_data);

  std::vector<std::vector<std::string>> db_dist_data(db_charges.size());
  //unsigned int i = 0;
  for (unsigned int i = 0; i < db_charges.size(); i++) {
  //for (auto db_charge : db_charges) {
    db_dist_data[i].resize(2);
    std::string dbc_link;
    for(auto chg : db_charges[i]){
      dbc_link.append(std::to_string(chg));
    }
    db_dist_data[i][0] = dbc_link;
    db_dist_data[i][1] = std::to_string(config_energies[i]);
    // std::cout << db_dist_data[i][0] << std::endl;
  }

  phys_con->setExportDBElecConfig(true);
  phys_con->setDBElecData(db_dist_data);

  phys_con->writeResultsXml();
}


bool SimAnneal::runSim()
{
  // grab all physical locations (in original distance unit)
  std::cout << "Grab all physical locations..." << std::endl;
  n_dbs = 0;
  phys_con->initCollections();
  for(auto db : *(phys_con->db_col)) {
    db_locs.push_back(std::make_pair(db->x, db->y));
    n_dbs++;
    std::cout << "DB loc: x=" << db_locs.back().first
        << ", y=" << db_locs.back().second << std::endl;
  }
  std::cout << "Free dbs, n_dbs=" << n_dbs << std::endl << std::endl;

  // exit if no dbs
  if(n_dbs == 0) {
    std::cout << "No dbs found, nothing to simulate. Exiting." << std::endl;
    return false;
  }

  // initialize variables & perform pre-calculation
  initVars();
  precalc();

  // SIM ANNEAL
  simAnneal();

  return true;
}


// PRIVATE

void SimAnneal::initVars()
{
  std::cout << "Initializing variables..." << std::endl;
  if(n_dbs <= 0){
    std::cout << "There are no dbs in the problem!" << std::endl;
    return;
  }
  t_preanneal = phys_con->parameterExists("preanneal_cycles") ?
                  std::stoi(phys_con->getParameter("preanneal_cycles")) : 1000;
  t_max = phys_con->parameterExists("anneal_cycles") ?
                  std::stoi(phys_con->getParameter("anneal_cycles")) : 10000;
  v_0 = phys_con->parameterExists("global_v0") ?
                  std::stof(phys_con->getParameter("global_v0")) : 0.25;
  debye_length = phys_con->parameterExists("debye_length") ?
                  std::stof(phys_con->getParameter("debye_length")) : 5;
  debye_length *= 1E-9; // TODO change the rest of the code to use nm / angstrom
                        //      instead of doing a conversion here.

  kT0 = constants::Kb;
  kT0 *= phys_con->parameterExists("min_T") ?
            std::stof(phys_con->getParameter("min_T")) : 4;
  std::cout << "kT0 retrieved: " << std::stof(phys_con->getParameter("min_T"));

  result_queue_size = phys_con->parameterExists("result_queue_size") ?
                  std::stoi(phys_con->getParameter("result_queue_size")) : 1000;
  result_queue_size = t_max < result_queue_size ? t_max : result_queue_size;

  Kc = 1/(4 * constants::PI * constants::EPS_SURFACE * constants::EPS0);
  kT = 300*constants::Kb; kT_step = 0.999;    // kT = Boltzmann constant (eV/K) * 298 K, NOTE kT_step arbitrary
  v_freeze = 0, v_freeze_step = 0.001;  // NOTE v_freeze_step arbitrary

  // resize vectors
  v_local.resize(n_dbs);
  v_ext.resize(n_dbs);
  v_ij.resize(n_dbs,n_dbs);
  db_r.resize(n_dbs,n_dbs);

  db_charges.resize(result_queue_size);
  n.resize(n_dbs);

  config_energies.resize(result_queue_size);
  config_energies.push_back(0);

  std::cout << "Variable initialization complete" << std::endl << std::endl;
}

void SimAnneal::precalc()
{
  std::cout << "Performing pre-calculation..." << std::endl;
  if(n_dbs <= 0){
    std::cout << "There are no dbs in the problem!" << std::endl;
    return;
  }

  for(int i=0; i<n_dbs; i++) {
    for(int j=i; j<n_dbs; j++) {
      if (j==i) {
        db_r(i,j) = 0;
        v_ij(i,j) = 0;
      } else {
        db_r(i,j) = distance(db_locs[i].first, db_locs[i].second, db_locs[j].first, db_locs[j].second)*db_distance_scale;
        v_ij(i,j) = interElecPotential(db_r(i,j));
        db_r(j,i) = db_r(i,j);
        v_ij(j,i) = v_ij(i,j);
      }
      std::cout << "db_r[" << i << "][" << j << "]=" << db_r(i,j) << ", v_ij[" << i << "][" << j << "]=" << v_ij(i,j) << std::endl;
    }

    // TODO add electrode effect to v_ext

    v_ext[i] = v_0;
  }
  std::cout << "Pre-calculation complete" << std::endl << std::endl;
}


void SimAnneal::simAnneal()
{
  std::cout << "Performing simulated annealing..." << std::endl;

  // Vars
  float E_sys, E_begin, E_end;
  ublas::vector<int> dn(n_dbs);
  int from_ind, to_ind; // hopping from -> to (indices)
  int hop_count;
  float E_pre_hop, E_post_hop;

  E_sys = systemEnergy();
  v_local = v_ext - ublas::prod(v_ij, n);

  // Run simulated annealing for predetermined time steps
  while(t < t_max) {
    E_begin = systemEnergy();


    printCharges();
    // Population
    std::cout << "Population update, v_freeze=" << v_freeze << ", kT=" << kT << std::endl;
    dn = genPopDelta();
    n += dn;
    E_sys += -1 * ublas::inner_prod(v_local, dn) + totalCoulombPotential(dn);
    v_local -= ublas::prod(v_ij, dn);

    std::cout << "dn = [ ";
    for (int i=0; i<dn.size(); i++)
      std::cout << dn(i) << " ";
    std::cout << "]" << std::endl;

    printCharges();
    std::cout << "v_local = [ ";
    for (int i=0; i<v_local.size(); i++)
      std::cout << v_local(i) << " ";
    std::cout << "]" << std::endl;

    std::cout << "E_calc = " << systemEnergy() << std::endl;
    std::cout << "E_sys = " << E_sys << std::endl;


    // Hopping
    /*std::cout << "Hopping" << std::endl;
    hop_count = 0;
    int unocc_count = chargedDBCount(1);
    while(hop_count < unocc_count*5) {
      // TODO instead of finding system energy twice, could just find the potential difference of the db between pre- and post-hop
      E_pre_hop = systemEnergy(); // original energy
      from_ind = getRandDBInd(1);
      to_ind = getRandDBInd(0);

      if(from_ind == -1 || to_ind == -1)
        break; // hopping not possible

      // perform the hop
      dbHop(from_ind, to_ind);
      E_post_hop = systemEnergy(); // new energy

      // accept hop given energy change? reverse hop if energy delta is unaccpted
      if(!acceptHop(E_post_hop-E_pre_hop))
        dbHop(to_ind, from_ind);
        //n[from_ind] = 1, n[to_ind] = 0;
      else{
        std::cout << "Hop performed: ";
        printCharges();
        std::cout << "Energy diff=" << E_post_hop-E_pre_hop << std::endl;
      }
      hop_count++;
    }
    std::cout << "Charge post-hop=";
    printCharges();*/

    E_end = systemEnergy();

    // push back the new arrangement
    db_charges.push_back(n);
    config_energies.push_back(E_end);

    // perform time-step if not pre-annealing
    if(t_preanneal > 0)
      t_preanneal--;
    else
      timeStep();

    // print statistics
    std::cout << "Cycle: " << ((t_preanneal > 0) ? -t_preanneal : t);
    std::cout << ", ending energy: " << E_end;
    std::cout << ", delta: " << E_end-E_begin << std::endl << std::endl;
  }
}

ublas::vector<int> SimAnneal::genPopDelta()
{
  ublas::vector<int> dn(n_dbs);
  for (int i=0; i<n.size(); i++) {
    float prob = 1. / ( 1 + exp( ((2*n[i]-1)*v_local[i] + v_freeze) / kT ) );
    dn[i] = evalProb(prob) ? 1 - 2*n[i] : 0;
  }
  return dn;
}

void SimAnneal::dbHop(int from_ind, int to_ind)
{
  n[from_ind] = 0;
  n[to_ind] = 1;
}


void SimAnneal::timeStep()
{
  t++;
  kT = kT0 + (kT - kT0) * kT_step;
  v_freeze += v_freeze_step;
}


void SimAnneal::printCharges()
{
  for(int i=0; i<n_dbs; i++)
    std::cout << n[i];
  //for(int i : *n)
  //  std::cout << i;
  std::cout << std::endl;
}





// ACCEPTANCE FUNCTIONS


bool SimAnneal::acceptPop(int db_ind)
{
  int curr_charge = n[db_ind];
  float v = curr_charge ? v_ext[db_ind] + v_freeze : - v_ext[db_ind] + v_freeze; // 1->0 : 0->1
  float prob;

  prob = 1. / ( 1 + exp( v/kT ) );

  //std::cout << "v_eff=" << v_eff[db_ind] << ", P(" << curr_charge << "->" << !curr_charge << ")=" << prob << std::endl;

  return evalProb(prob);
}


// acceptance function for hopping
bool SimAnneal::acceptHop(float v_diff)
{
  if (v_diff < 0)
    return true;

  // some acceptance function, acceptance probability falls off exponentially
  float prob = exp(-v_diff/kT);
  return evalProb(prob);
}


// takes a probability and generates true/false accordingly
bool SimAnneal::evalProb(float prob)
{
  boost::random::uniform_real_distribution<float> dis(0,1);
  boost::variate_generator<boost::random::mt19937&, boost::random::uniform_real_distribution<float>> rnd_gen(rng, dis);

  float generated_num = rnd_gen();

  return prob >= generated_num;
}





// ACCESSORS


int SimAnneal::chargedDBCount(int charge)
{
  int i=0;
  for(int db_charge : n)
    if(db_charge == charge)
      i++;
  return i;
}


int SimAnneal::getRandDBInd(int charge)
{
  std::vector<int> dbs;

  // store the indices of dbs that have the desired occupation
  for (unsigned int i=0; i<n.size(); i++)
    if (n[i] == charge)
      dbs.push_back(i);

  if (dbs.empty())
    return -1; // no potential candidates

  // pick one from them
  boost::random::uniform_int_distribution<int> dis(0,dbs.size()-1);
  boost::variate_generator<boost::random::mt19937&, boost::random::uniform_int_distribution<int>> rnd_gen(rng, dis);
  // TODO move these to init and make them class var, not reinitialize it every time

  return dbs[rnd_gen()];
}





// PHYS CALCU


float SimAnneal::systemEnergy()
{
  assert(n_dbs > 0);
  float v = 0;
  for(int i=0; i<n_dbs; i++) {
    v -= v_0 + n[i] * (v_ext[i]);
    for(int j=i+1; j<n_dbs; j++)
      v += n[i] * n[j] * v_ij(i,j);
  }
  //return v * har_to_ev; // revert back to this when going back to hartree calculations
  return v;
}


float SimAnneal::distance(float x1, float y1, float x2, float y2)
{
  return sqrt(pow(x1-x2, 2.0) + pow(y1-y2, 2.0));
}


float SimAnneal::totalCoulombPotential(ublas::vector<int> config)
{
  return 0.5 * ublas::inner_prod(config, ublas::prod(v_ij, config));
}


float SimAnneal::interElecPotential(float r)
{
  //return exp(-r/debye_length) / r;
  return constants::Q0 * Kc * erf(r/constants::ERFDB) * exp(-r/debye_length) / r;
}
