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
  sqconn = new SiQADConnector(std::string("SimAnneal"), i_path, o_path);

  rng.seed(std::time(NULL));
  dis01 = boost::random::uniform_real_distribution<float>(0,1);
}


void SimAnneal::exportData()
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


bool SimAnneal::runSim()
{
  // grab all physical locations (in original distance unit)
  std::cout << "Grab all physical locations..." << std::endl;
  n_dbs = 0;
  for(auto db : *(sqconn->dbCollection())) {
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
  t_max = std::stoi(sqconn->getParameter("anneal_cycles"));
  v_0 = std::stof(sqconn->getParameter("global_v0"));
  debye_length = std::stof(sqconn->getParameter("debye_length"));
  debye_length *= 1E-9; // TODO change the rest of the code to use nm / angstrom
                        //      instead of doing a conversion here.

  kT0 = constants::Kb;
  kT0 *= std::stof(sqconn->getParameter("min_T"));
  std::cout << "kT0 retrieved: " << std::stof(sqconn->getParameter("min_T"));

  result_queue_size = std::stoi(sqconn->getParameter("result_queue_size"));
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
  occ.resize(n_dbs);

  config_energies.resize(result_queue_size);

  std::cout << "Variable initialization complete" << std::endl << std::endl;
}

void SimAnneal::precalc()
{
  std::cout << "Performing pre-calculation..." << std::endl;

  for (int i=0; i<n_dbs; i++) {
    for (int j=i; j<n_dbs; j++) {
      if (j==i) {
        db_r(i,j) = 0;
        v_ij(i,j) = 0;
      } else {
        db_r(i,j) = distance(db_locs[i].first, db_locs[i].second, db_locs[j].first, db_locs[j].second)*db_distance_scale;
        v_ij(i,j) = interElecPotential(db_r(i,j));
        db_r(j,i) = db_r(i,j);
        v_ij(j,i) = v_ij(i,j);
      }
      std::cout << "db_r[" << i << "][" << j << "]=" << db_r(i,j) << ", v_ij["
          << i << "][" << j << "]=" << v_ij(i,j) << std::endl;
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
  float E_sys;                  // energy of the system
  ublas::vector<int> dn(n_dbs); // change of occupation for population update
  int from_occ_ind, to_occ_ind; // hopping from n[occ[from_ind]]
  int from_ind, to_ind;         // hopping from n[from_ind] to n[to_ind]
  int hop_attempts;

  E_sys = systemEnergy();
  v_local = v_ext - ublas::prod(v_ij, n);

  // Run simulated annealing for predetermined time steps
  while (t < t_max) {

    printCharges();

    // Population
    std::cout << "Population update, v_freeze=" << v_freeze << ", kT=" << kT << std::endl;
    dn = genPopDelta();

    bool pop_changed = false;
    for (unsigned i=0; i<dn.size(); i++) {
      if (dn[i] != 0) {
        pop_changed = true;
        break;
      }
    }

    if (pop_changed) {
      n += dn;
      E_sys += -1 * ublas::inner_prod(v_local, dn) + totalCoulombPotential(dn);
      v_local -= ublas::prod(v_ij, dn);

      std::cout << "dn = [ ";
      for (unsigned i=0; i<dn.size(); i++)
        std::cout << dn(i) << " ";
      std::cout << "]" << std::endl;

      printCharges();
      std::cout << "v_local = [ ";
      for (unsigned i=0; i<v_local.size(); i++)
        std::cout << v_local(i) << " ";
      std::cout << "]" << std::endl;

      //std::cout << "E_calc = " << systemEnergy() << std::endl;
      std::cout << "E_sys = " << E_sys << std::endl;
    }


    // Occupation list update
    int occ_ind=0, unocc_ind=n_dbs-1;
    for (int db_ind=0; db_ind<n_dbs; db_ind++) {
      if (n[db_ind])
        occ[occ_ind++] = db_ind;
      else
        occ[unocc_ind--] = db_ind;
    }
    std::cout << "occ = [ ";
    for (unsigned i=0; i<dn.size(); i++)
      std::cout << occ[i] << " ";
    std::cout << "]" << std::endl;
    n_elec = occ_ind;


    // Hopping
    std::cout << "Hopping with n_elec=" << n_elec << std::endl;
    hop_attempts = 0;
    if (n_elec != 0) {
      while (hop_attempts < (n_dbs-n_elec)*5) {
        from_occ_ind = getRandOccInd(1);
        to_occ_ind = getRandOccInd(0);
        from_ind = occ[from_occ_ind];
        to_ind = occ[to_occ_ind];

        //E_pre_hop = systemEnergy();
        float E_del = hopEnergyDelta(from_ind, to_ind);
        if (acceptHop(E_del)) {
          performHop(from_ind, to_ind);
          occ[from_occ_ind] = to_ind;
          occ[to_occ_ind] = from_ind;
          // calculate energy difference
          E_sys += E_del;
          ublas::matrix_column<ublas::matrix<float>> v_i (v_ij, from_ind);
          ublas::matrix_column<ublas::matrix<float>> v_j (v_ij, to_ind);
          v_local += v_i - v_j;

          std::cout << "Hop performed: ";
          printCharges();
          std::cout << "Energy diff=" << E_del << std::endl;
        }
        hop_attempts++;
      }
    }
    std::cout << "Charge post-hop=";
    printCharges();

    // push back the new arrangement
    db_charges.push_back(n);
    config_energies.push_back(E_sys);

    // perform time-step if not pre-annealing
    timeStep();

    // print statistics
    std::cout << "Cycle: " << t;
    std::cout << ", ending energy: " << E_sys << std::endl << std::endl;
    /*std::cout << ", delta: " << E_del << std::endl << std::endl;*/
  }

  std::cout << "Final energy should be: " << systemEnergy() << std::endl;
}

ublas::vector<int> SimAnneal::genPopDelta()
{
  ublas::vector<int> dn(n_dbs);
  for (unsigned i=0; i<n.size(); i++) {
    float prob = 1. / ( 1 + exp( ((2*n[i]-1)*v_local[i] + v_freeze) / kT ) );
    dn[i] = evalProb(prob) ? 1 - 2*n[i] : 0;
  }
  return dn;
}

void SimAnneal::performHop(int from_ind, int to_ind)
{
  n[from_ind] = 0;
  n[to_ind] = 1;
}


void SimAnneal::timeStep()
{
  t++;
  kT = kT0 + (kT - kT0) * kT_step;
  v_freeze = t * v_freeze_step;
}


void SimAnneal::printCharges()
{
  for(int i=0; i<n_dbs; i++)
    std::cout << n[i];
  std::cout << std::endl;
}





// ACCEPTANCE FUNCTIONS


bool SimAnneal::acceptPop(int db_ind)
{
  int curr_charge = n[db_ind];
  float v = curr_charge ? v_ext[db_ind] + v_freeze : - v_ext[db_ind] + v_freeze; // 1->0 : 0->1
  float prob;

  prob = 1. / ( 1 + exp( v/kT ) );

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
  //float generated_num = dis01(rng);
  boost::variate_generator<boost::random::mt19937&, boost::random::uniform_real_distribution<float>> rnd_gen(rng, dis01);

  return prob >= rnd_gen();
}





// ACCESSORS


int SimAnneal::getRandOccInd(int charge)
{
  int min,max;
  if (charge) {
    min = 0;
    max = n_elec-1;
  } else {
    min = n_elec;
    max = n_dbs-1;
  }
  boost::random::uniform_int_distribution<int> dis(min,max);
  return dis(rng);
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


float SimAnneal::distance(const float &x1, const float &y1, const float &x2, const float &y2)
{
  return sqrt(pow(x1-x2, 2.0) + pow(y1-y2, 2.0));
}


float SimAnneal::totalCoulombPotential(ublas::vector<int> config)
{
  return 0.5 * ublas::inner_prod(config, ublas::prod(v_ij, config));
}


float SimAnneal::interElecPotential(const float &r)
{
  //return exp(-r/debye_length) / r;
  return constants::Q0 * Kc * erf(r/constants::ERFDB) * exp(-r/debye_length) / r;
}


float SimAnneal::hopEnergyDelta(const int &i, const int &j)
{
  return v_local[i] - v_local[j] - v_ij(i,j);
}
