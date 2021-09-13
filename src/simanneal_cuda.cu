// @file:     simanneal_cuda.cu
// @author:   Samuel
// @created:  2021.07.27
// @license:  Apache License 2.0
//
// @desc:     Implementation of CUDA SimAnneal

#include <limits>

#include "simanneal_alg.cu"
#include "simanneal_cuda.h"

saglobal::TimeKeeper *saglobal::TimeKeeper::time_keeper=nullptr;
int saglobal::log_level = Logger::DBG;

using namespace phys;

// static variables
// TODO should probably make some of these non-static to support running multiple
// simulations
SimParamsCuda SimAnnealCuda::sim_params;
FPType SimAnnealCuda::db_distance_scale = 1e-10;

// alias for the commonly used sim_params static variable
constexpr auto sparams = &SimAnnealCuda::sim_params;

// SimParams implementation

void SimParamsCuda::setDBLocs(const std::vector<EuclCoord> &t_db_locs)
{
  db_locs = t_db_locs;
  if (db_locs.size() == 0) {
    throw "There must be 1 or more DBs when setting DBs for SimParams.";
  }
  n_dbs = db_locs.size();
  db_r.resize(n_dbs, n_dbs);
  v_ij.resize(n_dbs, n_dbs);
  v_ext.resize(n_dbs);
}

void SimParamsCuda::setDBLocs(const std::vector<LatCoord> &t_db_locs)
{
  std::vector<EuclCoord> db_locs;
  for (LatCoord lat_coord : t_db_locs) {
    assert(lat_coord.size() == 3);
    db_locs.push_back(latToEuclCoord(lat_coord[0], lat_coord[1], lat_coord[2]));
  }
  setDBLocs(db_locs);
}

EuclCoord SimParamsCuda::latToEuclCoord(const int &n, const int &m, const int &l)
{
  FPType x = n * constants::lat_a;
  FPType y = m * constants::lat_b + l * constants::lat_c;
  return std::make_pair(x, y);
}

// SimAnnealCuda implementation

SimAnnealCuda::SimAnnealCuda(SimParamsCuda &sparams)
{
  sim_params = sparams;
  initialize();
}

void SimAnnealCuda::invoke()
{
  Logger log(saglobal::log_level);
  SimParamsCuda &sp = sim_params;

  cudaFree(0);  // absorb the startup delay when profiling

  // vars to be shared with GPU
  //FPType *best_energy;
  //int *returned_configs;
  float *v_ext;
  float *db_locs_eucl;
  //cudaMallocManaged(&best_energy, sizeof(FPType));
  //cudaMallocManaged(&returned_configs, sp.n_dbs*sizeof(int));
  cudaMallocManaged(&v_ext, sp.n_dbs*sizeof(FPType));
  cudaMallocManaged(&db_locs_eucl, 2*sp.n_dbs*sizeof(FPType));

  // init vars to be shared
  for (int i=0; i<sparams->n_dbs; i++) {
    //returned_configs[i] = -2;  // -2 is an invalid charge state so init to it
    v_ext[i] = sp.v_ext[i];
    db_locs_eucl[IDX2C(i,0,sp.n_dbs)] = sp.db_locs[i].first;
    db_locs_eucl[IDX2C(i,1,sp.n_dbs)] = sp.db_locs[i].second;
  }

  // initialize handles and variables
  float kT_0 = sp.T_init*constants::Kb;
  float muzm = sp.mu;
  float mupz = sp.mu - constants::eta;
  ::initDeviceVars<<<1, 1>>>(sp.n_dbs, muzm, mupz, sp.alpha, kT_0, sp.kT_min, sp.v_freeze_threshold, sp.v_freeze_step, sp.anneal_cycles, sp.hop_attempt_factor, v_ext);
  cudaDeviceSynchronize();
  ::initVij<<<1, 1>>>(sp.n_dbs, sp.eps_r, sp.debye_length, db_locs_eucl); // TODO check multi-thread
  // TODO initialize handles

  // test with just one stream
  log.debug() << "Starting simanneal" << std::endl;
  cudaDeviceSynchronize();
  //int numBlocks = (N + blockSize - 1) / blockSize;
  int num_streams = sp.num_instances;
  if (num_streams == -1) {
    if (sp.n_dbs <= 9) {
      num_streams = 10;
    } else if (sp.n_dbs <= 25) {
      num_streams = 20;
    } else {
      num_streams = 40;
    }
  }
  cudaStream_t streams[num_streams];
  //FPType *best_energy[num_streams];
  int *returned_configs[num_streams];

  int numBlocks = 1;
  int blockSize = sp.threads_in_instance;
  for (int i=0; i<num_streams; i++) {
    gpuErrChk(cudaStreamCreate(&streams[i]));

    //gpuErrChk(cudaMallocManaged(&best_energy[i], sizeof(FPType)));
    gpuErrChk(cudaMallocManaged(&returned_configs[i], sp.n_dbs * sizeof(int)));

    ::runAnneal<<<numBlocks, blockSize, 0, streams[i]>>>(i, returned_configs[i]);
    //::runAnneal<<<numBlocks, blockSize, 0, streams[i]>>>();
  }

  // wait for GPU to finish before accessing on host
  for (int i=0; i<num_streams; i++) {
    gpuErrChk(cudaStreamSynchronize(streams[i]));
  }
  log.debug() << "Ended simanneal" << std::endl;

  // get results and free memory
  received_results.resize(num_streams);
  for (int stream_id=0; stream_id<num_streams; stream_id++) {
    received_results[stream_id].resize(sp.n_dbs);
    log.debug() << "Received result for stream " << stream_id << ": [";
    for (int db_i = 0; db_i < sp.n_dbs; db_i++) {
      received_results[stream_id][db_i] = returned_configs[stream_id][db_i];
      log.debug() << received_results[stream_id][db_i];
      if (db_i != sp.n_dbs - 1) {
        log.debug() << ", ";
      }
    }
    log.debug() << "]" << std::endl;

    //cudaFree(best_energy[stream_id]);
    cudaFree(returned_configs[stream_id]);
  }
}

FPType SimAnnealCuda::systemEnergy(const ublas::vector<int> &n_in, bool qubo)
{
  assert(n_in.size() > 0);

  //FPType E = 0.5 * ublas::inner_prod(n_in, ublas::prod(sim_params.v_ij, n_in))
    //- ublas::inner_prod(n_in, sim_params.v_ext);
  FPType E = ublas::inner_prod(n_in, sim_params.v_ext)
    + 0.5 * ublas::inner_prod(n_in, ublas::prod(sim_params.v_ij, n_in));
    

  if (qubo) {
    for (int n_i : n_in) {
      E += n_i * sim_params.mu;
    }
  }
  
  return E;
}

bool SimAnnealCuda::isMetastable(const ublas::vector<int> &n_in)
{
  assert(n_in.size() > 0);
  Logger log(saglobal::log_level);

  const FPType &muzm = sparams->mu;
  const FPType &mupz = sparams->mu - constants::eta;
  const FPType &zero_equiv = constants::RECALC_STABILITY_ERR;

  ublas::vector<FPType> v_local(n_in.size());
  // TODO reenable
  //log.debug() << "V_i and Charge State Config " << n_in << ":" << std::endl;
  for (unsigned int i=0; i<n_in.size(); i++) {
    // calculate v_i
    v_local[i] = - sim_params.v_ext[i];
    for (unsigned int j=0; j<n_in.size(); j++) {
      if (i == j) continue;
      v_local[i] -= sim_params.v_ij(i,j) * n_in[j];
    }
    log.debug() << "\tDB[" << i << "]: charge state=" << n_in[i]
      << ", v_local[i]=" << v_local[i] << " eV, and v_local[i]+muzm=" << v_local[i] + muzm << "eV" << std::endl;

    // return false if invalid
    if (!(   (n_in[i] == -1 && v_local[i] + muzm < zero_equiv)    // DB- valid condition
          || (n_in[i] == 1  && v_local[i] + mupz > - zero_equiv)  // DB+ valid condition
          || (n_in[i] == 0  && v_local[i] + muzm > - zero_equiv   // DB0 valid condition
                            && v_local[i] + mupz < zero_equiv))) {
      // TODO reenable
      //log.debug() << "config " << n_in << " has an invalid population, failed at index " << i << std::endl;
      //log.debug() << "v_local[i]=" << v_local[i] << ", muzm=" << muzm << ", mupz=" << mupz << std::endl;
      return false;
    }
  }
  //log.debug() << "config " << n_in << " has a valid population." << std::endl;

  auto hopDel = [v_local, n_in](const int &i, const int &j) -> FPType {
    int dn_i = (n_in[i]==-1) ? 1 : -1;
    int dn_j = - dn_i;
    return - v_local[i]*dn_i - v_local[j]*dn_j - sparams->v_ij(i,j);
  };

  for (unsigned int i=0; i<n_in.size(); i++) {
    // do nothing with DB+
    if (n_in[i] == 1)
      continue;

    for (unsigned int j=0; j<n_in.size(); j++) {
      // attempt hops from more negative charge states to more positive ones
      FPType E_del = hopDel(i, j);
      if ((n_in[j] > n_in[i]) && (E_del < -zero_equiv)) {
        /* TODO reenable
        log.debug() << "config " << n_in << " not stable since hopping from site "
          << i << " to " << j << " would result in an energy change of "
          << E_del << std::endl;
          */
        return false;
      }
    }
  }
  // TODO reenable
  //log.debug() << "config " << n_in << " has a stable configuration." << std::endl;
  return true;
}

void SimAnnealCuda::initialize()
{
  Logger log(saglobal::log_level);
  SimParamsCuda &sp = sim_params;

  log.debug() << "Performing pre-calculations..." << std::endl;

  // set default values
  if (sp.v_freeze_init < 0) {
    sp.v_freeze_init = fabs(sp.mu) / 2;
  } else {
    sp.v_freeze_reset = fabs(sp.mu);
  }

  // apply schedule scaling
  sp.alpha = std::pow(std::exp(-1.), 1./(sp.T_e_inv_point * sp.anneal_cycles));
  sp.v_freeze_cycles = sp.v_freeze_end_point * sp.anneal_cycles;
  sp.v_freeze_step = sp.v_freeze_threshold / sp.v_freeze_cycles;

  log.debug() << "Anneal cycles: " << sp.anneal_cycles << ", alpha: " 
    << sp.alpha << ", v_freeze_cycles: " << sp.v_freeze_cycles << std::endl;

  sp.result_queue_size = sp.anneal_cycles * sp.result_queue_factor;
  sp.result_queue_size = std::min(sp.result_queue_size, sp.anneal_cycles);
  sp.result_queue_size = std::max(sp.result_queue_size, 1);
  log.debug() << "Result queue size: " << sp.result_queue_size << std::endl;

  if (sp.preanneal_cycles > sp.anneal_cycles) {
    std::cerr << "Preanneal cycles > Anneal cycles";
    throw;
  }

  // phys
  sp.kT_min = constants::Kb * sp.T_min;
  sp.Kc = 1/(4 * constants::PI * sp.eps_r * constants::EPS0);

  // inter-db distances and voltages
  for (int i=0; i<sp.n_dbs; i++) {
    sp.db_r(i,i) = 0.;
    sp.v_ij(i,i) = 0.;
    for (int j=i+1; j<sp.n_dbs; j++) {
      sp.db_r(i,j) = db_distance_scale * distance(i,j);
      sp.v_ij(i,j) = interElecPotential(sp.db_r(i,j));
      sp.db_r(j,i) = sp.db_r(i,j);
      sp.v_ij(j,i) = sp.v_ij(i,j);

      log.debug() << "db_r[" << i << "][" << j << "]=" << sp.db_r(i,j) 
        << ", v_ij[" << i << "][" << j << "]=" << sp.v_ij(i,j) << std::endl;
    }
  }

  log.debug() << "Pre-calculations complete" << std::endl << std::endl;

  // determine number of threads to run
  // TODO write for CUDA
  /*
  if (sp.num_instances == -1) {
    if (sp.n_dbs <= 9) {
      sp.num_instances = 8;
    } else if (sp.n_dbs <= 25) {
      sp.num_instances = 16;
    } else {
      sp.num_instances = 128;
    }
  }
  */

  /* TODO deal with how results should be stored for CUDA ver
  charge_results.resize(sp.num_instances);
  energy_results.resize(sp.num_instances);
  //cpu_times.resize(sp.num_instances);
  suggested_gs_results.resize(sp.num_instances);
  */
}

FPType SimAnnealCuda::distance(const int &i, const int &j)
{
  FPType x1 = sim_params.db_locs[i].first;
  FPType y1 = sim_params.db_locs[i].second;
  FPType x2 = sim_params.db_locs[j].first;
  FPType y2 = sim_params.db_locs[j].second;
  return sqrt(pow(x1-x2, 2.0) + pow(y1-y2, 2.0));
}

FPType SimAnnealCuda::interElecPotential(const FPType &r)
{
  return constants::Q0 * sim_params.Kc * exp(-r/(sim_params.debye_length*1e-9)) / r;
}

FPType SimAnnealCuda::hopEnergyDelta(ublas::vector<int> n_in, const int &from_ind, 
    const int &to_ind)
{
  // TODO make an efficient implementation with energy delta implementation
  FPType orig_energy = systemEnergy(n_in);
  int from_state = n_in[from_ind];
  n_in[from_ind] = n_in[to_ind];
  n_in[to_ind] = from_state;
  return systemEnergy(n_in) - orig_energy;
}