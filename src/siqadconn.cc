// @file:     siqadconn.cc
// @author:   Samuel
// @created:  2017.08.23
// @editted:  2020.01.28 - Samuel
// @license:  Apache License 2.0
//
// @desc:     Convenient functions for interacting with SiQAD

#include "siqadconn.h"
#include <iostream>
#include <stdexcept>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/algorithm/string/join.hpp>
#include <boost/optional/optional.hpp>


using namespace phys;

#define debugStream \
  if (!verbose) {}  \
  else std::cout

boost::bimap<SQCommand::CommandAction, std::string> SQCommand::command_action_string;
boost::bimap<SQCommand::CommandItem, std::string> SQCommand::command_item_string;

//CONSTRUCTOR
SiQADConnector::SiQADConnector(const std::string &eng_name,
  const std::string &input_path, const std::string &output_path, 
  const bool &verbose)
  : eng_name(eng_name), input_path(input_path), output_path(output_path),
    verbose(verbose)
{
  // initialize variables
  item_tree = std::make_shared<Aggregate>();
  start_time = std::chrono::system_clock::now();
  elec_col = new ElectrodeCollection(item_tree);
  elec_poly_col = new ElectrodePolyCollection(item_tree);
  db_col = new DBCollection(item_tree);
  defect_col = new DefectCollection(item_tree);

  // read problem from input_path
  readProblem(input_path);
}

SiQADConnector::~SiQADConnector()
{
  if (output_path != "")
    writeResultsXml();
}

void SiQADConnector::setExport(std::string type, std::vector< std::pair< std::string, std::string > > &data_in)
{
  if (type == "db_loc") {
    dbl_data = data_in;
  } else if (type == "misc") {
    misc_data = data_in;
  } else {
    throw std::invalid_argument(std::string("No candidate for export type '") +
        type + std::string("' with class std::vector<std::pair<std::string, std::string>>"));
  }
}

void SiQADConnector::setExport(std::string type, std::vector< std::vector< std::string > > &data_in)
{
  if (type == "potential")
    pot_data = data_in;
  else if (type == "electrodes")
    elec_data = data_in;
  else if (type == "db_pot")
    db_pot_data = data_in;
  else if (type == "db_charge")
    db_charge_data = data_in;
  else
    throw std::invalid_argument(std::string("No candidate for export type '") +
        type + std::string("' with class std::vector<std::vector<std::string>>"));
}

void SiQADConnector::addSQCommand(SQCommand *command)
{
  export_commands.push_back(command->finalCommand());
  debugStream << "Command added to SiQADConnector: " << std::endl;
  debugStream << export_commands.back() << std::endl;
}



// FILE HANDLING
// parse problem XML, return true if successful
void SiQADConnector::readProblem(const std::string &path)
{
  debugStream << "Reading problem file: " << input_path << std::endl;

  boost::property_tree::ptree tree; // create empty property tree object
  boost::property_tree::read_xml(path, tree, boost::property_tree::xml_parser::no_comments); // parse the input file into property tree

  // parse XML

  // read program properties
  // TODO read program node

  // read simulation parameters
  debugStream << "Read simulation parameters" << std::endl;
  readSimulationParam(tree.get_child("siqad.sim_params"));

  // read layer properties
  debugStream << "Read layer properties" << std::endl;
  readLayers(tree.get_child("siqad.layers"));

  // read items
  debugStream << "Read items tree" << std::endl;
  readDesign(tree.get_child("siqad.design"), item_tree);
}

void SiQADConnector::readProgramProp(const boost::property_tree::ptree &program_prop_tree)
{
  for (boost::property_tree::ptree::value_type const &v : program_prop_tree) {
    program_props.insert(std::map<std::string, std::string>::value_type(v.first, v.second.data()));
    debugStream << "ProgramProp: Key=" << v.first << ", Value=" << program_props[v.first] << std::endl;
  }
}

void SiQADConnector::readLayers(const boost::property_tree::ptree &layer_prop_tree)
{
  // if this were structured the same way as readDesign, then only the first layer_prop subtree would be read.
  // TODO: make this more general.
  for (boost::property_tree::ptree::value_type const &v : layer_prop_tree)
    readLayerProp(v.second);
}

void SiQADConnector::readLayerProp(const boost::property_tree::ptree &layer_node)
{
  Layer lay;
  lay.name = layer_node.get<std::string>("name");
  lay.type = layer_node.get<std::string>("type");
  if (layer_node.count("role") != 0)
    lay.role = layer_node.get<std::string>("role");
  lay.zoffset = layer_node.get<float>("zoffset");
  lay.zheight = layer_node.get<float>("zheight");

  layers.push_back(lay);
  debugStream << "Retrieved layer " << lay.name << " of type " << lay.type << std::endl;
}


void SiQADConnector::readSimulationParam(const boost::property_tree::ptree &sim_params_tree)
{
  for (boost::property_tree::ptree::value_type const &v : sim_params_tree) {
    sim_params.insert(std::map<std::string, std::string>::value_type(v.first, v.second.data()));
    debugStream << "SimParam: Key=" << v.first << ", Value=" << sim_params[v.first] << std::endl;
  }
}

void SiQADConnector::readDesign(const boost::property_tree::ptree &subtree, const std::shared_ptr<Aggregate> &agg_parent)
{
  debugStream << "Beginning to read design" << std::endl;
  for (boost::property_tree::ptree::value_type const &layer_tree : subtree) {
    std::string layer_type = layer_tree.second.get<std::string>("<xmlattr>.type");
    if ((!layer_type.compare("DB"))) {
      debugStream << "Encountered node " << layer_tree.first << " with type " << layer_type << ", entering" << std::endl;
      readItemTree(layer_tree.second, agg_parent);
    } else if ( (!layer_type.compare("Electrode"))) {
      debugStream << "Encountered node " << layer_tree.first << " with type " << layer_type << ", entering" << std::endl;
      readItemTree(layer_tree.second, agg_parent);
    } else if (!layer_type.compare("Defects")) {
      debugStream << "Encountered node " << layer_tree.first << " with type " << layer_type << ", entering" << std::endl;
      readItemTree(layer_tree.second, agg_parent);
    } else {
      debugStream << "Encountered node " << layer_tree.first << " with type " << layer_type << ", no defined action for this layer. Skipping." << std::endl;
    }
  }
}

void SiQADConnector::readItemTree(const boost::property_tree::ptree &subtree, const std::shared_ptr<Aggregate> &agg_parent)
{
  for (boost::property_tree::ptree::value_type const &item_tree : subtree) {
    std::string item_name = item_tree.first;
    debugStream << "item_name: " << item_name << std::endl;
    if (!item_name.compare("aggregate")) {
      // add aggregate child to tree
      agg_parent->aggs.push_back(std::make_shared<Aggregate>());
      readItemTree(item_tree.second, agg_parent->aggs.back());
    } else if (!item_name.compare("dbdot")) {
      // add DBDot to tree
      readDBDot(item_tree.second, agg_parent);
    } else if (!item_name.compare("electrode")) {
      // add Electrode to tree
      readElectrode(item_tree.second, agg_parent);
    } else if (!item_name.compare("electrode_poly")) {
      // add Electrode to tree
      readElectrodePoly(item_tree.second, agg_parent);
    } else if (!item_name.compare("defect")) {
      // add Defect to tree
      readDefect(item_tree.second, agg_parent);
    } else {
      debugStream << "Encountered unknown item node: " << item_tree.first << std::endl;
    }
  }
}

void SiQADConnector::readElectrode(const boost::property_tree::ptree &subtree, const std::shared_ptr<Aggregate> &agg_parent)
{
  double x1, x2, y1, y2, pixel_per_angstrom, potential, phase, angle, pot_offset;
  int layer_id, electrode_type=0, net;
  // read values from XML stream
  layer_id = subtree.get<int>("layer_id");
  angle = subtree.get<double>("angle");
  potential = subtree.get<double>("property_map.potential.val");
  pot_offset = subtree.get<double>("property_map.pot_offset.val");
  phase = subtree.get<double>("property_map.phase.val");
  std::string electrode_type_s = subtree.get<std::string>("property_map.type.val");
  if (!electrode_type_s.compare("fixed")){
    electrode_type = 0;
  } else if (!electrode_type_s.compare("clocked")) {
    electrode_type = 1;
  }
  net = subtree.get<int>("property_map.net.val");
  pixel_per_angstrom = subtree.get<double>("pixel_per_angstrom");
  x1 = subtree.get<double>("dim.<xmlattr>.x1");
  x2 = subtree.get<double>("dim.<xmlattr>.x2");
  y1 = subtree.get<double>("dim.<xmlattr>.y1");
  y2 = subtree.get<double>("dim.<xmlattr>.y2");
  agg_parent->elecs.push_back(std::make_shared<Electrode>(layer_id,x1,x2,y1,y2,potential,pot_offset,phase,electrode_type,pixel_per_angstrom,net,angle));

  debugStream << "Electrode created with x1=" << agg_parent->elecs.back()->x1 << ", y1=" << agg_parent->elecs.back()->y1 <<
    ", x2=" << agg_parent->elecs.back()->x2 << ", y2=" << agg_parent->elecs.back()->y2 <<
    ", potential=" << agg_parent->elecs.back()->potential <<
    ", pot_offset=" << agg_parent->elecs.back()->pot_offset << std::endl;
}

void SiQADConnector::readElectrodePoly(const boost::property_tree::ptree &subtree, const std::shared_ptr<Aggregate> &agg_parent)
{
  double pixel_per_angstrom, potential, phase;
  std::vector<std::pair<double, double>> vertices;
  int layer_id, electrode_type=0, net;
  // read values from XML stream
  layer_id = subtree.get<int>("layer_id");
  potential = subtree.get<double>("property_map.potential.val");
  phase = subtree.get<double>("property_map.phase.val");
  std::string electrode_type_s = subtree.get<std::string>("property_map.type.val");
  if (!electrode_type_s.compare("fixed")){
    electrode_type = 0;
  } else if (!electrode_type_s.compare("clocked")) {
    electrode_type = 1;
  }
  pixel_per_angstrom = subtree.get<double>("pixel_per_angstrom");
  net = subtree.get<int>("property_map.net.val");
  //cycle through the vertices
  std::pair<double, double> point;
  for (auto val: subtree) {
    if(val.first == "vertex") {
      double x = std::stod(val.second.get_child("<xmlattr>.x").data());
      double y = std::stod(val.second.get_child("<xmlattr>.y").data());
      point = std::make_pair(x, y);
      vertices.push_back(point);
    }
  }
  agg_parent->elec_polys.push_back(std::make_shared<ElectrodePoly>(layer_id,vertices,potential,phase,electrode_type,pixel_per_angstrom,net));

  debugStream << "ElectrodePoly created with " << agg_parent->elec_polys.back()->vertices.size() <<
    " vertices, potential=" << agg_parent->elec_polys.back()->potential << std::endl;
}

void SiQADConnector::readDBDot(const boost::property_tree::ptree &subtree, const std::shared_ptr<Aggregate> &agg_parent)
{
  float x, y;
  int n, m, l;

  // read x and y physical locations
  x = subtree.get<float>("physloc.<xmlattr>.x");
  y = subtree.get<float>("physloc.<xmlattr>.y");

  // read n, m and l lattice coordinates
  n = subtree.get<int>("latcoord.<xmlattr>.n");
  m = subtree.get<int>("latcoord.<xmlattr>.m");
  l = subtree.get<int>("latcoord.<xmlattr>.l");

  agg_parent->dbs.push_back(std::make_shared<DBDot>(x, y, n, m, l));

  debugStream << "DBDot created with x=" << agg_parent->dbs.back()->x
            << ", y=" << agg_parent->dbs.back()->y
            << ", n=" << agg_parent->dbs.back()->n
            << ", m=" << agg_parent->dbs.back()->m
            << ", l=" << agg_parent->dbs.back()->l
            << std::endl;
}

void SiQADConnector::readDefect(const boost::property_tree::ptree &subtree, const std::shared_ptr<Aggregate> &agg_parent)
{
  bool has_eucl = false;
  float x, y, z;
  bool has_lat_coord = false;
  int n, m, l;
  int w, h;
  float charge;
  float eps_r, lambda_tf;

  // read anchor lat coord and width/height
  boost::optional<const boost::property_tree::ptree &> child = subtree.get_child_optional("latcoord");
  if (child) {
    debugStream << "******* has lat coord in siqadconn **********" << std::endl;
    has_lat_coord = true;
    n = subtree.get<int>("latcoord.<xmlattr>.n");
    m = subtree.get<int>("latcoord.<xmlattr>.m");
    l = subtree.get<int>("latcoord.<xmlattr>.l");
    w = subtree.get<int>("latdim.<xmlattr>.w");
    h = subtree.get<int>("latdim.<xmlattr>.h");
  }

  // read x and y physical locations
  child = subtree.get_child_optional("physloc");
  if (child) {
    debugStream << "******* has eucl in siqadconn **********" << std::endl;
    has_eucl = true;
    x = subtree.get<float>("physloc.<xmlattr>.x");
    y = subtree.get<float>("physloc.<xmlattr>.y");
    z = subtree.get<float>("physloc.<xmlattr>.z");
  }

  // read electrostatic props
  charge = subtree.get<float>("coulomb.<xmlattr>.charge");
  eps_r = subtree.get<float>("coulomb.<xmlattr>.eps_r");
  lambda_tf = subtree.get<float>("coulomb.<xmlattr>.lambda_tf");

  if (has_lat_coord) {
    agg_parent->defects.push_back(std::make_shared<Defect>(n, m, l, w, h, charge, eps_r, lambda_tf));
  } else if (has_eucl) {
    agg_parent->defects.push_back(std::make_shared<Defect>(x, y, z, charge, eps_r, lambda_tf));
  }

  debugStream << "Defect created with x=" << agg_parent->defects.back()->x
            << ", y=" << agg_parent->defects.back()->y
            << ", z=" << agg_parent->defects.back()->z
            << ", charge=" << agg_parent->defects.back()->charge
            << ", eps_r=" << agg_parent->defects.back()->eps_r
            << ", lambda_tf=" << agg_parent->defects.back()->lambda_tf
            << std::endl;
}

void SiQADConnector::writeResultsXml()
{
  if (output_path == "")
    throw std::invalid_argument("Output path not set.");

  boost::property_tree::ptree node_root;

  debugStream << "Write results to XML..." << std::endl;

  // eng_info
  node_root.add_child("eng_info", engInfoPropertyTree());

  // sim_params
  node_root.add_child("sim_params", simParamsPropertyTree());

  // DB locations
  if (!dbl_data.empty())
    node_root.add_child("physloc", dbLocPropertyTree());

  // DB electron distributions
  if(!db_charge_data.empty())
    node_root.add_child("elec_dist", dbChargePropertyTree());

  // electrode
  if (!elec_data.empty())
    node_root.add_child("electrode", electrodePropertyTree());

  // electric potentials
  if (!pot_data.empty())
    node_root.add_child("potential_map", potentialPropertyTree());

  // potentials at db locations
  if (!db_pot_data.empty())
    node_root.add_child("dbdots", dbPotentialPropertyTree());

  // misc outputs
  if (!misc_data.empty())
    node_root.add_child("misc", miscPropertyTree());

  // SQCommands
  if (!export_commands.empty()) {
    debugStream << "export commands not empty, starting to fill them in." << std::endl;
    node_root.add_child("sqcommands", sqCommandsPropertyTree());
  }

  // write full tree to file
  boost::property_tree::ptree tree;
  tree.add_child("sim_out", node_root);
  boost::property_tree::write_xml(output_path, tree, std::locale(), boost::property_tree::xml_writer_make_settings<std::string>(' ',4));

  debugStream << "Write to XML complete." << std::endl;
}

boost::property_tree::ptree SiQADConnector::engInfoPropertyTree()
{
  boost::property_tree::ptree node_eng_info;
  node_eng_info.put("engine", eng_name);
  node_eng_info.put("version", "TBD"); // TODO real version
  node_eng_info.put("return_code", std::to_string(return_code).c_str());

  //get timing information
  end_time = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end_time-start_time;
  std::time_t end = std::chrono::system_clock::to_time_t(end_time);
  char* end_c_str = std::ctime(&end);
  *std::remove(end_c_str, end_c_str+strlen(end_c_str), '\n') = '\0'; // removes _all_ new lines from the cstr
  node_eng_info.put("timestamp", end_c_str);
  node_eng_info.put("time_elapsed_s", std::to_string(elapsed_seconds.count()).c_str());

  return node_eng_info;
}

boost::property_tree::ptree SiQADConnector::simParamsPropertyTree()
{
  boost::property_tree::ptree node_sim_params;
  for (std::pair<std::string, std::string> param : sim_params)
    node_sim_params.put(param.first, param.second);
  return node_sim_params;
}

boost::property_tree::ptree SiQADConnector::dbLocPropertyTree()
{
  boost::property_tree::ptree node_physloc;
  for (unsigned int i = 0; i < dbl_data.size(); i++){
    boost::property_tree::ptree node_dbdot;
    node_dbdot.put("<xmlattr>.x", dbl_data[i].first.c_str());
    node_dbdot.put("<xmlattr>.y", dbl_data[i].second.c_str());
    node_physloc.add_child("dbdot", node_dbdot);
  }
  return node_physloc;
}

boost::property_tree::ptree SiQADConnector::dbChargePropertyTree()
{
  boost::property_tree::ptree node_elec_dist;
  for (unsigned int i = 0; i < db_charge_data.size(); i++){
    boost::property_tree::ptree node_dist;
    node_dist.put("", db_charge_data[i][0]);
    node_dist.put("<xmlattr>.energy", db_charge_data[i][1]);
    node_dist.put("<xmlattr>.count", db_charge_data[i][2]);
    node_dist.put("<xmlattr>.physically_valid", db_charge_data[i][3]);
    std::string state_count = "2";    // 0 for DB0 and 1 for DB-
    if (db_charge_data[i].size() > 4)
      state_count = db_charge_data[i][4]; // if 3, - for DB-, 0 for DB0, + for DB+
    node_dist.put("<xmlattr>.state_count", state_count);
    node_elec_dist.add_child("dist", node_dist);
  }
  return node_elec_dist;
}

boost::property_tree::ptree SiQADConnector::electrodePropertyTree()
{
  boost::property_tree::ptree node_electrode;
  for (unsigned int i = 0; i < elec_data.size(); i++){
    boost::property_tree::ptree node_dim;
    node_dim.put("<xmlattr>.x1", elec_data[i][0].c_str());
    node_dim.put("<xmlattr>.y1", elec_data[i][1].c_str());
    node_dim.put("<xmlattr>.x2", elec_data[i][2].c_str());
    node_dim.put("<xmlattr>.y2", elec_data[i][3].c_str());
    node_electrode.add_child("dim", node_dim);
    boost::property_tree::ptree node_pot;
    node_pot.put("", elec_data[i][4].c_str());
    node_electrode.add_child("potential", node_pot);
  }
  return node_electrode;
}

boost::property_tree::ptree SiQADConnector::potentialPropertyTree()
{
  boost::property_tree::ptree node_potential_map;
  for (unsigned int i = 0; i < pot_data.size(); i++){
    boost::property_tree::ptree node_potential_val;
    node_potential_val.put("<xmlattr>.x", pot_data[i][0].c_str());
    node_potential_val.put("<xmlattr>.y", pot_data[i][1].c_str());
    node_potential_val.put("<xmlattr>.val", pot_data[i][2].c_str());
    node_potential_map.add_child("potential_val", node_potential_val);
  }
  return node_potential_map;
}

boost::property_tree::ptree SiQADConnector::dbPotentialPropertyTree()
{
  boost::property_tree::ptree node_dbdots;
  for (unsigned int i = 0; i < db_pot_data.size(); i++){
    boost::property_tree::ptree node_dbdot;
    boost::property_tree::ptree node_physloc;
    boost::property_tree::ptree node_db_step;
    node_db_step.put("", db_pot_data[i][0].c_str());
    node_dbdot.add_child("step", node_db_step);
    node_physloc.put("<xmlattr>.x", db_pot_data[i][1].c_str());
    node_physloc.put("<xmlattr>.y", db_pot_data[i][2].c_str());
    node_dbdot.add_child("physloc", node_physloc);
    boost::property_tree::ptree node_db_pot;
    node_db_pot.put("", db_pot_data[i][3].c_str());
    node_dbdot.add_child("potential", node_db_pot);
    node_dbdots.add_child("dbdot", node_dbdot);
  }
  return node_dbdots;
}

boost::property_tree::ptree SiQADConnector::sqCommandsPropertyTree()
{
  boost::property_tree::ptree node_sqcommands;
  for (unsigned int i = 0; i < export_commands.size(); i++) {
    debugStream << "command " << i << ": ";
    debugStream << export_commands.at(i) << std::endl;
    boost::property_tree::ptree node_sqc;
    node_sqc.put("", export_commands.at(i).c_str());
    //std::cout << export_commands.at(i)->finalCommand() << std::endl;
    node_sqcommands.add_child("sqc", node_sqc);
  }
  return node_sqcommands;
}

boost::property_tree::ptree SiQADConnector::miscPropertyTree()
{
  boost::property_tree::ptree node_misc;
  for (unsigned int i=0; i < misc_data.size(); i++) {
    boost::property_tree::ptree node_misc_item;
    node_misc_item.put("", misc_data[i].second);
    node_misc.add_child(misc_data[i].first, node_misc_item);
  }
  return node_misc;
}

//DB ITERATOR

DBIterator::DBIterator(std::shared_ptr<Aggregate> root, bool begin)
{
  if(begin){
    // keep finding deeper aggregates until one that contains dbs is found
    while(root->dbs.empty() && !root->aggs.empty()) {
      push(root);
      root = root->aggs.front();
    }
    push(root);
  }
  else{
    db_iter = root->dbs.cend();
  }
}

DBIterator& DBIterator::operator++()
{
  // exhaust the current Aggregate DBs first
  if(db_iter != curr->dbs.cend())
    return ++db_iter != curr->dbs.cend() ? *this : ++(*this);

  // if available, push the next aggregate onto the stack
  if(agg_stack.top().second != curr->aggs.cend()){
    push(*agg_stack.top().second);
    return db_iter != curr->dbs.cend() ? *this : ++(*this);
  }

  // aggregate is complete, pop off stack
  pop();
  return agg_stack.size() == 0 ? *this : ++(*this);
}

void DBIterator::push(std::shared_ptr<Aggregate> agg)
{
  if(!agg_stack.empty())
    ++agg_stack.top().second;
  agg_stack.push(std::make_pair(agg, agg->aggs.cbegin()));
  db_iter = agg->dbs.cbegin();
  curr = agg;
}

void DBIterator::pop()
{
  agg_stack.pop();              // pop complete aggregate off stack
  if(agg_stack.size() > 0){
    curr = agg_stack.top().first; // update current to new top
    db_iter = curr->dbs.cend();   // don't reread dbs
  }
}


// FIXED CHARGE ITERATOR

DefectIterator::DefectIterator(std::shared_ptr<Aggregate> root, bool begin)
{
  if(begin){
    // keep finding deeper aggregates until one that contains dbs is found
    while(root->defects.empty() && !root->aggs.empty()) {
      push(root);
      root = root->aggs.front();
    }
    push(root);
  }
  else{
    defect_iter = root->defects.cend();
  }
}

DefectIterator& DefectIterator::operator++()
{
  // exhaust the current Aggregate DBs first
  if(defect_iter != curr->defects.cend())
    return ++defect_iter != curr->defects.cend() ? *this : ++(*this);

  // if available, push the next aggregate onto the stack
  if(agg_stack.top().second != curr->aggs.cend()){
    push(*agg_stack.top().second);
    return defect_iter != curr->defects.cend() ? *this : ++(*this);
  }

  // aggregate is complete, pop off stack
  pop();
  return agg_stack.size() == 0 ? *this : ++(*this);
}

void DefectIterator::push(std::shared_ptr<Aggregate> agg)
{
  if(!agg_stack.empty())
    ++agg_stack.top().second;
  agg_stack.push(std::make_pair(agg, agg->aggs.cbegin()));
  defect_iter = agg->defects.cbegin();
  curr = agg;
}

void DefectIterator::pop()
{
  agg_stack.pop();              // pop complete aggregate off stack
  if(agg_stack.size() > 0){
    curr = agg_stack.top().first; // update current to new top
    defect_iter = curr->defects.cend();   // don't reread dbs
  }
}



// ELEC ITERATOR
ElecIterator::ElecIterator(std::shared_ptr<Aggregate> root, bool begin)
{
  if(begin){
    // keep finding deeper aggregates until one that contains dbs is found
    while(root->elecs.empty() && !root->aggs.empty()) {
      push(root);
      root = root->aggs.front();
    }
    push(root);
  }
  else{
    elec_iter = root->elecs.cend();
  }
}

ElecIterator& ElecIterator::operator++()
{
  // exhaust the current Aggregate DBs first
  if(elec_iter != curr->elecs.cend())
    return ++elec_iter != curr->elecs.cend() ? *this : ++(*this);

  // if available, push the next aggregate onto the stack
  if(agg_stack.top().second != curr->aggs.cend()){
    push(*agg_stack.top().second);
    return elec_iter != curr->elecs.cend() ? *this : ++(*this);
  }

  // aggregate is complete, pop off stack
  pop();
  return agg_stack.size() == 0 ? *this : ++(*this);
}

void ElecIterator::push(std::shared_ptr<Aggregate> agg)
{
  if(!agg_stack.empty())
    ++agg_stack.top().second;
  agg_stack.push(std::make_pair(agg, agg->aggs.cbegin()));
  elec_iter = agg->elecs.cbegin();
  curr = agg;
}

void ElecIterator::pop()
{
  agg_stack.pop();              // pop complete aggregate off stack
  if(agg_stack.size() > 0){
    curr = agg_stack.top().first; // update current to new top
    elec_iter = curr->elecs.cend();   // don't reread dbs
  }
}

// ELECPOLY ITERATOR
ElecPolyIterator::ElecPolyIterator(std::shared_ptr<Aggregate> root, bool begin)
{
  if(begin){
    // keep finding deeper aggregates until one that contains dbs is found
    while(root->elec_polys.empty() && !root->aggs.empty()) {
      push(root);
      root = root->aggs.front();
    }
    push(root);
  }
  else{
    elec_poly_iter = root->elec_polys.cend();
  }
}

ElecPolyIterator& ElecPolyIterator::operator++()
{
  // exhaust the current Aggregate DBs first
  if(elec_poly_iter != curr->elec_polys.cend())
    return ++elec_poly_iter != curr->elec_polys.cend() ? *this : ++(*this);

  // if available, push the next aggregate onto the stack
  if(agg_stack.top().second != curr->aggs.cend()){
    push(*agg_stack.top().second);
    return elec_poly_iter != curr->elec_polys.cend() ? *this : ++(*this);
  }

  // aggregate is complete, pop off stack
  pop();
  return agg_stack.size() == 0 ? *this : ++(*this);
}

void ElecPolyIterator::push(std::shared_ptr<Aggregate> agg)
{
  if(!agg_stack.empty())
    ++agg_stack.top().second;
  agg_stack.push(std::make_pair(agg, agg->aggs.cbegin()));
  elec_poly_iter = agg->elec_polys.cbegin();
  curr = agg;
}

void ElecPolyIterator::pop()
{
  agg_stack.pop();              // pop complete aggregate off stack
  if(agg_stack.size() > 0){
    curr = agg_stack.top().first; // update current to new top
    elec_poly_iter = curr->elec_polys.cend();   // don't reread dbs
  }
}

// AGGREGATE
int Aggregate::size()
{
  int n_elecs=elecs.size();
  if(!aggs.empty())
    for(auto agg : aggs)
      n_elecs += agg->size();
  return n_elecs;
}



// SQCommand implementation

std::string SQCommand::commandItemString(CommandItem t_item)
{
  if (command_item_string.empty())
    constructStatics();
  return command_item_string.left.at(t_item);
}

std::string SQCommand::commandActionString(CommandAction t_action)
{
  if (command_action_string.empty())
    constructStatics();
  return command_action_string.left.at(t_action);
}

SQCommand::CommandItem SQCommand::commandItemEnum(std::string str_item)
{
  if (command_item_string.empty())
    constructStatics();
  return command_item_string.right.at(str_item);
}

SQCommand::CommandAction SQCommand::commandActionEnum(std::string str_action)
{
  if (command_action_string.empty())
    constructStatics();
  return command_action_string.right.at(str_action);
}

std::string SQCommand::finalCommand()
{
  /* TODO fill this in when more command actions are implemented
  switch (action) {
    case Add:
      return addActionCommand();
    default:
      return "";
  }
  */
  return addActionCommand();
}

std::string SQCommand::addActionCommand()
{
  // store the command in components and join them with spaces at the end
  std::vector<std::string> command_components;

  std::vector<std::string> command_args;

  switch (item) {
    case Aggregate:
    {
      AggregateCommand *agg_cmd = static_cast<AggregateCommand*>(this);
      command_args = agg_cmd->addActionArguments();
      break;
    }
    default:
      break;
  }

  command_components.push_back(commandActionString(commandAction()));
  command_components.push_back(commandItemString(commandItem()));
  command_components.insert(command_components.end(), command_args.begin(), command_args.end());

  return boost::algorithm::join(command_components, " ");
}

void SQCommand::constructStatics()
{
  typedef boost::bimap<CommandAction, std::string> ActionStringMap;
  typedef ActionStringMap::value_type ASMEntry;
  command_action_string.insert(ASMEntry(Add, "add"));
  command_action_string.insert(ASMEntry(Remove, "remove"));
  command_action_string.insert(ASMEntry(Echo, "echo"));
  command_action_string.insert(ASMEntry(Run, "run"));
  command_action_string.insert(ASMEntry(Move, "move"));

  typedef boost::bimap<CommandItem, std::string> ItemStringMap;
  typedef ItemStringMap::value_type ISMEntry;
  command_item_string.insert(ISMEntry(NoItem, "NoItem"));
  command_item_string.insert(ISMEntry(DBDot, "DBDot"));
  command_item_string.insert(ISMEntry(Electrode, "Electrode"));
  command_item_string.insert(ISMEntry(Aggregate, "Aggregate"));
}


// Aggregate command
std::vector<std::string> AggregateCommand::addActionArguments()
{
  // store the command in components and join them with spaces at the end
  std::vector<std::string> command_args;

  command_args.push_back(layer == -1 ? "auto" : std::to_string(layer));

  for (std::pair<float, float> db_loc : db_locs) {
    command_args.push_back("(" + std::to_string(db_loc.first) + " "
        + std::to_string(db_loc.second) + ")");
  }

  return command_args;
}
