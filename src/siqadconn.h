// @file:     siqadconn.h
// @author:   Samuel
// @created:  2017.08.23
// @editted:  2020.01.28 - Samuel
// @license:  Apache License 2.0
//
// @desc:     Convenient functions for interacting with SiQAD including
//            setting expected problem parameters, parsing problem files,
//            writing result files, etc. Use of the class is recommended, but
//            ultimately optional as devs may want to implement their own
//            I/O with SiQAD

#ifndef _SIQAD_PLUGIN_CONNECTOR_H_
#define _SIQAD_PLUGIN_CONNECTOR_H_


#include <stack>
#include <memory>
#include <map>
#include <iostream>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/bimap.hpp>
#include <string>
#include <vector>
#include <ctime>
#include <chrono>

namespace phys{
  //namespace bpt = boost::property_tree;

  // forward declaration
  struct Layer;
  struct DBDot;
  class DBIterator;
  class DBCollection;
  struct Defect;
  class DefectIterator;
  class DefectCollection;
  struct Electrode;
  class ElecIterator;
  class ElectrodeCollection;
  struct ElectrodePoly;
  class ElecPolyIterator;
  class ElectrodePolyCollection;
  struct Aggregate;

  class SQCommand;
  class AggregateCommand;

  typedef std::vector< std::shared_ptr<DBDot> >::const_iterator DBIter;
  typedef std::vector< std::shared_ptr<Defect> >::const_iterator DefectIter;
  typedef std::vector< std::shared_ptr<Electrode> >::const_iterator ElecIter;
  typedef std::vector< std::shared_ptr<ElectrodePoly> >::const_iterator ElecPolyIter;
  typedef std::vector< std::shared_ptr<Aggregate> >::const_iterator AggIter;

  // SiQAD connector class
  class SiQADConnector
  {
  public:
    // CONSTRUCTOR
    SiQADConnector(const std::string &eng_name, const std::string &input_path,
        const std::string &output_path="", const bool &verbose=false);
    // DESTRUCTOR
    ~SiQADConnector();

    // Write results to the provided output_path
    void writeResultsXml();


    // EXPORTING

    // Set export type and variable
    void setExport(std::string type, std::vector< std::pair< std::string, std::string > > &data_in);
    void setExport(std::string type, std::vector< std::vector< std::string > > &data_in);

    // Add SQCommand export
    void addSQCommand(SQCommand *);


    // SIMULATION PARAMETERS

    // Checks if a parameter with the given key exists.
    bool parameterExists(const std::string &key) {return sim_params.find(key) != sim_params.end();}

    // Get the parameter with the given key.
    std::string getParameter(const std::string &key) {return sim_params.find(key) != sim_params.end() ? sim_params.at(key) : "";}

    std::vector<Layer> getLayers(void){return layers;}
    std::map<std::string, std::string> getAllParameters(void){return sim_params;}


    // ITERABLE COLLECTIONS

    // Return pointer to DB collection, which allows iteration through DBs
    // across all aggregate levels.
    DBCollection* dbCollection() {return db_col;}

    // Return pointer to defect collection, which allows iteration through
    // defects across all defect layers.
    DefectCollection* defectCollection() {return defect_col;}

    // Return pointer to Electrode collection, which allows iteration through
    // electrodes across all electrode layers.
    ElectrodeCollection* electrodeCollection() {return elec_col;}

    // Return pointer to Electrode collection, which allows iteration through
    // electrodes across all electrode layers.
    ElectrodePolyCollection* electrodePolyCollection() {return elec_poly_col;}


    // Misc Accessors
    std::string inputPath(){return input_path;}

    void setOutputPath(std::string path){output_path = path;}
    std::string outputPath(){return output_path;}

  private:

    // Read the problem file
    void readProblem(const std::string &path);

    // Read program properties
    void readProgramProp(const boost::property_tree::ptree &);

    // Read layer properties
    void readLayers(const boost::property_tree::ptree &);
    void readLayerProp(const boost::property_tree::ptree &);

    // Read simulation parameters
    void readSimulationParam(const boost::property_tree::ptree &);

    // Read design
    void readDesign(const boost::property_tree::ptree &, const std::shared_ptr<Aggregate> &);
    void readItemTree(const boost::property_tree::ptree &, const std::shared_ptr<Aggregate> &);
    void readElectrode(const boost::property_tree::ptree &, const std::shared_ptr<Aggregate> &);
    void readElectrodePoly(const boost::property_tree::ptree &, const std::shared_ptr<Aggregate> &);
    void readDBDot(const boost::property_tree::ptree &, const std::shared_ptr<Aggregate> &);
    void readDefect(const boost::property_tree::ptree &, const std::shared_ptr<Aggregate> &);

    // Generate property trees for writing
    boost::property_tree::ptree engInfoPropertyTree();
    boost::property_tree::ptree simParamsPropertyTree();
    boost::property_tree::ptree dbLocPropertyTree();
    boost::property_tree::ptree dbChargePropertyTree();
    boost::property_tree::ptree electrodePropertyTree();
    // boost::property_tree::ptree electrodePolyPropertyTree();
    boost::property_tree::ptree potentialPropertyTree();
    boost::property_tree::ptree dbPotentialPropertyTree(); // TODO fix up this function, a lot of redundant information
    boost::property_tree::ptree sqCommandsPropertyTree();
    boost::property_tree::ptree miscPropertyTree();

    // Engine properties
    std::string eng_name;                 // name of simulation engine
    std::string input_path;               // path to problem file
    std::string output_path;              // path to result export

    // Iterable collections
    ElectrodeCollection* elec_col;
    DBCollection* db_col;
    DefectCollection* defect_col;
    ElectrodePolyCollection* elec_poly_col;

    // Retrieved items and properties
    std::map<std::string, std::string> program_props; // SiQAD properties
    std::shared_ptr<Aggregate> item_tree;             // all physical items
    std::vector<Layer> layers;                        // layers
    std::map<std::string, std::string> sim_params;    // simulation parameters

    // Exportable data
    std::vector< std::vector<std::string> > pot_data;
    std::vector< std::vector<std::string> > db_pot_data;
    // std::vector<std::vector<std::vector<std::string>>> db_pot_history;
    std::vector< std::vector<std::string> > elec_data;
    std::vector< std::pair<std::string, std::string> > dbl_data;  // pair of location x and y
    std::vector< std::vector<std::string> > db_charge_data;       // pair of elec dist and energy
    std::vector<std::string> export_commands;                   // SQCommands to be exported
    std::vector< std::pair<std::string, std::string> > misc_data; // misc data output that is ignored by SiQAD, first string for element name and second string for value

    // Runtime information
    bool verbose;
    int return_code=0;
    std::chrono::time_point<std::chrono::system_clock> start_time;
    std::chrono::time_point<std::chrono::system_clock> end_time;
  };


  // layer struct
  struct Layer {
    Layer(std::string name, std::string type, float zoffset, float zheight)
      : name(name), type(type), zoffset(zoffset), zheight(zheight) {};
    Layer() {};
    std::string name;   // layer name
    std::string type;   // layer type
    std::string role;   // layer role
    float zoffset=0;    // layer offset from lattice surface
    float zheight=0;    // layer thickness
  };

  // dangling bond
  struct DBDot {
    float x,y;  // physical location in angstroms
    int n,m,l;  // location in lattice coordinates
    DBDot(float in_x, float in_y, int n, int m, int l)
      : x(in_x), y(in_y), n(n), m(m), l(l) {};
  };

  // a constant iterator that iterates through all dangling bonds in the problem
  class DBIterator
  {
  public:
    explicit DBIterator(std::shared_ptr<Aggregate> root, bool begin=true);

    DBIterator& operator++(); // recursive part here
    bool operator==(const DBIterator &other) {return other.db_iter == db_iter;}
    bool operator!=(const DBIterator &other) {return other.db_iter != db_iter;}
    std::shared_ptr<DBDot> operator*() const {return *db_iter;}

    void setCollection(DBCollection *coll) {collection = coll;}
    DBCollection *collection; // needed for python wrapper
  private:

    DBIter db_iter;                   // points to the current DB
    std::shared_ptr<Aggregate> curr;  // current working Aggregate
    std::stack< std::pair<std::shared_ptr<Aggregate>, AggIter> > agg_stack;

    // add a new aggregate pair to the stack
    void push(std::shared_ptr<Aggregate> agg);

    // pop the aggregate stack
    void pop();
  };

  class DBCollection
  {
  public:
    DBCollection(std::shared_ptr<Aggregate> db_tree_in)
      : db_tree_inner(db_tree_in) {};
    DBIterator begin() {return DBIterator(db_tree_inner);}
    DBIterator end() {return DBIterator(db_tree_inner, false);}
    std::shared_ptr<Aggregate> db_tree_inner;
  };


  // fixed charges
  struct Defect {
    bool has_lat_coord;
    int n, m, l;    // physical anchor in lattice coordinates
    int w, h;       // lattice coordinate width and height
    bool has_eucl;
    float x, y, z;  // physical location in angstrom
    float charge;   // contained electron charge
    float eps_r, lambda_tf; // relative permittivity and screening length for this fixed charge
    Defect(float t_x, float t_y, float t_z, float t_charge, float t_eps_r, float t_lambda_tf)
      : has_eucl(true), has_lat_coord(false), x(t_x), y(t_y), z(t_z), charge(t_charge), eps_r(t_eps_r),
        lambda_tf(t_lambda_tf) {};
    Defect(int t_n, int t_m, int t_l, int t_w, int t_h, float t_charge, float t_eps_r, float t_lambda_tf)
      : has_eucl(false), has_lat_coord(true), n(t_n), m(t_m), l(t_l), w(t_w), h(t_h), charge(t_charge),
        eps_r(t_eps_r), lambda_tf(t_lambda_tf) {};
  };

  // a constant iterator that iterates through all fixed charges in the problem
  class DefectIterator
  {
  public:
    explicit DefectIterator(std::shared_ptr<Aggregate> root, bool begin=true);
    DefectIterator& operator++();
    bool operator==(const DefectIterator &other) {return other.defect_iter == defect_iter;}
    bool operator!=(const DefectIterator &other) {return other.defect_iter != defect_iter;}
    std::shared_ptr<Defect> operator*() const {return *defect_iter;}

    void setCollection(DefectCollection *coll) {collection = coll;}
    DefectCollection *collection;  // needed for python wrapper
  
  private:
    DefectIter defect_iter;    // points tot he current fixed charge
    std::shared_ptr<Aggregate> curr;      // current working Aggregate
    std::stack< std::pair<std::shared_ptr<Aggregate>, AggIter> > agg_stack;

    // add a new aggregate pair to the stack
    void push(std::shared_ptr<Aggregate> agg);

    // pop the aggregate stack
    void pop();
  };

  class DefectCollection
  {
  public:
    DefectCollection(std::shared_ptr<Aggregate> defect_in)
      : defect_tree_inner(defect_in) {};
    DefectIterator begin() {return DefectIterator(defect_tree_inner);}
    DefectIterator end() {return DefectIterator(defect_tree_inner, false);}
    std::shared_ptr<Aggregate> defect_tree_inner;
  };



  // electrode_poly
  struct ElectrodePoly {
    int layer_id;
    std::vector< std::pair<double, double> > vertices;  // vertex points
    double potential;
    double phase;
    int electrode_type;
    int net;
    double pixel_per_angstrom;
    ElectrodePoly(int in_layer_id, std::vector< std::pair<double, double> > in_vertices, \
              double in_potential, double in_phase, int in_electrode_type, double in_pixel_per_angstrom, int in_net)
      : layer_id(in_layer_id), vertices(in_vertices), \
        potential(in_potential), phase(in_phase), electrode_type(in_electrode_type), \
        net(in_net), pixel_per_angstrom(in_pixel_per_angstrom) {};
  };

  class ElecPolyIterator
  {
  public:
    explicit ElecPolyIterator(std::shared_ptr<Aggregate> root, bool begin=true);
    ElecPolyIterator& operator++(); // recursive part here
    bool operator==(const ElecPolyIterator &other) {return other.elec_poly_iter == elec_poly_iter;}
    bool operator!=(const ElecPolyIterator &other) {return other.elec_poly_iter != elec_poly_iter;}
    std::shared_ptr<ElectrodePoly> operator*() const {return *elec_poly_iter;}

    void setCollection(ElectrodePolyCollection *coll) {collection = coll;}
    ElectrodePolyCollection *collection; // needed for python wrapper
  private:
    ElecPolyIter elec_poly_iter;               // points to the current electrode
    std::shared_ptr<Aggregate> curr;  // current working Aggregate
    std::stack<std::pair< std::shared_ptr<Aggregate>, AggIter> > agg_stack;
    // add a new aggregate pair to the stack
    void push(std::shared_ptr<Aggregate> agg);
    // pop the aggregate stack
    void pop();
  };

  class ElectrodePolyCollection
  {
  public:
    ElectrodePolyCollection(std::shared_ptr<Aggregate> elec_poly_tree_in)
      : elec_poly_tree_inner(elec_poly_tree_in) {};
    ElecPolyIterator begin() {return ElecPolyIterator(elec_poly_tree_inner);}
    ElecPolyIterator end() {return ElecPolyIterator(elec_poly_tree_inner, false);}
    std::shared_ptr<Aggregate> elec_poly_tree_inner;
  };

  // electrode
  struct Electrode {
    int layer_id;
    double x1,x2,y1,y2;      // pixel location of electrode.
    double potential;  // voltage that the electrode is set to
    double pot_offset;
    double phase;
    int electrode_type;
    int net;
    double angle;
    double pixel_per_angstrom;
    Electrode(int in_layer_id, double in_x1, double in_x2, double in_y1, double in_y2, \
              double in_potential, double in_pot_offset, double in_phase, int in_electrode_type, double in_pixel_per_angstrom, int in_net, double in_angle)
      : layer_id(in_layer_id), x1(in_x1), x2(in_x2), y1(in_y1), y2(in_y2), \
        potential(in_potential), pot_offset(in_pot_offset), phase(in_phase), electrode_type(in_electrode_type), \
        net(in_net), angle(in_angle), pixel_per_angstrom(in_pixel_per_angstrom) {};
  };

  class ElecIterator
  {
  public:
    explicit ElecIterator(std::shared_ptr<Aggregate> root, bool begin=true);
    ElecIterator& operator++(); // recursive part here
    bool operator==(const ElecIterator &other) {return other.elec_iter == elec_iter;}
    bool operator!=(const ElecIterator &other) {return other.elec_iter != elec_iter;}
    std::shared_ptr<Electrode> operator*() const {return *elec_iter;}

    void setCollection(ElectrodeCollection *coll) {collection = coll;}
    ElectrodeCollection *collection; // needed for python wrapper
  private:
    ElecIter elec_iter;               // points to the current electrode
    std::shared_ptr<Aggregate> curr;  // current working Aggregate
    std::stack< std::pair<std::shared_ptr<Aggregate>, AggIter> > agg_stack;
    // add a new aggregate pair to the stack
    void push(std::shared_ptr<Aggregate> agg);
    // pop the aggregate stack
    void pop();
  };

  class ElectrodeCollection
  {
  public:
    ElectrodeCollection(std::shared_ptr<Aggregate> elec_tree_in)
      : elec_tree_inner(elec_tree_in) {};
    ElecIterator begin() {return ElecIterator(elec_tree_inner);}
    ElecIterator end() {return ElecIterator(elec_tree_inner, false);}
    std::shared_ptr<Aggregate> elec_tree_inner;
  };

  // aggregate
  struct Aggregate
  {
  public:
    std::vector< std::shared_ptr<Aggregate> > aggs;
    std::vector< std::shared_ptr<DBDot> > dbs;
    std::vector< std::shared_ptr<Defect> > defects;
    std::vector< std::shared_ptr<Electrode> > elecs;
    std::vector< std::shared_ptr<ElectrodePoly> > elec_polys;

    // Properties
    int size(); // returns the number of contained elecs, including those in children aggs
  };

  // SQCommand base class
  class SQCommand
  {
  public:

    // Enum indicating the item type this command contains.
    enum CommandItem{NoItem, DBDot, Electrode, Aggregate};

    // Enum indicating the type of command this is.
    enum CommandAction{Add, Remove, Echo, Run, Move};

    // Constructor taking enums.
    SQCommand(CommandAction action, CommandItem item)
      : action(action), item(item) {};

    // Constructor taking strings (for Python SWIG calls), the strings are
    // converted to appropriate enums internally.
    SQCommand(std::string str_action, std::string str_item)
      : action(commandActionEnum(str_action)), item(commandItemEnum(str_item)) {};

    // Destructor.
    virtual ~SQCommand() {};

    // TODO need a way to uniquely reference all items in order to implement
    // movement and removal commands.

    // Return the string corresponding to the specified CommandItem.
    static std::string commandItemString(CommandItem);

    // Return the string corresponding to the specified CommandAction.
    static std::string commandActionString(CommandAction);

    // Return the enum corresponding to the specified command item string.
    static CommandItem commandItemEnum(std::string);

    // Return the enum corresponding to the specified command action string.
    static CommandAction commandActionEnum(std::string);

    // Return the item type.
    CommandItem commandItem() {return item;}

    // Return the action type.
    CommandAction commandAction() {return action;}

    // Return the final command.
    std::string finalCommand();

    // Return the item creation command.
    std::string addActionCommand();

    // Return arguments for item creation command.
    virtual std::vector<std::string> addActionArguments() {return std::vector<std::string>();}

  private:

    // Construct the static variables
    static void constructStatics();

    CommandAction action;   // store the action type of command
    CommandItem item;       // store the item type of this command

    static boost::bimap<CommandAction, std::string> command_action_string;
    static boost::bimap<CommandItem, std::string> command_item_string;

  }; // end of SQCommand class

  // Aggregate creation command
  class AggregateCommand : public SQCommand
  {
  public:

    // Constructor with only the action.
    AggregateCommand(CommandAction action, const int &t_layer=-1)
      : SQCommand(action, Aggregate), layer(t_layer) {};

    // Constructor with the action in string (for SWIG).
    AggregateCommand(std::string str_action, const int &t_layer=-1)
      : SQCommand(commandActionEnum(str_action), Aggregate), layer(t_layer) {};

    // Constructor taking a vector of DB physical locations that should be
    // contained in a new Aggregate, implies an Add action.
    AggregateCommand(const std::vector< std::pair<float, float> > &t_db_locs,
                     const int &t_layer=-1)
      : SQCommand(Add, Aggregate), layer(t_layer)
    {
      addDBsToAggregateFormation(t_db_locs);
    }

    // Set the layer index.
    void setLayer(const int &t_layer) {layer = t_layer;}

    // Return the layer index.
    int getLayer() {return layer;}

    // Return the Aggregate creation command. The format should be:
    // add Aggregate (db_x1 db_y1) (db_x2 db_y2) ...
    std::vector<std::string> addActionArguments();

    // Add DB physical locations (only successful if the command action is Add.
    void addDBsToAggregateFormation(const std::vector< std::pair<float, float> > &t_db_locs)
    {
      db_locs.insert(db_locs.end(), t_db_locs.begin(), t_db_locs.end());
    }

    // Return the db_locs vector.
    std::vector< std::pair<float, float> > dbLocations() {return db_locs;}

    // TODO need a way to uniquely reference Aggregates in order to reference
    // aggregates for forming higher level aggregates, movement or removal.

  private:

    int layer=-1;   // store the layer which this command affects, or auto if -1
    std::vector< std::pair<float, float> > db_locs;

  }; // end of AggregateCommand class


}//end namespace phys

#endif
