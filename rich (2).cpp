#ifdef RICH_MPI
#include "source/mpi/MeshPointsMPI.hpp"
#endif
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include "source/tessellation/geometry.hpp"
#include "source/newtonian/two_dimensional/hdsim2d.hpp"
#include "source/tessellation/tessellation.hpp"
#include "source/newtonian/common/hllc.hpp"
#include "source/newtonian/common/ideal_gas.hpp"
#include "source/tessellation/VoronoiMesh.hpp"
#include "source/newtonian/two_dimensional/spatial_distributions/uniform2d.hpp"
#include "source/newtonian/two_dimensional/point_motions/eulerian.hpp"
#include "source/newtonian/two_dimensional/point_motions/lagrangian.hpp"
#include "source/newtonian/two_dimensional/point_motions/round_cells.hpp"
#include "source/newtonian/two_dimensional/source_terms/zero_force.hpp"
#include "source/newtonian/two_dimensional/geometric_outer_boundaries/SquareBox.hpp"
#include "source/newtonian/test_2d/random_pert.hpp"
#include "source/newtonian/two_dimensional/diagnostics.hpp"
#include "source/misc/simple_io.hpp"
#include "source/misc/mesh_generator.hpp"
#include "source/newtonian/test_2d/main_loop_2d.hpp"
#include "source/newtonian/two_dimensional/hdf5_diagnostics.hpp"
#include "source/tessellation/shape_2d.hpp"
#include "source/newtonian/test_2d/piecewise.hpp"
#include "source/newtonian/two_dimensional/simple_flux_calculator.hpp"
#include "source/newtonian/two_dimensional/simple_cell_updater.hpp"
#include "source/newtonian/two_dimensional/simple_extensive_updater.hpp"
#include "source/newtonian/two_dimensional/stationary_box.hpp"
#include "source/tessellation/right_rectangle.hpp"
#include "source/newtonian/test_2d/clip_grid.hpp"
#include "source/newtonian/test_2d/multiple_diagnostics.hpp"
#include "source/misc/vector_initialiser.hpp"
#include "source/newtonian/test_2d/consecutive_snapshots.hpp"
#include "source/newtonian/two_dimensional/source_terms/cylindrical_complementary.hpp"

using namespace std;
using namespace simulation2d;

namespace {

  vector<Vector2D> centered_hexagonal_grid(double r_min,
					   double r_max)
  {
    const vector<double> r_list = arange(0,r_max,r_min);
    vector<Vector2D> res;
    for(size_t i=0;i<r_list.size();++i){
      const size_t angle_num = max<size_t>(6*i,1);
      vector<double> angle_list(angle_num,0);
      for(size_t j=0;j<angle_num;++j)
	angle_list.at(j) = 2*M_PI*static_cast<double>(j)/static_cast<double>(angle_num);
      for(size_t j=0;j<angle_num;++j)
	res.push_back(r_list.at(i)*Vector2D(cos(angle_list.at(j)),
					    sin(angle_list.at(j))));
    }
    return res;
  }

  vector<Vector2D> centered_logarithmic_spiral(double r_min,
					       double r_max,
					       double alpha,
					       const Vector2D& center)
  {
    const double theta_max = log(r_max/r_min)/alpha;
    const vector<double> theta_list = 
      arange(0,theta_max,2*M_PI*alpha/(1-0.5*alpha));
  
    vector<double> r_list(theta_list.size(),0);
    for(size_t i=0;i<r_list.size();++i)
      r_list.at(i) = r_min*exp(alpha*theta_list.at(i));
  
    vector<Vector2D> res(r_list.size());
    for(size_t i=0;i<res.size();++i)
      res[i] = center+r_list[i]*Vector2D(cos(theta_list.at(i)),
					 sin(theta_list.at(i)));
    return res;
  }

  vector<Vector2D> complete_grid(double r_inner,
				 double r_outer,
				 double alpha)
  {
    const vector<Vector2D> inner = 
      centered_hexagonal_grid(r_inner*alpha*2*M_PI,
			      r_inner);
    const vector<Vector2D> outer =
      centered_logarithmic_spiral(r_inner,
				  r_outer,
				  alpha,
				  Vector2D(0,0));
    return join(inner, outer);
  }

#ifdef RICH_MPI

  vector<Vector2D> process_positions(const SquareBox& boundary)
  {
    const Vector2D lower_left = boundary.getBoundary().first;
    const Vector2D upper_right = boundary.getBoundary().second;
	int ws=0;
	MPI_Comm_size(MPI_COMM_WORLD,&ws);
    return RandSquare(ws,lower_left.x,upper_right.x,lower_left.y,upper_right.y);
  }

#endif

  vector<ComputationalCell> calc_init_cond(const Tessellation& tess)
  {
    const double s = read_number("radius_ratio.txt");
	vector<ComputationalCell> res(static_cast<size_t>(tess.GetPointNo()));
    for(size_t i=0;i<res.size();++i){
      res.at(i).density = 1e-11;
      res.at(i).pressure = 1e-12;
      res.at(i).velocity = Vector2D(0,0);
      //res.at(i).tracers.push_back(1.4);
      const Vector2D& r = tess.GetMeshPoint(static_cast<int>(i));
      // Impactor
      if(abs(r-Vector2D(0,s))<s){
	res.at(i).density = 1;
	res.at(i).velocity = Vector2D(0,-1);
      }
      // Target
      if(abs(r-Vector2D(0,-1))<1){
	res.at(i).density = 1;
	res.at(i).velocity = Vector2D(0,0);
      }
      // Core
      /*
      if(abs(r-Vector2D(0,-1))<0.5){
	res.at(i).density = 5;
	res.at(i).velocity = Vector2D(0,0);
	res.at(i).tracers.at(0) = 2.2;
      }
      */
    }
    return res;
  }

  class PressureFloor: public CellUpdater
  {
  public:

    PressureFloor(void) {}

    vector<ComputationalCell> operator()
      (const Tessellation& tess,
       const PhysicalGeometry& /*pg*/,
       const EquationOfState& eos,
       vector<Extensive>& extensives,
       const vector<ComputationalCell>& old,
       const CacheData& cd,
       TracerStickerNames const& tracerstickernames) const
    {
      size_t N = static_cast<size_t>(tess.GetPointNo());
      vector<ComputationalCell> res(N, old[0]);
      for(size_t i=0;i<N;++i){
	Extensive& extensive = extensives[i];
	const double volume = cd.volumes[i];
	res[i].density = extensive.mass / volume;
	if(res[i].density<0)
	  //	  throw UniversalError("Negative density");
	  res.at(i).density = 1e-12;
	res[i].velocity = extensive.momentum / extensive.mass;
	const double energy = extensive.energy / extensive.mass - 
	  0.5*ScalarProd(res[i].velocity, res[i].velocity);
	try{
	  if(energy>0)
	    res[i].pressure = eos.de2p(res[i].density,
				       energy,
				       res[i].tracers,
				       tracerstickernames.tracer_names);
	  else
	    res[i].pressure = 1e-12;
	}
	catch(UniversalError& eo){
	  eo.AddEntry("cell density", res[i].density);
	  eo.AddEntry("cell energy", energy);
	  throw;
	}
	for(size_t j=0;j<extensive.tracers.size();++j)
	  res.at(i).tracers.at(j) = extensive.tracers.at(j)/extensive.mass;
      }	
      return res;
    }
  };

  class MultiIdealGas: public EquationOfState
  {
  public:

    MultiIdealGas(void) {}

    double dp2e(double d,
		double p,
		const tvector& tracers,
		const vector<string>& tracernames) const
    {
      assert(tracernames.front()=="adiabatic_index");
      const double g = tracers.front();
      return p/d/(g-1);
    }

    double de2p(double d,
		double e,
		const tvector& tracers,
		const vector<string>& tracernames) const 
    {
      assert(tracernames.front()=="adiabatic_index");
      const double g = tracers.front();
      return (g-1)*e*d;
    }

    double dp2c(double d,
		double p,
		const tvector& tracers,
		const vector<string>& tracernames) const
    {
      assert(tracernames.front()=="adiabatic_index");
      const double g = tracers.front();
      return sqrt(g*p/d);
    }

    double de2c(double d,
		double e,
		const tvector& tracers,
		const vector<string>& tracernames) const
    {
      return dp2c(d, de2p(d,e,tracers,tracernames),
		  tracers,
		  tracernames);
    }

    double dp2s(double /*d*/,
		double /*p*/,
		const tvector& /*tracers*/,
		const vector<string>& /*tracernames*/) const
    {
      throw "not implemented";
    }

    double sd2p(double /*s*/,
		double /*d*/,
		const tvector& /*travers*/,
		const vector<string>& /*tracernames*/) const
    {
      throw "not implemented";
    }
  };

  class SimData
  {
  public:

    SimData(void):
      pg_(Vector2D(0,0), Vector2D(0,1)),
      width_(1e1),
      outer_(1e-4,width_,width_,-width_),
#ifdef RICH_MPI
	  vproc_(process_positions(outer_),outer_),
		init_points_(SquareMeshM(50,50,vproc_,outer_.getBoundary().first,outer_.getBoundary().second)),
		tess_(vproc_,init_points_,outer_),
#else
      init_points_(clip_grid
		   (RightRectangle(Vector2D(1e-4,-width_), Vector2D(width_, width_)),
		    complete_grid(0.1,
				  2*width_,
				  //0.002))),
				  0.002))),
		tess_(init_points_, outer_),
#endif
      eos_(5./3.),
      //eos_(),
      //      bpm_(),
      //point_motion_(bpm_, eos_, outer_, 0.7, 0.2),
      point_motion_(),
      sb_(),
      rs_(),
      force_(pg_.getAxis()),
      tsf_(0.3),
      fc_(rs_),
      eu_(),
      cu_(),
      tsn_(),
      sim_(
#ifdef RICH_MPI
	   vproc_,
#endif
	   tess_,
	   outer_,
	   pg_,
	   calc_init_cond(tess_),
	   eos_,
	   point_motion_,
	   sb_,
	   force_,
	   tsf_,
	   fc_,
	   eu_,
		  cu_,
	   tsn_) {}

    hdsim& getSim(void)
    {
      return sim_;
    }

  private:
    const CylindricalSymmetry pg_;
    const double width_;
    const SquareBox outer_;
#ifdef RICH_MPI
	VoronoiMesh vproc_;
#endif
    const vector<Vector2D> init_points_;
    VoronoiMesh tess_;
    const IdealGas eos_;
    //    const MultiIdealGas eos_;
#ifdef RICH_MPI
    //Eulerian point_motion_;
    //	Lagrangian point_motion_;
#else
    Eulerian point_motion_;
    //Lagrangian bpm_;
    //    RoundCells point_motion_;
#endif
    const StationaryBox sb_;
    const Hllc rs_;
    //    ZeroForce force_;
    CylindricalComplementary force_;
    const SimpleCFL tsf_;
    const SimpleFluxCalculator fc_;
    const SimpleExtensiveUpdater eu_;
    const PressureFloor cu_;
    TracerStickerNames tsn_;
    hdsim sim_;
  };

  class CraterSizeHistory: public DiagnosticFunction
  {
  public:

    CraterSizeHistory(const string& fname):
      fname_(fname),
      r_list_(),
      v_list_(),
      t_list_(),
      x_list_(),
      y_list_() {}

    void operator()(const hdsim& sim)
    {
      const vector<ComputationalCell>& cells = sim.getAllCells();
      const Tessellation& tess = sim.getTessellation();
      size_t idx = 0;
      //double min_vy = 0;
      double min_y = 0;
      for(size_t i=0;i<cells.size();++i){
	const Vector2D r = tess.GetMeshPoint(static_cast<int>(i));
	if(r.y>min_y)
	  continue;
	const ComputationalCell cell = cells.at(i);
	//const double entropy = log10(cell.pressure)-(5./3.)*log10(cell.density);
	if(cell.pressure<1e-6)
	  continue;
	idx = i;
	min_y = r.y;
	//const double candid = cells.at(i).velocity.y;
	//if(candid<min_vy){
	// idx = i;
	//  min_vy = candid;
	//}
      }
      const Vector2D r = tess.GetMeshPoint(static_cast<int>(idx));
      const ComputationalCell cell = cells.at(idx);
      r_list_.push_back(abs(r));
      v_list_.push_back(cell.velocity.y);
      x_list_.push_back(r.x);
      y_list_.push_back(r.y);
      t_list_.push_back(sim.getTime());
    }

    ~CraterSizeHistory(void)
    {
      ofstream f(fname_.c_str());
      for(size_t i=0;i<r_list_.size();++i)
	f << t_list_.at(i) << " "
	  << r_list_.at(i) << " "
	  << v_list_.at(i) << " "
	  << x_list_.at(i) << " "
	  << y_list_.at(i) << endl;
      f.close();
    }

  private:
    const string fname_;
    mutable vector<double> r_list_;
    mutable vector<double> v_list_;
    mutable vector<double> t_list_;
    mutable vector<double> x_list_;
    mutable vector<double> y_list_;
  };

  class WriteCycle: public DiagnosticFunction
  {
  public:

    WriteCycle(const string& fname):
      fname_(fname) {}

    void operator()(const hdsim& sim)
    {
      write_number(sim.getCycle(),fname_);
    }

  private:
    const string fname_;
  };

  size_t find_closest_index
  (const hdsim& sim,
   const Vector2D& pos)
  {
    const Tessellation& tess = sim.getTessellation();
    size_t res = 0;
    double dist = 1e30;
    const size_t n = static_cast<size_t>(tess.GetPointNo());
    for(size_t i=0;i<n;++i){
      const Vector2D r = tess.GetMeshPoint(static_cast<int>(i));
      const double dist_cand = abs(r-pos);
      if (dist_cand<dist){
	dist = dist_cand;
	res = i;
      }
    }
    return res;
  }

  class VelocityProbe: public DiagnosticFunction
  {
  public:

    VelocityProbe(const Vector2D& pos,
		  const string& fname):
      pos_(pos), 
      fname_(fname), 
      time_(), 
      velocity_x_(),
      velocity_y_(),
      density_() {}

    void operator()(const hdsim& sim)
    {
      const size_t index = find_closest_index(sim, pos_);
      const vector<ComputationalCell> cells = sim.getAllCells();
      const ComputationalCell& cell = cells.at(index);
      time_.push_back(sim.getTime());
      velocity_x_.push_back(cell.velocity.x);
      velocity_y_.push_back(cell.velocity.y);	
      density_.push_back(cell.density);
    }

    ~VelocityProbe(void)
    {
      ofstream f(fname_.c_str());
      for(size_t i=0;i<time_.size();++i){
	f << time_.at(i) << " "
	  << velocity_x_.at(i) << " "
	  << velocity_y_.at(i) << " "
	  << density_.at(i) << endl;
      }
      f.close();
    }

  private:
    const Vector2D pos_;
    const string fname_;
    mutable vector<double> time_;
    mutable vector<double> velocity_x_;
    mutable vector<double> velocity_y_;
    mutable vector<double> density_;
  };

  class DensityThreshold: public TerminationCondition
  {
  public:
 
    DensityThreshold(const Vector2D pos,
		     double thres):
      pos_(pos),
      thres_(thres) {}
 
    bool operator()(const hdsim& sim)
    {
      const size_t index = find_closest_index(sim,pos_);
      const vector<ComputationalCell>& cells = sim.getAllCells();
      return cells.at(index).density<thres_;
    }
 
  private:
    const Vector2D pos_;
    const double thres_;
  };
}

int main(void)
{
#ifdef RICH_MPI
	MPI_Init(NULL,NULL);
	int rank=0;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
#endif
  SimData sim_data;
  hdsim& sim = sim_data.getSim();
  write_snapshot_to_hdf5(sim, "initial.h5");

  //  const double tf = 5;
  //  SafeTimeTermination term_cond(tf,1e6);
  DensityThreshold term_cond(Vector2D(0,-2.3), 1e-1);
  MultipleDiagnostics diag
  (VectorInitialiser<DiagnosticFunction*>()
   //   (new ConsecutiveSnapshots(new ConstantTimeInterval(tf/100),
   //      			     new Rubric("output/snapshot_",".h5")))
   (new VelocityProbe(Vector2D(1.05*sin(0.1*M_PI), -1-1.05*cos(0.1*M_PI)),
		      "velocity_probe_out_0.txt"))
   (new VelocityProbe(Vector2D(1.05*sin(0.2*M_PI), -1-1.05*cos(0.2*M_PI)),
		      "velocity_probe_out_1.txt"))
   (new VelocityProbe(Vector2D(1.05*sin(0.3*M_PI), -1-1.05*cos(0.3*M_PI)),
		      "velocity_probe_out_2.txt"))
   (new VelocityProbe(Vector2D(1.05*sin(0.4*M_PI), -1-1.05*cos(0.4*M_PI)),
		      "velocity_probe_out_3.txt"))
   (new VelocityProbe(Vector2D(1.05*sin(0.5*M_PI), -1-1.05*cos(0.5*M_PI)),
		      "velocity_probe_out_4.txt"))
   (new VelocityProbe(Vector2D(1.05*sin(0.6*M_PI), -1-1.05*cos(0.6*M_PI)),
		      "velocity_probe_out_5.txt"))
   (new VelocityProbe(Vector2D(1.05*sin(0.7*M_PI), -1-1.05*cos(0.7*M_PI)),
		      "velocity_probe_out_6.txt"))
   (new VelocityProbe(Vector2D(1.05*sin(0.8*M_PI), -1-1.05*cos(0.8*M_PI)),
		      "velocity_probe_out_7.txt"))
   (new VelocityProbe(Vector2D(1.05*sin(0.9*M_PI), -1-1.05*cos(0.9*M_PI)),
		      "velocity_probe_out_8.txt"))
   (new VelocityProbe(Vector2D(0.95*sin(0.1*M_PI), -1-0.95*cos(0.1*M_PI)),
		      "velocity_probe_in_0.txt"))
   (new VelocityProbe(Vector2D(0.95*sin(0.2*M_PI), -1-0.95*cos(0.2*M_PI)),
		      "velocity_probe_in_1.txt"))
   (new VelocityProbe(Vector2D(0.95*sin(0.3*M_PI), -1-0.95*cos(0.3*M_PI)),
		      "velocity_probe_in_2.txt"))
   (new VelocityProbe(Vector2D(0.95*sin(0.4*M_PI), -1-0.95*cos(0.4*M_PI)),
		      "velocity_probe_in_3.txt"))
   (new VelocityProbe(Vector2D(0.95*sin(0.5*M_PI), -1-0.95*cos(0.5*M_PI)),
		      "velocity_probe_in_4.txt"))
   (new VelocityProbe(Vector2D(0.95*sin(0.6*M_PI), -1-0.95*cos(0.6*M_PI)),
		      "velocity_probe_in_5.txt"))
   (new VelocityProbe(Vector2D(0.95*sin(0.7*M_PI), -1-0.95*cos(0.7*M_PI)),
		      "velocity_probe_in_6.txt"))
   (new VelocityProbe(Vector2D(0.95*sin(0.8*M_PI), -1-0.95*cos(0.8*M_PI)),
		      "velocity_probe_in_7.txt"))
   (new VelocityProbe(Vector2D(0.95*sin(0.9*M_PI), -1-0.95*cos(0.9*M_PI)),
		      "velocity_probe_in_8.txt"))
   (new WriteTime("time.txt"))
   (new WriteCycle("cycle.txt"))());
  main_loop(sim,
	    term_cond,
	    &hdsim::TimeAdvance,
	    &diag);
	    

#ifdef RICH_MPI
  write_snapshot_to_hdf5(sim, "process_"+int2str(rank)+"_final"+".h5");
  MPI_Finalize();
#else
  write_snapshot_to_hdf5(sim, "final.h5");
#endif


  return 0;
}

