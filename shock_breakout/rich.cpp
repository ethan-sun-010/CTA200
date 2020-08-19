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
#include "source/newtonian/two_dimensional/source_terms/SeveralSources.hpp"

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
    vector<ComputationalCell> res(static_cast<size_t>(tess.GetPointNo()));
    for(size_t i=0;i<res.size();++i){
      res.at(i).density = 1e-9;
      res.at(i).pressure = 1e-9;
      res.at(i).velocity = Vector2D(0,0);
      const Vector2D& r = tess.GetMeshPoint(static_cast<int>(i));
      const double x = 1 - abs(r);

      if(abs(r)<1)
	res.at(i).density = pow(x,1.5);

      if(abs(r-Vector2D(0,1))<0.04)
	res.at(i).pressure = 1e6;
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
       TracerStickerNames const& tracerstickernames,
       double /*time*/) const
    {
      size_t N = static_cast<size_t>(tess.GetPointNo());
      vector<ComputationalCell> res(N, old[0]);
      for(size_t i=0;i<N;++i){
	Extensive& extensive = extensives[i];
	const double volume = cd.volumes[i];
	res[i].density = extensive.mass / volume;
	if(res[i].density<0)
	  throw UniversalError("Negative density");
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
	    res[i].pressure = 1e-9;
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

  class SimData
  {
  public:

    SimData(void):
      pg_(Vector2D(0,0), Vector2D(0,1)),
      width_(96),
      outer_(1e-3,width_,width_,-width_),
#ifdef RICH_MPI
      vproc_(process_positions(outer_),outer_),
      init_points_(SquareMeshM(50,50,vproc_,outer_.getBoundary().first,outer_.getBoundary().second)),
      tess_(vproc_,init_points_,outer_),
#else
      init_points_(clip_grid
		   (RightRectangle(Vector2D(1e-3,-width_), Vector2D(width_, width_)),
		    complete_grid(0.15,
				  2*width_,
				  0.005))),
      tess_(init_points_, outer_),
#endif
      eos_(5./3.),
      //bpm_(),
      //point_motion_(bpm_,eos_),
      point_motion_(),
      sb_(),
      rs_(),
      geom_force_(pg_.getAxis()),
      force_(VectorInitialiser<SourceTerm*>
	     (&geom_force_)()),
      tsf_(0.3),
      fc_(rs_),
      eu_(),
      cu_(),
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
	   cu_) {}

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
#ifdef RICH_MPI
    //Eulerian point_motion_;
    //	Lagrangian point_motion_;
#else
    Eulerian point_motion_;
    //    Lagrangian bpm_;
    //    RoundCells point_motion_;
    //Lagrangian point_motion_;
#endif
    const StationaryBox sb_;
    const Hllc rs_;
    CylindricalComplementary geom_force_;
    SeveralSources force_;
    const SimpleCFL tsf_;
    const SimpleFluxCalculator fc_;
    const SimpleExtensiveUpdater eu_;
    //    const SimpleCellUpdater cu_;
    const PressureFloor cu_;
    hdsim sim_;
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

  class VolumeCalculator: public DiagnosticAppendix
  {
  public:

    vector<double> operator()(const hdsim& sim) const
    {
      vector<double> res(sim.getAllCells().size(),0);
      for(size_t i=0;i<res.size();++i)
	res.at(i) = sim.getCellVolume(i);
      return res;
    }

    string getName(void) const
    {
      return "volumes";
    }    
  };
}

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


int main(void)
{
#ifdef RICH_MPI
	MPI_Init(NULL,NULL);
	int rank=0;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
#endif
  SimData sim_data;
  hdsim& sim = sim_data.getSim();

  const double tf = 4e-1;
  SafeTimeTermination term_cond(tf,1e6);
  MultipleDiagnostics diag
  (VectorInitialiser<DiagnosticFunction*>()
   (new ConsecutiveSnapshots(new ConstantTimeInterval(tf/100),
			     new Rubric("output/snapshot_",".h5"),
			     VectorInitialiser<DiagnosticAppendix*>(new VolumeCalculator())
			     ()))
   (new VelocityProbe(Vector2D(1.05*sin(0*M_PI), -1.05*cos(0*M_PI)),
		      "velocity_probe_00.txt"))
   (new VelocityProbe(Vector2D(1.05*sin(0.11*M_PI), -1.05*cos(0.11*M_PI)),
		      "velocity_probe_01.txt"))
   (new VelocityProbe(Vector2D(1.05*sin(0.22*M_PI), -1.05*cos(0.22*M_PI)),
		      "velocity_probe_02.txt"))
   (new VelocityProbe(Vector2D(1.05*sin(0.33*M_PI), -1.05*cos(0.33*M_PI)),
		      "velocity_probe_03.txt"))
   (new VelocityProbe(Vector2D(1.05*sin(0.44*M_PI), -1.05*cos(0.44*M_PI)),
		      "velocity_probe_04.txt"))
   (new VelocityProbe(Vector2D(1.05*sin(0.56*M_PI), -1.05*cos(0.56*M_PI)),
		      "velocity_probe_05.txt"))
   (new VelocityProbe(Vector2D(1.05*sin(0.67*M_PI), -1.05*cos(0.67*M_PI)),
		      "velocity_probe_06.txt"))
   (new VelocityProbe(Vector2D(1.05*sin(0.78*M_PI), -1.05*cos(0.78*M_PI)),
		      "velocity_probe_07.txt"))
   (new VelocityProbe(Vector2D(1.05*sin(0.89*M_PI), -1.05*cos(0.89*M_PI)),
		      "velocity_probe_08.txt"))
   (new VelocityProbe(Vector2D(1.05*sin(M_PI), -1.05*cos(M_PI)),
                      "velocity_probe_09.txt"))

   (new VelocityProbe(Vector2D(sin(0*M_PI), -cos(0*M_PI)),
                      "velocity_probe_10.txt"))
   (new VelocityProbe(Vector2D(sin(0.11*M_PI), -cos(0.11*M_PI)),
                      "velocity_probe_11.txt"))
   (new VelocityProbe(Vector2D(sin(0.22*M_PI), -cos(0.22*M_PI)),
                      "velocity_probe_12.txt"))
   (new VelocityProbe(Vector2D(sin(0.33*M_PI), -cos(0.33*M_PI)),
                      "velocity_probe_13.txt"))
   (new VelocityProbe(Vector2D(sin(0.44*M_PI), -cos(0.44*M_PI)),
                      "velocity_probe_14.txt"))
   (new VelocityProbe(Vector2D(sin(0.56*M_PI), -cos(0.56*M_PI)),
                      "velocity_probe_15.txt"))
   (new VelocityProbe(Vector2D(sin(0.67*M_PI), -cos(0.67*M_PI)),
                      "velocity_probe_16.txt"))
   (new VelocityProbe(Vector2D(sin(0.78*M_PI), -cos(0.78*M_PI)),
                      "velocity_probe_17.txt"))
   (new VelocityProbe(Vector2D(sin(0.89*M_PI), -cos(0.89*M_PI)),
                      "velocity_probe_18.txt"))
   (new VelocityProbe(Vector2D(sin(M_PI), -cos(M_PI)),
                      "velocity_probe_19.txt"))

   (new VelocityProbe(Vector2D(0.95*sin(0*M_PI), -0.95*cos(0*M_PI)),
		      "velocity_probe_20.txt"))
   (new VelocityProbe(Vector2D(0.95*sin(0.11*M_PI), -0.95*cos(0.11*M_PI)),
		      "velocity_probe_21.txt"))
   (new VelocityProbe(Vector2D(0.95*sin(0.22*M_PI), -0.95*cos(0.22*M_PI)),
		      "velocity_probe_22.txt"))
   (new VelocityProbe(Vector2D(0.95*sin(0.33*M_PI), -0.95*cos(0.33*M_PI)),
		      "velocity_probe_23.txt"))
   (new VelocityProbe(Vector2D(0.95*sin(0.44*M_PI), -0.95*cos(0.44*M_PI)),
		      "velocity_probe_24.txt"))
   (new VelocityProbe(Vector2D(0.95*sin(0.56*M_PI), -0.95*cos(0.56*M_PI)),
		      "velocity_probe_25.txt"))
   (new VelocityProbe(Vector2D(0.95*sin(0.67*M_PI), -0.95*cos(0.67*M_PI)),
		      "velocity_probe_26.txt"))
   (new VelocityProbe(Vector2D(0.95*sin(0.78*M_PI), -0.95*cos(0.78*M_PI)),
		      "velocity_probe_27.txt"))
   (new VelocityProbe(Vector2D(0.95*sin(0.89*M_PI), -0.95*cos(0.89*M_PI)),
		      "velocity_probe_28.txt"))
   (new VelocityProbe(Vector2D(0.95*sin(M_PI), -0.95*cos(M_PI)),
                      "velocity_probe_29.txt"))
 
   (new VelocityProbe(Vector2D(0.9*sin(0*M_PI), -0.9*cos(0*M_PI)),
                      "velocity_probe_30.txt"))
   (new VelocityProbe(Vector2D(0.9*sin(0.11*M_PI), -0.9*cos(0.11*M_PI)),
                      "velocity_probe_31.txt"))
   (new VelocityProbe(Vector2D(0.9*sin(0.22*M_PI), -0.9*cos(0.22*M_PI)),
                      "velocity_probe_32.txt"))
   (new VelocityProbe(Vector2D(0.9*sin(0.33*M_PI), -0.9*cos(0.33*M_PI)),
                      "velocity_probe_33.txt"))
   (new VelocityProbe(Vector2D(0.9*sin(0.44*M_PI), -0.9*cos(0.44*M_PI)),
                      "velocity_probe_34.txt"))
   (new VelocityProbe(Vector2D(0.9*sin(0.56*M_PI), -0.9*cos(0.56*M_PI)),
                      "velocity_probe_35.txt"))
   (new VelocityProbe(Vector2D(0.9*sin(0.67*M_PI), -0.9*cos(0.67*M_PI)),
                      "velocity_probe_36.txt"))
   (new VelocityProbe(Vector2D(0.9*sin(0.78*M_PI), -0.9*cos(0.78*M_PI)),
                      "velocity_probe_37.txt"))
   (new VelocityProbe(Vector2D(0.9*sin(0.89*M_PI), -0.9*cos(0.89*M_PI)),
                      "velocity_probe_38.txt"))
   (new VelocityProbe(Vector2D(0.9*sin(M_PI), -0.9*cos(M_PI)),
                      "velocity_probe_39.txt"))

   (new VelocityProbe(Vector2D(0.85*sin(0*M_PI), -0.85*cos(0*M_PI)),
                      "velocity_probe_40.txt"))
   (new VelocityProbe(Vector2D(0.85*sin(0.11*M_PI), -0.85*cos(0.11*M_PI)),
                      "velocity_probe_41.txt"))
   (new VelocityProbe(Vector2D(0.85*sin(0.22*M_PI), -0.85*cos(0.22*M_PI)),
                      "velocity_probe_42.txt"))
   (new VelocityProbe(Vector2D(0.85*sin(0.33*M_PI), -0.85*cos(0.33*M_PI)),
                      "velocity_probe_43.txt"))
   (new VelocityProbe(Vector2D(0.85*sin(0.44*M_PI), -0.85*cos(0.44*M_PI)),
                      "velocity_probe_44.txt"))
   (new VelocityProbe(Vector2D(0.85*sin(0.56*M_PI), -0.85*cos(0.56*M_PI)),
                      "velocity_probe_45.txt"))
   (new VelocityProbe(Vector2D(0.85*sin(0.67*M_PI), -0.85*cos(0.67*M_PI)),
                      "velocity_probe_46.txt"))
   (new VelocityProbe(Vector2D(0.85*sin(0.78*M_PI), -0.85*cos(0.78*M_PI)),
                      "velocity_probe_47.txt"))
   (new VelocityProbe(Vector2D(0.85*sin(0.89*M_PI), -0.85*cos(0.89*M_PI)),
                      "velocity_probe_48.txt"))
   (new VelocityProbe(Vector2D(0.85*sin(M_PI), -0.85*cos(M_PI)),
                      "velocity_probe_49.txt"))

   (new VelocityProbe(Vector2D(0.8*sin(0*M_PI), -0.8*cos(0*M_PI)),
                      "velocity_probe_50.txt"))
   (new VelocityProbe(Vector2D(0.8*sin(0.11*M_PI), -0.8*cos(0.11*M_PI)),
                      "velocity_probe_51.txt"))
   (new VelocityProbe(Vector2D(0.8*sin(0.22*M_PI), -0.8*cos(0.22*M_PI)),
                      "velocity_probe_52.txt"))
   (new VelocityProbe(Vector2D(0.8*sin(0.33*M_PI), -0.8*cos(0.33*M_PI)),
                      "velocity_probe_53.txt"))
   (new VelocityProbe(Vector2D(0.8*sin(0.44*M_PI), -0.8*cos(0.44*M_PI)),
                      "velocity_probe_54.txt"))
   (new VelocityProbe(Vector2D(0.8*sin(0.56*M_PI), -0.8*cos(0.56*M_PI)),
                      "velocity_probe_55.txt"))
   (new VelocityProbe(Vector2D(0.8*sin(0.67*M_PI), -0.8*cos(0.67*M_PI)),
                      "velocity_probe_56.txt"))
   (new VelocityProbe(Vector2D(0.8*sin(0.78*M_PI), -0.8*cos(0.78*M_PI)),
                      "velocity_probe_57.txt"))
   (new VelocityProbe(Vector2D(0.8*sin(0.89*M_PI), -0.8*cos(0.89*M_PI)),
                      "velocity_probe_58.txt"))
   (new VelocityProbe(Vector2D(0.8*sin(M_PI), -0.8*cos(M_PI)),
                      "velocity_probe_59.txt"))

   (new VelocityProbe(Vector2D(0.75*sin(0*M_PI), -0.75*cos(0*M_PI)),
                      "velocity_probe_40.txt"))
   (new VelocityProbe(Vector2D(0.75*sin(0.11*M_PI), -0.75*cos(0.11*M_PI)),
                      "velocity_probe_41.txt"))
   (new VelocityProbe(Vector2D(0.75*sin(0.22*M_PI), -0.75*cos(0.22*M_PI)),
                      "velocity_probe_42.txt"))
   (new VelocityProbe(Vector2D(0.75*sin(0.33*M_PI), -0.75*cos(0.33*M_PI)),
                      "velocity_probe_43.txt"))
   (new VelocityProbe(Vector2D(0.75*sin(0.44*M_PI), -0.75*cos(0.44*M_PI)),
                      "velocity_probe_44.txt"))
   (new VelocityProbe(Vector2D(0.75*sin(0.56*M_PI), -0.75*cos(0.56*M_PI)),
                      "velocity_probe_45.txt"))
   (new VelocityProbe(Vector2D(0.75*sin(0.67*M_PI), -0.75*cos(0.67*M_PI)),
                      "velocity_probe_46.txt"))
   (new VelocityProbe(Vector2D(0.75*sin(0.78*M_PI), -0.75*cos(0.78*M_PI)),
                      "velocity_probe_47.txt"))
   (new VelocityProbe(Vector2D(0.75*sin(0.89*M_PI), -0.75*cos(0.89*M_PI)),
                      "velocity_probe_48.txt"))
   (new VelocityProbe(Vector2D(0.75*sin(M_PI), -0.75*cos(M_PI)),
                      "velocity_probe_49.txt"))

   (new VelocityProbe(Vector2D(0.7*sin(0*M_PI), -0.7*cos(0*M_PI)),
                      "velocity_probe_50.txt"))
   (new VelocityProbe(Vector2D(0.7*sin(0.11*M_PI), -0.7*cos(0.11*M_PI)),
                      "velocity_probe_51.txt"))
   (new VelocityProbe(Vector2D(0.7*sin(0.22*M_PI), -0.7*cos(0.22*M_PI)),
                      "velocity_probe_52.txt"))
   (new VelocityProbe(Vector2D(0.7*sin(0.33*M_PI), -0.7*cos(0.33*M_PI)),
                      "velocity_probe_53.txt"))
   (new VelocityProbe(Vector2D(0.7*sin(0.44*M_PI), -0.7*cos(0.44*M_PI)),
                      "velocity_probe_54.txt"))
   (new VelocityProbe(Vector2D(0.7*sin(0.56*M_PI), -0.7*cos(0.56*M_PI)),
                      "velocity_probe_55.txt"))
   (new VelocityProbe(Vector2D(0.7*sin(0.67*M_PI), -0.7*cos(0.67*M_PI)),
                      "velocity_probe_56.txt"))
   (new VelocityProbe(Vector2D(0.7*sin(0.78*M_PI), -0.7*cos(0.78*M_PI)),
                      "velocity_probe_57.txt"))
   (new VelocityProbe(Vector2D(0.7*sin(0.89*M_PI), -0.7*cos(0.89*M_PI)),
                      "velocity_probe_58.txt"))
   (new VelocityProbe(Vector2D(0.7*sin(M_PI), -0.7*cos(M_PI)),
                      "velocity_probe_59.txt"))

   (new VelocityProbe(Vector2D(0.65*sin(0*M_PI), -0.65*cos(0*M_PI)),
                      "velocity_probe_60.txt"))
   (new VelocityProbe(Vector2D(0.65*sin(0.11*M_PI), -0.65*cos(0.11*M_PI)),
                      "velocity_probe_61.txt"))
   (new VelocityProbe(Vector2D(0.65*sin(0.22*M_PI), -0.65*cos(0.22*M_PI)),
                      "velocity_probe_62.txt"))
   (new VelocityProbe(Vector2D(0.65*sin(0.33*M_PI), -0.65*cos(0.33*M_PI)),
                      "velocity_probe_63.txt"))
   (new VelocityProbe(Vector2D(0.65*sin(0.44*M_PI), -0.65*cos(0.44*M_PI)),
                      "velocity_probe_64.txt"))
   (new VelocityProbe(Vector2D(0.65*sin(0.56*M_PI), -0.65*cos(0.56*M_PI)),
                      "velocity_probe_65.txt"))
   (new VelocityProbe(Vector2D(0.65*sin(0.67*M_PI), -0.65*cos(0.67*M_PI)),
                      "velocity_probe_66.txt"))
   (new VelocityProbe(Vector2D(0.65*sin(0.78*M_PI), -0.65*cos(0.78*M_PI)),
                      "velocity_probe_67.txt"))
   (new VelocityProbe(Vector2D(0.65*sin(0.89*M_PI), -0.65*cos(0.89*M_PI)),
                      "velocity_probe_68.txt"))
   (new VelocityProbe(Vector2D(0.65*sin(M_PI), -0.65*cos(M_PI)),
                      "velocity_probe_69.txt"))

   (new VelocityProbe(Vector2D(0.6*sin(0*M_PI), -0.6*cos(0*M_PI)),
                      "velocity_probe_70.txt"))
   (new VelocityProbe(Vector2D(0.6*sin(0.11*M_PI), -0.6*cos(0.11*M_PI)),
                      "velocity_probe_71.txt"))
   (new VelocityProbe(Vector2D(0.6*sin(0.22*M_PI), -0.6*cos(0.22*M_PI)),
                      "velocity_probe_72.txt"))
   (new VelocityProbe(Vector2D(0.6*sin(0.33*M_PI), -0.6*cos(0.33*M_PI)),
                      "velocity_probe_73.txt"))
   (new VelocityProbe(Vector2D(0.6*sin(0.44*M_PI), -0.6*cos(0.44*M_PI)),
                      "velocity_probe_74.txt"))
   (new VelocityProbe(Vector2D(0.6*sin(0.56*M_PI), -0.6*cos(0.56*M_PI)),
                      "velocity_probe_75.txt"))
   (new VelocityProbe(Vector2D(0.6*sin(0.67*M_PI), -0.6*cos(0.67*M_PI)),
                      "velocity_probe_76.txt"))
   (new VelocityProbe(Vector2D(0.6*sin(0.78*M_PI), -0.6*cos(0.78*M_PI)),
                      "velocity_probe_77.txt"))
   (new VelocityProbe(Vector2D(0.6*sin(0.89*M_PI), -0.6*cos(0.89*M_PI)),
                      "velocity_probe_78.txt"))
   (new VelocityProbe(Vector2D(0.6*sin(M_PI), -0.6*cos(M_PI)),
                      "velocity_probe_79.txt"))

   (new VelocityProbe(Vector2D(0.55*sin(0*M_PI), -0.55*cos(0*M_PI)),
                      "velocity_probe_80.txt"))
   (new VelocityProbe(Vector2D(0.55*sin(0.11*M_PI), -0.55*cos(0.11*M_PI)),
                      "velocity_probe_81.txt"))
   (new VelocityProbe(Vector2D(0.55*sin(0.22*M_PI), -0.55*cos(0.22*M_PI)),
                      "velocity_probe_82.txt"))
   (new VelocityProbe(Vector2D(0.55*sin(0.33*M_PI), -0.55*cos(0.33*M_PI)),
                      "velocity_probe_83.txt"))
   (new VelocityProbe(Vector2D(0.55*sin(0.44*M_PI), -0.55*cos(0.44*M_PI)),
                      "velocity_probe_84.txt"))
   (new VelocityProbe(Vector2D(0.55*sin(0.56*M_PI), -0.55*cos(0.56*M_PI)),
                      "velocity_probe_85.txt"))
   (new VelocityProbe(Vector2D(0.55*sin(0.67*M_PI), -0.55*cos(0.67*M_PI)),
                      "velocity_probe_86.txt"))
   (new VelocityProbe(Vector2D(0.55*sin(0.78*M_PI), -0.55*cos(0.78*M_PI)),
                      "velocity_probe_87.txt"))
   (new VelocityProbe(Vector2D(0.55*sin(0.89*M_PI), -0.55*cos(0.89*M_PI)),
                      "velocity_probe_88.txt"))
   (new VelocityProbe(Vector2D(0.55*sin(M_PI), -0.55*cos(M_PI)),
                      "velocity_probe_89.txt"))

   (new VelocityProbe(Vector2D(0.5*sin(0*M_PI), -0.5*cos(0*M_PI)),
                      "velocity_probe_90.txt"))
   (new VelocityProbe(Vector2D(0.5*sin(0.11*M_PI), -0.5*cos(0.11*M_PI)),
                      "velocity_probe_91.txt"))
   (new VelocityProbe(Vector2D(0.5*sin(0.22*M_PI), -0.5*cos(0.22*M_PI)),
                      "velocity_probe_92.txt"))
   (new VelocityProbe(Vector2D(0.5*sin(0.33*M_PI), -0.5*cos(0.33*M_PI)),
                      "velocity_probe_93.txt"))
   (new VelocityProbe(Vector2D(0.5*sin(0.44*M_PI), -0.5*cos(0.44*M_PI)),
                      "velocity_probe_94.txt"))
   (new VelocityProbe(Vector2D(0.5*sin(0.56*M_PI), -0.5*cos(0.56*M_PI)),
                      "velocity_probe_95.txt"))
   (new VelocityProbe(Vector2D(0.5*sin(0.67*M_PI), -0.5*cos(0.67*M_PI)),
                      "velocity_probe_96.txt"))
   (new VelocityProbe(Vector2D(0.5*sin(0.78*M_PI), -0.5*cos(0.78*M_PI)),
                      "velocity_probe_97.txt"))
   (new VelocityProbe(Vector2D(0.5*sin(0.89*M_PI), -0.5*cos(0.89*M_PI)),
                      "velocity_probe_98.txt"))
   (new VelocityProbe(Vector2D(0.5*sin(M_PI), -0.5*cos(M_PI)),
                      "velocity_probe_99.txt"))

   (new VelocityProbe(Vector2D(0.45*sin(0*M_PI), -0.45*cos(0*M_PI)),
                      "velocity_probe_100.txt"))
   (new VelocityProbe(Vector2D(0.45*sin(0.11*M_PI), -0.45*cos(0.11*M_PI)),
                      "velocity_probe_101.txt"))
   (new VelocityProbe(Vector2D(0.45*sin(0.22*M_PI), -0.45*cos(0.22*M_PI)),
                      "velocity_probe_102.txt"))
   (new VelocityProbe(Vector2D(0.45*sin(0.33*M_PI), -0.45*cos(0.33*M_PI)),
                      "velocity_probe_103.txt"))
   (new VelocityProbe(Vector2D(0.45*sin(0.44*M_PI), -0.45*cos(0.44*M_PI)),
                      "velocity_probe_104.txt"))
   (new VelocityProbe(Vector2D(0.45*sin(0.56*M_PI), -0.45*cos(0.56*M_PI)),
                      "velocity_probe_105.txt"))
   (new VelocityProbe(Vector2D(0.45*sin(0.67*M_PI), -0.45*cos(0.67*M_PI)),
                      "velocity_probe_106.txt"))
   (new VelocityProbe(Vector2D(0.45*sin(0.78*M_PI), -0.45*cos(0.78*M_PI)),
                      "velocity_probe_107.txt"))
   (new VelocityProbe(Vector2D(0.45*sin(0.89*M_PI), -0.45*cos(0.89*M_PI)),
                      "velocity_probe_108.txt"))
   (new VelocityProbe(Vector2D(0.45*sin(M_PI), -0.45*cos(M_PI)),
                      "velocity_probe_109.txt"))

   (new VelocityProbe(Vector2D(0.4*sin(0*M_PI), -0.4*cos(0*M_PI)),
                      "velocity_probe_110.txt"))
   (new VelocityProbe(Vector2D(0.4*sin(0.11*M_PI), -0.4*cos(0.11*M_PI)),
                      "velocity_probe_111.txt"))
   (new VelocityProbe(Vector2D(0.4*sin(0.22*M_PI), -0.4*cos(0.22*M_PI)),
                      "velocity_probe_112.txt"))
   (new VelocityProbe(Vector2D(0.4*sin(0.33*M_PI), -0.4*cos(0.33*M_PI)),
                      "velocity_probe_113.txt"))
   (new VelocityProbe(Vector2D(0.4*sin(0.44*M_PI), -0.4*cos(0.44*M_PI)),
                      "velocity_probe_114.txt"))
   (new VelocityProbe(Vector2D(0.4*sin(0.56*M_PI), -0.4*cos(0.56*M_PI)),
                      "velocity_probe_115.txt"))
   (new VelocityProbe(Vector2D(0.4*sin(0.67*M_PI), -0.4*cos(0.67*M_PI)),
                      "velocity_probe_116.txt"))
   (new VelocityProbe(Vector2D(0.4*sin(0.78*M_PI), -0.4*cos(0.78*M_PI)),
                      "velocity_probe_117.txt"))
   (new VelocityProbe(Vector2D(0.4*sin(0.89*M_PI), -0.4*cos(0.89*M_PI)),
                      "velocity_probe_118.txt"))
   (new VelocityProbe(Vector2D(0.4*sin(M_PI), -0.4*cos(M_PI)),
                      "velocity_probe_119.txt"))

   (new VelocityProbe(Vector2D(0.35*sin(0*M_PI), -0.35*cos(0*M_PI)),
                      "velocity_probe_120.txt"))
   (new VelocityProbe(Vector2D(0.35*sin(0.11*M_PI), -0.35*cos(0.11*M_PI)),
                      "velocity_probe_121.txt"))
   (new VelocityProbe(Vector2D(0.35*sin(0.22*M_PI), -0.35*cos(0.22*M_PI)),
                      "velocity_probe_122.txt"))
   (new VelocityProbe(Vector2D(0.35*sin(0.33*M_PI), -0.35*cos(0.33*M_PI)),
                      "velocity_probe_123.txt"))
   (new VelocityProbe(Vector2D(0.35*sin(0.44*M_PI), -0.35*cos(0.44*M_PI)),
                      "velocity_probe_124.txt"))
   (new VelocityProbe(Vector2D(0.35*sin(0.56*M_PI), -0.35*cos(0.56*M_PI)),
                      "velocity_probe_125.txt"))
   (new VelocityProbe(Vector2D(0.35*sin(0.67*M_PI), -0.35*cos(0.67*M_PI)),
                      "velocity_probe_126.txt"))
   (new VelocityProbe(Vector2D(0.35*sin(0.78*M_PI), -0.35*cos(0.78*M_PI)),
                      "velocity_probe_127.txt"))
   (new VelocityProbe(Vector2D(0.35*sin(0.89*M_PI), -0.35*cos(0.89*M_PI)),
                      "velocity_probe_128.txt"))
   (new VelocityProbe(Vector2D(0.35*sin(M_PI), -0.35*cos(M_PI)),
                      "velocity_probe_129.txt"))

   (new VelocityProbe(Vector2D(0.3*sin(0*M_PI), -0.3*cos(0*M_PI)),
                      "velocity_probe_130.txt"))
   (new VelocityProbe(Vector2D(0.3*sin(0.11*M_PI), -0.3*cos(0.11*M_PI)),
                      "velocity_probe_131.txt"))
   (new VelocityProbe(Vector2D(0.3*sin(0.22*M_PI), -0.3*cos(0.22*M_PI)),
                      "velocity_probe_132.txt"))
   (new VelocityProbe(Vector2D(0.3*sin(0.33*M_PI), -0.3*cos(0.33*M_PI)),
                      "velocity_probe_133.txt"))
   (new VelocityProbe(Vector2D(0.3*sin(0.44*M_PI), -0.3*cos(0.44*M_PI)),
                      "velocity_probe_134.txt"))
   (new VelocityProbe(Vector2D(0.3*sin(0.56*M_PI), -0.3*cos(0.56*M_PI)),
                      "velocity_probe_135.txt"))
   (new VelocityProbe(Vector2D(0.3*sin(0.67*M_PI), -0.3*cos(0.67*M_PI)),
                      "velocity_probe_136.txt"))
   (new VelocityProbe(Vector2D(0.3*sin(0.78*M_PI), -0.3*cos(0.78*M_PI)),
                      "velocity_probe_137.txt"))
   (new VelocityProbe(Vector2D(0.3*sin(0.89*M_PI), -0.3*cos(0.89*M_PI)),
                      "velocity_probe_138.txt"))
   (new VelocityProbe(Vector2D(0.3*sin(M_PI), -0.3*cos(M_PI)),
                      "velocity_probe_139.txt"))

   (new VelocityProbe(Vector2D(0.25*sin(0*M_PI), -0.25*cos(0*M_PI)),
                      "velocity_probe_140.txt"))
   (new VelocityProbe(Vector2D(0.25*sin(0.11*M_PI), -0.25*cos(0.11*M_PI)),
                      "velocity_probe_141.txt"))
   (new VelocityProbe(Vector2D(0.25*sin(0.22*M_PI), -0.25*cos(0.22*M_PI)),
                      "velocity_probe_142.txt"))
   (new VelocityProbe(Vector2D(0.25*sin(0.33*M_PI), -0.25*cos(0.33*M_PI)),
                      "velocity_probe_143.txt"))
   (new VelocityProbe(Vector2D(0.25*sin(0.44*M_PI), -0.25*cos(0.44*M_PI)),
                      "velocity_probe_144.txt"))
   (new VelocityProbe(Vector2D(0.25*sin(0.56*M_PI), -0.25*cos(0.56*M_PI)),
                      "velocity_probe_145.txt"))
   (new VelocityProbe(Vector2D(0.25*sin(0.67*M_PI), -0.25*cos(0.67*M_PI)),
                      "velocity_probe_146.txt"))
   (new VelocityProbe(Vector2D(0.25*sin(0.78*M_PI), -0.25*cos(0.78*M_PI)),
                      "velocity_probe_147.txt"))
   (new VelocityProbe(Vector2D(0.25*sin(0.89*M_PI), -0.25*cos(0.89*M_PI)),
                      "velocity_probe_148.txt"))
   (new VelocityProbe(Vector2D(0.25*sin(M_PI), -0.25*cos(M_PI)),
                      "velocity_probe_149.txt"))

   (new VelocityProbe(Vector2D(0.2*sin(0*M_PI), -0.2*cos(0*M_PI)),
                      "velocity_probe_150.txt"))
   (new VelocityProbe(Vector2D(0.2*sin(0.11*M_PI), -0.2*cos(0.11*M_PI)),
                      "velocity_probe_151.txt"))
   (new VelocityProbe(Vector2D(0.2*sin(0.22*M_PI), -0.2*cos(0.22*M_PI)),
                      "velocity_probe_152.txt"))
   (new VelocityProbe(Vector2D(0.2*sin(0.33*M_PI), -0.2*cos(0.33*M_PI)),
                      "velocity_probe_153.txt"))
   (new VelocityProbe(Vector2D(0.2*sin(0.44*M_PI), -0.2*cos(0.44*M_PI)),
                      "velocity_probe_154.txt"))
   (new VelocityProbe(Vector2D(0.2*sin(0.56*M_PI), -0.2*cos(0.56*M_PI)),
                      "velocity_probe_155.txt"))
   (new VelocityProbe(Vector2D(0.2*sin(0.67*M_PI), -0.2*cos(0.67*M_PI)),
                      "velocity_probe_156.txt"))
   (new VelocityProbe(Vector2D(0.2*sin(0.78*M_PI), -0.2*cos(0.78*M_PI)),
                      "velocity_probe_157.txt"))
   (new VelocityProbe(Vector2D(0.2*sin(0.89*M_PI), -0.2*cos(0.89*M_PI)),
                      "velocity_probe_158.txt"))
   (new VelocityProbe(Vector2D(0.2*sin(M_PI), -0.2*cos(M_PI)),
                      "velocity_probe_159.txt"))

   (new VelocityProbe(Vector2D(0.15*sin(0*M_PI), -0.15*cos(0*M_PI)),
                      "velocity_probe_160.txt"))
   (new VelocityProbe(Vector2D(0.15*sin(0.11*M_PI), -0.15*cos(0.11*M_PI)),
                      "velocity_probe_161.txt"))
   (new VelocityProbe(Vector2D(0.15*sin(0.22*M_PI), -0.15*cos(0.22*M_PI)),
                      "velocity_probe_162.txt"))
   (new VelocityProbe(Vector2D(0.15*sin(0.33*M_PI), -0.15*cos(0.33*M_PI)),
                      "velocity_probe_163.txt"))
   (new VelocityProbe(Vector2D(0.15*sin(0.44*M_PI), -0.15*cos(0.44*M_PI)),
                      "velocity_probe_164.txt"))
   (new VelocityProbe(Vector2D(0.15*sin(0.56*M_PI), -0.15*cos(0.56*M_PI)),
                      "velocity_probe_165.txt"))
   (new VelocityProbe(Vector2D(0.15*sin(0.67*M_PI), -0.15*cos(0.67*M_PI)),
                      "velocity_probe_166.txt"))
   (new VelocityProbe(Vector2D(0.15*sin(0.78*M_PI), -0.15*cos(0.78*M_PI)),
                      "velocity_probe_167.txt"))
   (new VelocityProbe(Vector2D(0.15*sin(0.89*M_PI), -0.15*cos(0.89*M_PI)),
                      "velocity_probe_168.txt"))
   (new VelocityProbe(Vector2D(0.15*sin(M_PI), -0.15*cos(M_PI)),
                      "velocity_probe_169.txt"))

   (new VelocityProbe(Vector2D(0.1*sin(0*M_PI), -0.1*cos(0*M_PI)),
                      "velocity_probe_170.txt"))
   (new VelocityProbe(Vector2D(0.1*sin(0.11*M_PI), -0.1*cos(0.11*M_PI)),
                      "velocity_probe_171.txt"))
   (new VelocityProbe(Vector2D(0.1*sin(0.22*M_PI), -0.1*cos(0.22*M_PI)),
                      "velocity_probe_172.txt"))
   (new VelocityProbe(Vector2D(0.1*sin(0.33*M_PI), -0.1*cos(0.33*M_PI)),
                      "velocity_probe_173.txt"))
   (new VelocityProbe(Vector2D(0.1*sin(0.44*M_PI), -0.1*cos(0.44*M_PI)),
                      "velocity_probe_174.txt"))
   (new VelocityProbe(Vector2D(0.1*sin(0.56*M_PI), -0.1*cos(0.56*M_PI)),
                      "velocity_probe_175.txt"))
   (new VelocityProbe(Vector2D(0.1*sin(0.67*M_PI), -0.1*cos(0.67*M_PI)),
                      "velocity_probe_176.txt"))
   (new VelocityProbe(Vector2D(0.1*sin(0.78*M_PI), -0.1*cos(0.78*M_PI)),
                      "velocity_probe_177.txt"))
   (new VelocityProbe(Vector2D(0.1*sin(0.89*M_PI), -0.1*cos(0.89*M_PI)),
                      "velocity_probe_178.txt"))
   (new VelocityProbe(Vector2D(0.1*sin(M_PI), -0.1*cos(M_PI)),
                      "velocity_probe_179.txt"))

   (new VelocityProbe(Vector2D(0.05*sin(0*M_PI), -0.05*cos(0*M_PI)),
                      "velocity_probe_180.txt"))
   (new VelocityProbe(Vector2D(0.05*sin(0.11*M_PI), -0.05*cos(0.11*M_PI)),
                      "velocity_probe_181.txt"))
   (new VelocityProbe(Vector2D(0.05*sin(0.22*M_PI), -0.05*cos(0.22*M_PI)),
                      "velocity_probe_182.txt"))
   (new VelocityProbe(Vector2D(0.05*sin(0.33*M_PI), -0.05*cos(0.33*M_PI)),
                      "velocity_probe_183.txt"))
   (new VelocityProbe(Vector2D(0.05*sin(0.44*M_PI), -0.05*cos(0.44*M_PI)),
                      "velocity_probe_184.txt"))
   (new VelocityProbe(Vector2D(0.05*sin(0.56*M_PI), -0.05*cos(0.56*M_PI)),
                      "velocity_probe_185.txt"))
   (new VelocityProbe(Vector2D(0.05*sin(0.67*M_PI), -0.05*cos(0.67*M_PI)),
                      "velocity_probe_186.txt"))
   (new VelocityProbe(Vector2D(0.05*sin(0.78*M_PI), -0.05*cos(0.78*M_PI)),
                      "velocity_probe_187.txt"))
   (new VelocityProbe(Vector2D(0.05*sin(0.89*M_PI), -0.05*cos(0.89*M_PI)),
                      "velocity_probe_188.txt"))
   (new VelocityProbe(Vector2D(0.05*sin(M_PI), -0.05*cos(M_PI)),
                      "velocity_probe_189.txt"))


   (new WriteTime("time.txt"))
   (new WriteCycle("cycle.txt"))());
  write_snapshot_to_hdf5(sim, "output/initial.h5",
			 VectorInitialiser<DiagnosticAppendix*>(new VolumeCalculator())());
  main_loop(sim,
	    term_cond,
	    &hdsim::TimeAdvance,
	    &diag);
	    

#ifdef RICH_MPI
  write_snapshot_to_hdf5(sim, "process_"+int2str(rank)+"_final"+".h5");
  MPI_Finalize();
#else
  write_snapshot_to_hdf5(sim, "output/final.h5",
			 VectorInitialiser<DiagnosticAppendix*>(new VolumeCalculator())());
#endif

  return 0;
}

