/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::Instrumentation::GaussianDumbbells class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_GAUSSIANDUMBBELLS_HPP
#define OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_GAUSSIANDUMBBELLS_HPP

#include <OpenMPCD/CUDA/MPCFluid/GaussianDumbbells.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/Base.hpp>
#include <OpenMPCD/CUDA/Simulation.hpp>
#include <OpenMPCD/Graph.hpp>
#include <OpenMPCD/Histogram.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{
namespace Instrumentation
{
/**
 * Instrumentation for Gaussian dumbbell fluids.
 */
class GaussianDumbbells : public Base
{
	public:
		/**
		 * The constructor.
		 * @param[in] sim       The simulation instance.
		 * @param[in] devMemMgr The Device memory manager.
		 * @param[in] mpcFluid_ The MPC fluid to measure.
		 */
		GaussianDumbbells(
			const Simulation* const sim, DeviceMemoryManager* const devMemMgr,
			const MPCFluid::GaussianDumbbells* const mpcFluid_);

		/**
		 * The destructor.
		 */
		virtual ~GaussianDumbbells()
		{
		}

	protected:
		/**
		 * Performs measurements.
		 */
		virtual void measureSpecific();

		/**
		 * Saves the data to the given run directory.
		 * @param[in] rundir The path to the run directory.
		 */
		virtual void saveSpecific(const std::string& rundir) const;

	private:
		const Simulation* const simulation; ///< The simulation instance.
		const MPCFluid::GaussianDumbbells* const mpcFluid; ///< The fluid.

		Histogram dumbbellBondLengthHistogram; ///< Histogram for the dumbbell bond lengths.
		Histogram dumbbellBondLengthSquaredHistogram; ///< Histogram for the squares of the dumbbell bond lengths.

		Histogram dumbbellBondXHistogram; ///< Histogram for the x components of the dumbbell bonds.
		Histogram dumbbellBondYHistogram; ///< Histogram for the y components of the dumbbell bonds.
		Histogram dumbbellBondZHistogram; ///< Histogram for the z components of the dumbbell bonds.

		Histogram dumbbellBondXXHistogram; ///< Histogram for the squares of the x components of the dumbbell bonds.
		Histogram dumbbellBondYYHistogram; ///< Histogram for the squares of the y components of the dumbbell bonds.
		Histogram dumbbellBondZZHistogram; ///< Histogram for the squares of the z components of the dumbbell bonds.

		Histogram dumbbellBondXYHistogram; ///< Histogram for the product of the x and y components of the dumbbell bonds.

		Graph dumbbellAverageBondXXVSTime; /**< Graph of the global average of the squares of the x components of the
												dumbbell bonds vs. MPC simulation time.*/
		Graph dumbbellAverageBondYYVSTime; /**< Graph of the global average of the squares of the y components of the
												dumbbell bonds vs. MPC simulation time.*/
		Graph dumbbellAverageBondZZVSTime; /**< Graph of the global average of the squares of the z components of the
												dumbbell bonds vs. MPC simulation time.*/

		Histogram dumbbellBondAngleWithFlowDirectionHistogram;   /**< Histogram for the angle between the dumbbell bond
																	  and the flow direction.*/
		Histogram dumbbellBondXYAngleWithFlowDirectionHistogram; /**< Histogram for the angle between the dumbbell bond
																	  vector's projection to the xy plane and the flow
																	  direction.*/
}; //class GaussianDumbbells
} //namespace Instrumentation
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
