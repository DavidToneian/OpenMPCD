/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::Instrumentation::GaussianRods class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_GAUSSIANRODS_HPP
#define OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_GAUSSIANRODS_HPP

#include <OpenMPCD/CUDA/MPCFluid/GaussianRods.hpp>
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
 * Instrumentation for Gaussian Rod fluids.
 */
class GaussianRods : public Base
{
	public:
		/**
		 * The constructor.
		 * @param[in] sim       The simulation instance.
		 * @param[in] devMemMgr The Device memory manager.
		 * @param[in] mpcFluid_ The MPC fluid to measure.
		 */
		GaussianRods(
			const Simulation* const sim, DeviceMemoryManager* const devMemMgr,
			const MPCFluid::GaussianRods* const mpcFluid_);

		/**
		 * The destructor.
		 */
		virtual ~GaussianRods()
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
		const MPCFluid::GaussianRods* const mpcFluid; ///< The fluid.

		Histogram bondLengthHistogram; ///< Histogram for the bond lengths.
}; //class GaussianRods
} //namespace Instrumentation
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
