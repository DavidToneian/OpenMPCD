/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::Instrumentation::VelocityAutocorrelation::Doublets class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_VELOCITYAUTOCORRELATION_DOUBLETS_HPP
#define OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_VELOCITYAUTOCORRELATION_DOUBLETS_HPP

#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/VelocityAutocorrelation/Base.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{
namespace Instrumentation
{
namespace VelocityAutocorrelation
{
/**
 * Class for measurements of velocity autocorrelation in MPC fluids that consist of doublets of particles.
 *
 * The particles that belong to one doublet are assumed to be adjacent to each other in memory.
 * E.g. doublet 0 consists of particles 0 and 1, doublet 1 of particles 2 and 3, etc.
 */
class Doublets : public Base
{
	public:
		/**
		 * The constructor.
		 * @param[in] sim       The simulation instance.
		 * @param[in] devMemMgr The Device memory manager.
		 * @param[in] mpcFluid_ The MPC fluid to measure.
		 */
		Doublets(
			const Simulation* const sim, DeviceMemoryManager* const devMemMgr,
			const MPCFluid::Base* const mpcFluid_);

	private:
		Doublets(const Doublets&); ///< The copy constructor.

	public:
		/**
		 * The destructor.
		 */
		virtual ~Doublets()
		{
		}

	protected:
		/**
		 * Populates the currentVelocities buffer with the current velocities of the fluid constituents.
		 */
		virtual void populateCurrentVelocities();
}; //class Doublets
} //namespace VelocityAutocorrelation
} //namespace Instrumentation
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
