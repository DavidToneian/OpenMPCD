/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::Instrumentation::VelocityAutocorrelation::Triplets class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_VELOCITYAUTOCORRELATION_TRIPLETS_HPP
#define OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_VELOCITYAUTOCORRELATION_TRIPLETS_HPP

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
 * Class for measurements of velocity autocorrelation in MPC fluids that consist of triplets of particles.
 *
 * The particles that belong to one triplet are assumed to be adjacent to each other in memory.
 * E.g. triplet 0 consists of particles 0 and 1 and 2, triplet 1 of particles 3 and 4 and 5, etc.
 */
class Triplets : public Base
{
	public:
		/**
		 * The constructor.
		 * @param[in] sim       The simulation instance.
		 * @param[in] devMemMgr The Device memory manager.
		 * @param[in] mpcFluid_ The MPC fluid to measure.
		 */
		Triplets(const Simulation* const sim, DeviceMemoryManager* devMemMgr, const MPCFluid::Base* const mpcFluid_);

	private:
		Triplets(const Triplets&); ///< The copy constructor.

	public:
		/**
		 * The destructor.
		 */
		virtual ~Triplets()
		{
		}

	protected:
		/**
		 * Populates the currentVelocities buffer with the current velocities of the fluid constituents.
		 */
		virtual void populateCurrentVelocities();
}; //class Triplets
} //namespace VelocityAutocorrelation
} //namespace Instrumentation
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
