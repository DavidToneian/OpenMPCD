/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::Instrumentation::FourierTransformedVelocity::Doublets class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_FOURIERTRANSFORMEDVELOCITY_DOUBLETS_HPP
#define OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_FOURIERTRANSFORMEDVELOCITY_DOUBLETS_HPP

#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/FourierTransformedVelocity/Base.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Simple.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{
namespace Instrumentation
{
namespace FourierTransformedVelocity
{
/**
 * Class for measurements of Fourier-transformed velocities in MPC fluids that consist of doublets of particles.
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
		Doublets(const Simulation* const sim, DeviceMemoryManager* devMemMgr, const MPCFluid::Base* const mpcFluid_);

	private:
		Doublets(const Simple&); ///< The copy constructor.

	public:
		/**
		 * The destructor.
		 */
		virtual ~Doublets()
		{
		}

	public:
		/**
		 * Performs measurements.
		 */
		virtual void measure();
}; //class Doublets
} //namespace FourierTransformedVelocity
} //namespace Instrumentation
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
