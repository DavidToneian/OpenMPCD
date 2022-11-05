/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::Instrumentation::FourierTransformedVelocity::Chains class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_FOURIERTRANSFORMEDVELOCITY_CHAINS_HPP
#define OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_FOURIERTRANSFORMEDVELOCITY_CHAINS_HPP

#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/FourierTransformedVelocity/Base.hpp>

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
 * Class for measurements of Fourier-transformed velocities in MPC fluids that consist of chains of arbitrary size.
 *
 * All chains are assumed to be of the same length.
 *
 * The particles that belong to one chain are assumed to be adjacent to each other in memory.
 * E.g. chain 0 consists of particles [0,n-1], chain 1 of particles [n,2n-1], etc.
 */
class Chains : public Base
{
	public:
		/**
		 * The constructor.
		 * @param[in] chainLength_ The number of MPC fluid particles per chain.
		 * @param[in] sim          The simulation instance.
		 * @param[in] devMemMgr    The Device memory manager.
		 * @param[in] mpcFluid_    The MPC fluid to measure.
		 */
		Chains(const unsigned int chainLength_, const Simulation* const sim, DeviceMemoryManager* devMemMgr,
		       const MPCFluid::Base* const mpcFluid_);

	private:
		Chains(const Chains&); ///< The copy constructor.

	public:
		/**
		 * The destructor.
		 */
		virtual ~Chains()
		{
		}

	public:
		/**
		 * Performs measurements.
		 */
		virtual void measure();

	private:
		const unsigned int chainLength; ///< The number of MPC fluid particles per chain.
}; //class Chains
} //namespace FourierTransformedVelocity
} //namespace Instrumentation
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
