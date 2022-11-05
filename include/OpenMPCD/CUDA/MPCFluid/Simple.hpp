/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::Simple class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_SIMPLE_HPP
#define OPENMPCD_CUDA_MPCFLUID_SIMPLE_HPP

#include <OpenMPCD/CUDA/MPCFluid/Base.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{
	/**
	 * Fluid consisting of independent particles
	 */
	class Simple : public Base
	{
		public:
			/**
			 * The constructor.
			 * @param[in] sim                 The simulation instance.
			 * @param[in] count               The number of fluid particles.
			 * @param[in] streamingTimestep_  The timestep for a streaming step.
			 * @param[in] rng_                A random number generator to seed this instance's RNG with.
			 * @param[in] devMemMgr           The Device memory manager.
			 */
			Simple(const CUDA::Simulation* const sim, const unsigned int count,
			       const FP streamingTimestep_, RNG& rng_, DeviceMemoryManager* const devMemMgr);

			/**
			 * The destructor.
			 */
			virtual ~Simple()
			{
			}

		public:
			virtual unsigned int getNumberOfLogicalEntities() const
			{
				return getParticleCount();
			}

			virtual bool numberOfParticlesPerLogicalEntityIsConstant() const
			{
				return true;
			}

			virtual unsigned int getNumberOfParticlesPerLogicalEntity() const
			{
				return 1;
			}

			virtual void stream();

		private:
			/**
			 * Reads the configuration.
			 */
			void readConfiguration();

			/**
			 * Initializes the particle positions and velocities on the host.
			 */
			void initializeOnHost();
	};
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
