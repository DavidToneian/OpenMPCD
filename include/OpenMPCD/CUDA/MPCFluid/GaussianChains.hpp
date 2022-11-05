/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::GaussianChains class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_GAUSSIANCHAINS_HPP
#define OPENMPCD_CUDA_MPCFLUID_GAUSSIANCHAINS_HPP

#include <OpenMPCD/CUDA/MPCFluid/Base.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{

/**
 * Generalization of GaussianDumbbells to chains with an arbitrary number of constituent particles.
 *
 * The particles interact with the following potential:
 * \f[
 * 	V
 * 	= \frac{K}{2} \sum_{i = 1}^{N-1} \left( \vec{r}_i - \vec{r}_{i+1} \right)^2
 * \f]
 * where \f$ \vec{r}_i \f$ is the current position of particle \f$ i \f$, and
 * \f$ N \f$ is the number of particles in a chain.
 *
 * The configuration group of this fluid is expected to be named
 * `mpc.fluid.gaussianChains`, and contains:
 *   - The positive integer value `particlesPerChain`, which defines \f$ N \f$;
 *   - The floating-point value `springConstant`, which defines the spring
 *     constant \f$ K \f$ in the interaction potential \f$ V \f$;
 *   - The integer value `mdStepCount`, which specifies how many molecular
 *     dynamics steps should be performed per MPC streaming step to integrate
 *     the equations of motion that correspond to the potential \f$ V \f$.
 */
class GaussianChains : public Base
{
	public:
		/**
		 * The constructor.
		 * @param[in] sim                  The simulation instance.
		 * @param[in] count                The number of fluid particles.
		 * @param[in] streamingTimestep_   The timestep for a streaming step.
		 * @param[in] rng_                 A random number generator to seed this instance's RNG with.
		 * @param[in] devMemMgr            The Device memory manager.
		 */
		GaussianChains(const CUDA::Simulation* const sim, const unsigned int count,
		               const FP streamingTimestep_, RNG& rng_,
		               DeviceMemoryManager* const devMemMgr);

		/**
		 * The destructor.
		 */
		virtual ~GaussianChains();

	public:
		virtual unsigned int getNumberOfLogicalEntities() const
		{
			return getParticleCount() / particlesPerChain;
		}

		virtual bool numberOfParticlesPerLogicalEntityIsConstant() const
		{
			return true;
		}

		virtual unsigned int getNumberOfParticlesPerLogicalEntity() const
		{
			return particlesPerChain;
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

	private:
		unsigned int particlesPerChain; ///< The number of MPC particles per chain.
		FP springConstant; ///< The spring constant \f$ k \f$.

		unsigned int mdStepCount; ///< The number of velocity-Verlet steps in each streaming step.

		FP* d_velocityVerletAccelerationBuffer; /**< Buffer for the initial accelerations in the velocity
		                                             Verlet algorithm.*/
}; //class GaussianChains

} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
