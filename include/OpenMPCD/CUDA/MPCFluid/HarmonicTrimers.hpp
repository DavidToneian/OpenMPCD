/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::HarmonicTrimers class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_HARMONICTRIMERS_HPP
#define OPENMPCD_CUDA_MPCFLUID_HARMONICTRIMERS_HPP

#include <OpenMPCD/CUDA/MPCFluid/Base.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{
	/**
	 * Fluid consisting of three particles, with two harmonic springs coupling them.
	 *
	 * The three particles interact with the following potential:
	 * \f[
	 * 	V
	 * 	= \frac{K_1}{2} \left( \vec{r}_1 - \vec{r}_2 \right)^2 +
	 * 	  \frac{K_2}{2} \left( \vec{r}_2 - \vec{r}_3 \right)^2
	 * \f]
	 * where \f$ \vec{r}_i \f$ is the current position of particle \f$ i \f$.
	 *
	 * The configuration group of this fluid is expected to be named
	 * `mpc.fluid.harmonicTrimers`, and contains:
	 *   - The floating-point values `springConstant1` and `springConstant2`,
	 *     which define the spring constants \f$ K_1 \f$ and \f$ K_2 \f$ in the
	 *     interaction potential \f$ V \f$;
	 *   - The boolean value `analyticalStreaming`; currently, the only
	 *     supported value is `false`, which means that molecular dynamics (MD)
	 *     is going to be used to integrate the equations of motion during the
	 *     streaming step;
	 *   - If `analyticalStreaming` is `false`, the positive integer value
	 *     `mdStepCount`, which specifies how many MD steps should be performed
	 *     per MPC streaming step.
	 */
	class HarmonicTrimers : public Base
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
			HarmonicTrimers(const CUDA::Simulation* const sim, const unsigned int count,
			                const FP streamingTimestep_, RNG& rng_, DeviceMemoryManager* const devMemMgr);

			/**
			 * The destructor.
			 */
			virtual ~HarmonicTrimers()
			{
			}

		public:
			virtual unsigned int getNumberOfLogicalEntities() const
			{
				return getParticleCount() / 3;
			}

			virtual bool numberOfParticlesPerLogicalEntityIsConstant() const
			{
				return true;
			}

			virtual unsigned int getNumberOfParticlesPerLogicalEntity() const
			{
				return 3;
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
			FP springConstant1; ///< The spring constant between particles 1 and 2.
			FP springConstant2; ///< The spring constant between particles 2 and 3.

			bool streamAnalyticallyFlag; ///< Whether to use the analytical equations of motion or MD simulation.
			unsigned int mdStepCount;    ///< The number of velocity-Verlet steps in each streaming step.
	};

} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
