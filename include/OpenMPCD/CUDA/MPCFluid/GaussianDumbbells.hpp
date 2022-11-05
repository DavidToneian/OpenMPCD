/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::GaussianDumbbells class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_GAUSSIANDUMBBELLS_HPP
#define OPENMPCD_CUDA_MPCFLUID_GAUSSIANDUMBBELLS_HPP

#include <OpenMPCD/CUDA/MPCFluid/Base.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{

	/**
	 * Fluid consisting of Gaussian dumbbells.
	 *
	 * The two particles interact with the following potential:
	 * \f[
	 * 	V
	 * 	= \frac{K}{2} \left( \vec{r}_1 - \vec{r}_2 \right)^2
	 * \f]
	 * where \f$ \vec{r}_i \f$ is the current position of particle \f$ i \f$
	 * and \f$ K \f$ is the spring constant.
	 *
	 * The configuration group of this fluid is expected to be named
	 * `mpc.fluid.dumbbell`, and contains:
	 *   - The boolean value `analyticalStreaming`, which controls whether to
	 *     use the analytically known solution to the equations of motion (value
	 *     `true`), or whether to integrate the equations of motion using
	 *     molecular dynamics (MD) during the streaming step (value `false`);
	 *   - The floating-point value `rootMeanSquareLength`, which defines the
	 *     root of the average of the squared bond length;
	 *   - The floating-point value `zeroShearRelaxationTime`, which defines
	 *     the relaxation time at zero shear.
	 *   - If `analyticalStreaming` is `false`, the positive integer value
	 *     `mdStepCount`, which specifies how many MD steps should be performed
	 *     per MPC streaming step.
	 *
	 * The Gaussian dumbbell model is the one described in
	 * "Multiparticle collision dynamics simulations of viscoelastic fluids: Shear-thinning
	 * Gaussian dumbbells" by Bartosz Kowalik and Roland G. Winkler.
	 * Journal of Chemical Physics 138, 104903 (2013). http://dx.doi.org/10.1063/1.4792196
	 */
	class GaussianDumbbells : public Base
	{
		public:
			/**
			 * The constructor.
			 * @param[in] sim                  The simulation instance.
			 * @param[in] count                The number of fluid particles.
			 * @param[in] streamingTimestep_   The timestep for a streaming step.
			 * @param[in] rng_                 A random number generator to seed this instance's RNG with.
			 * @param[in] devMemMgr            The Device memory manager.
			 * @param[in] kT                   The fluid temperature, times Boltzmann's constant.
			 * @param[in] leesEdwardsShearRate The shear rate for the Lees-Edwards boundary conditions.
			 */
			GaussianDumbbells(const CUDA::Simulation* const sim, const unsigned int count,
			                  const FP streamingTimestep_, RNG& rng_,
			                  DeviceMemoryManager* const devMemMgr,
			                  const FP kT, const FP leesEdwardsShearRate);

			/**
			 * The destructor.
			 */
			virtual ~GaussianDumbbells()
			{
			}

		public:
			virtual unsigned int getNumberOfLogicalEntities() const
			{
				return getParticleCount() / 2;
			}

			virtual bool numberOfParticlesPerLogicalEntityIsConstant() const
			{
				return true;
			}

			virtual unsigned int getNumberOfParticlesPerLogicalEntity() const
			{
				return 2;
			}

			virtual void stream();

		private:
			/**
			 * Reads the configuration.
			 */
			void readConfiguration();

			/**
			 * Initializes the particle positions and velocities on the host.
			 * @param[in] leesEdwardsShearRate The shear rate for the Lees-Edwards boundary conditions.
			 */
			void initializeOnHost(const FP leesEdwardsShearRate);

			/**
			 * Returns the initial position of the given particle's dumbbell partner.
			 * @param[in] position1 The position of the first particle of the dumbbell.
			 * @param[in] Wi        The Weissenberg number.
			 */
			const Vector3D<MPCParticlePositionType>
				getInitialDumbbellPartnerPosition(
					const RemotelyStoredVector<MPCParticlePositionType>& position1,
					const FP Wi) const;

		private:
			bool streamAnalyticallyFlag; ///< Whether to use the analytical equations of motion or MD simulation.

			FP dumbbellRootMeanSquareLength; ///< The root-mean-square of the dumbbell bond length.
			FP zeroShearRelaxationTime;      ///< The zero-shear relaxation time of the dumbbells.
			FP reducedSpringConstant; /**< The spring constant for the spring connecting the dumbbell constituents,
			                               divided by the mass of each one constituent.*/

			unsigned int mdStepCount; ///< The number of velocity-Verlet steps in each streaming step.
	};

} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
