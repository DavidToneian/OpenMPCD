/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::GaussianRods class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_GAUSSIANRODS_HPP
#define OPENMPCD_CUDA_MPCFLUID_GAUSSIANRODS_HPP

#include <OpenMPCD/CUDA/MPCFluid/Base.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{

/**
 * Fluid consisting of Gaussian rods with a certain mean length.
 *
 * If \f$ \vec{R} \f$ is the bond vector pointing from the first particle to the
 * second, the potential is
 * \f$ U = \frac{k}{2} \left( \left| R \right| - L \right)^2 \f$,
 * where \f$ L \f$ is the mean bond length and \f$ k \f$ is called the spring
 * constant.
 */
class GaussianRods : public Base
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
		GaussianRods(const CUDA::Simulation* const sim, const unsigned int count,
		             const FP streamingTimestep_, RNG& rng_,
		             DeviceMemoryManager* const devMemMgr);

		/**
		 * The destructor.
		 */
		virtual ~GaussianRods()
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
		 */
		void initializeOnHost();

		/**
		 * Returns the initial position of the given particle's partner.
		 * @param[in] position1 The position of the first particle of the dumbbell.
		 */
		const Vector3D<MPCParticlePositionType>
			getInitialPartnerPosition(
				const RemotelyStoredVector<MPCParticlePositionType>& position1) const;

	private:
		FP meanBondLength; ///< The mean bond length \f$ L \f$.
		FP springConstant; ///< The spring constant \f$ k \f$.

		unsigned int mdStepCount; ///< The number of velocity-Verlet steps in each streaming step.
}; //class GaussianRods

} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
