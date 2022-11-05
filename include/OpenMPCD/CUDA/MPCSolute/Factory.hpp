/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCSolute::Factory class.
 */

#ifndef OPENMPCD_CUDA_MPCSOLUTE_FACTORY_HPP
#define OPENMPCD_CUDA_MPCSOLUTE_FACTORY_HPP

#include <OpenMPCD/CUDA/MPCSolute/Base.hpp>
#include <OpenMPCD/CUDA/Simulation.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCSolute
{

	/**
	 * Class used to construct MPC solute instances.
	 */
	class Factory
	{
		private:
			Factory(); ///< The constructor.

		public:
			/**
			 * Returns a newly constructed MPC solute.
			 * The caller is responsible for deleting the pointer.
			 * Returns nullptr if mpc.solute is not set.
			 * @throw OpenMPCD::UnimplementedException
			 * @param[in] sim    The simulation instance.
			 * @param[in] config The simulation configuration.
			 * @param[in] rng    A random number generator.
			 */
			static
			MPCSolute::Base<MPCParticlePositionType, MPCParticleVelocityType>*
			getInstance(
				CUDA::Simulation* const sim,
				const Configuration& config,
				RNG& rng);
	};

} //namespace MPCSolute
} //namespace CUDA
} //namespace OpenMPCD

#endif
