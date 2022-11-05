/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCFluid::Factory class.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_FACTORY_HPP
#define OPENMPCD_CUDA_MPCFLUID_FACTORY_HPP

#include <OpenMPCD/CUDA/MPCFluid/Base.hpp>
#include <OpenMPCD/CUDA/Simulation.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{

	/**
	 * Class used to construct MPC fluid instances.
	 */
	class Factory
	{
		private:
			Factory(); ///< The constructor.

		public:
			/**
			 * Returns a newly constructed MPC fluid.
			 *
			 * The caller is responsible for deleting the pointer.
			 * If no MPC fluid has been configured, or more precisely, if
			 * the configuration key `mpc.fluid` is not set, returns `nullptr`.
			 *
			 * @throw OpenMPCD::InvalidConfigurationException
			 *        Throws if an unknown key is set in `mpc.fluid`.
			 *
			 * @param[in] sim    The simulation instance.
			 * @param[in] config The simulation configuration.
			 * @param[in] count  The number of fluid particles.
			 * @param[in] rng    A random number generator.
			 */
			static MPCFluid::Base* getInstance(CUDA::Simulation* const sim,
			                                   const Configuration& config,
			                                   const unsigned int count,
			                                   RNG& rng);
	};

} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
