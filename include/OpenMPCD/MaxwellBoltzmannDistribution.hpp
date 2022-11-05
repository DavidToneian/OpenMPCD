/**
 * @file
 * Defines the MaxwellBoltzmannDistribution class.
 */

#ifndef OPENMPCD_MAXWELLBOLTZMANNDISTRIBUTION_HPP
#define OPENMPCD_MAXWELLBOLTZMANNDISTRIBUTION_HPP

#include <OpenMPCD/Types.hpp>

namespace OpenMPCD
{
	/**
	 * Represents a Maxwell-Boltzmann distribution.
	 */
	class MaxwellBoltzmannDistribution
	{
		private:
			MaxwellBoltzmannDistribution(); ///< The constructor.

		public:
			/**
			 * Generates a random number drawn from the Maxwell-Boltzmann distribution.
			 * The Maxwell-Boltzmann distribution is given by
			 * \f$ \left(v\right) = \left( \frac{m}{2\pi kT} \right)^{\frac{3}{2}}
			 * \cdot 4\pi v^2 \exp\left(- \frac{m v^2}{2 kT} \right) \f$
			 * @tparam RNG The type of the random number generator.
			 * @param[in] m   The mass parameter.
			 * @param[in] kT  The Boltzmann constant times the temperature.
			 * @param[in] rng The random number generator to use.
			 */
			template<typename RNG> FP getRandomMaxwell(const FP m, const FP kT, RNG& rng);
	};
}

#endif
