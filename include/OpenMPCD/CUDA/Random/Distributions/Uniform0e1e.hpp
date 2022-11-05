/**
 * @file
 * Defines the `OpenMPCD::CUDA::Random::Distributions::Uniform0e1e` class.
 */

#ifndef OPENMPCD_CUDA_RANDOM_DISTRIBUTIONS_UNIFORM0E1E_HPP
#define OPENMPCD_CUDA_RANDOM_DISTRIBUTIONS_UNIFORM0E1E_HPP

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/Random/Distributions/Uniform0e1i.hpp>


namespace OpenMPCD
{
namespace CUDA
{
namespace Random
{
namespace Distributions
{

/**
 * The uniform distribution in the open interval \f$ (0, 1) \f$.
 *
 * Neither \f$ 0 \f$ nor \f$ 1 \f$ lie within the range of possible values.
 *
 * @tparam T The underlying data type, which must be `float` or `double`.
 */
template<typename T>
class Uniform0e1e
{
public:
	/**
	 * Generates a random value sampled from the distribution.
	 *
	 * @tparam RNG The random number generator type.
	 *
	 * @param[in] rng The random number generator instance.
	 */
	template<typename RNG>
	OPENMPCD_CUDA_DEVICE
	T operator()(RNG& rng) const
	{
		Uniform0e1i<T> dist;

		T value;

		do
		{
			value = dist(rng);
		}
		while(value == 1);

		return value;
	}
}; //class Uniform0e1e

} //namespace Distributions
} //namespace Random
} //namespace CUDA
} //namespace OpenMPCD


#endif //OPENMPCD_CUDA_RANDOM_DISTRIBUTIONS_UNIFORM0E1E_HPP
