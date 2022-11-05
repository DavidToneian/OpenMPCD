/**
 * @file
 * Defines the `OpenMPCD::CUDA::Random::Distributions::StandardNormal` class.
 */

#ifndef OPENMPCD_CUDA_RANDOM_DISTRIBUTIONS_STANDARDNORMAL_HPP
#define OPENMPCD_CUDA_RANDOM_DISTRIBUTIONS_STANDARDNORMAL_HPP

#include <OpenMPCD/CUDA/Macros.hpp>


namespace OpenMPCD
{
namespace CUDA
{
namespace Random
{
namespace Distributions
{

/**
 * The standard normal distribution.
 *
 * This is the normal distribution with mean \f$ 0 \f$ and variance \f$ 1 \f$.
 *
 * @tparam T The underlying data type, which must be `float` or `double`.
 */
template<typename T>
class StandardNormal
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
	T operator()(RNG& rng) const;
}; //class StandardNormal

///@cond
template<> template<typename RNG>
OPENMPCD_CUDA_DEVICE
float StandardNormal<float>::operator()(RNG& rng) const
{
	return curand_normal(rng.getState());
}

template<> template<typename RNG>
OPENMPCD_CUDA_DEVICE
double StandardNormal<double>::operator()(RNG& rng) const
{
	return curand_normal_double(rng.getState());
}
///@endcond

} //namespace Distributions
} //namespace Random
} //namespace CUDA
} //namespace OpenMPCD


#endif //OPENMPCD_CUDA_RANDOM_DISTRIBUTIONS_STANDARDNORMAL_HPP
