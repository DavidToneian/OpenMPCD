/**
 * @file
 * Defines the `OpenMPCD::CUDA::Random::Distributions::Uniform0e1i` class.
 */

#ifndef OPENMPCD_CUDA_RANDOM_DISTRIBUTIONS_UNIFORM0E1I_HPP
#define OPENMPCD_CUDA_RANDOM_DISTRIBUTIONS_UNIFORM0E1I_HPP

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/Random/Generators/Philox4x32-10.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>


namespace OpenMPCD
{
namespace CUDA
{
namespace Random
{

/**
 * Distributions for the generation of random numbers.
 */
namespace Distributions
{

/**
 * The uniform distribution in the left-open interval \f$ (0, 1] \f$.
 *
 * The value \f$ 0 \f$ does not lie within the range of possible values, but
 * \f$ 1 \f$ does.
 *
 * @tparam T The underlying data type, which must be `float` or `double`.
 */
template<typename T>
class Uniform0e1i
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

	/**
	 * Generates two random values sampled from the distribution.
	 *
	 * @tparam RNG The random number generator type.
	 *
	 * @param[in]  rng
	 *             The random number generator instance.
	 * @param[out] r1
	 *             Pointer to where the first random number should be stored;
	 *             must not be `nullptr`.
	 * @param[out] r2
	 *             Pointer to where the second random number should be stored;
	 *             must not be `nullptr`.
	 */
	template<typename RNG>
	OPENMPCD_CUDA_DEVICE
	void operator()(RNG& rng, T* const r1, T* const r2) const;
}; //class Uniform0e1i

///@cond
template<> template<typename RNG>
OPENMPCD_CUDA_DEVICE
float Uniform0e1i<float>::operator()(RNG& rng) const
{
	return curand_uniform(rng.getState());
}

template<> template<typename RNG>
OPENMPCD_CUDA_DEVICE
double Uniform0e1i<double>::operator()(RNG& rng) const
{
	return curand_uniform_double(rng.getState());
}


template<typename T> template<typename RNG>
OPENMPCD_CUDA_DEVICE
void Uniform0e1i<T>::operator()(
	RNG& rng, T* const r1, T* const r2) const
{
	OPENMPCD_DEBUG_ASSERT(r1 != NULL);
	OPENMPCD_DEBUG_ASSERT(r2 != NULL);

	*r1 = operator()(rng);
	*r2 = operator()(rng);
}

template<> template<>
OPENMPCD_CUDA_DEVICE inline
void Uniform0e1i<double>::operator()<Generators::Philox4x32_10>(
	Generators::Philox4x32_10& rng, double* const r1, double* const r2) const
{
	OPENMPCD_DEBUG_ASSERT(r1 != NULL);
	OPENMPCD_DEBUG_ASSERT(r2 != NULL);

	const double2 result = curand_uniform2_double(rng.getState());
	*r1 = result.x;
	*r2 = result.y;
}
///@endcond

} //namespace Distributions
} //namespace Random
} //namespace CUDA
} //namespace OpenMPCD


#endif //OPENMPCD_CUDA_RANDOM_DISTRIBUTIONS_UNIFORM0E1I_HPP
