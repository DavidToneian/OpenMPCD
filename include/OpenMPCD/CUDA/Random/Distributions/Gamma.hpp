/**
 * @file
 * Defines the `OpenMPCD::CUDA::Random::Distributions::Gamma` class.
 */

#ifndef OPENMPCD_CUDA_RANDOM_DISTRIBUTIONS_GAMMA_HPP
#define OPENMPCD_CUDA_RANDOM_DISTRIBUTIONS_GAMMA_HPP

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/Random/Distributions/Gamma_shape_ge_1.hpp>
#include <OpenMPCD/CUDA/Random/Distributions/Uniform0e1e.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_floating_point.hpp>


namespace OpenMPCD
{
namespace CUDA
{
namespace Random
{
namespace Distributions
{

/**
 * The gamma distribution.
 *
 * This is the distribution which has the probability density function
 * \f[
 * 	f \left(x ; k, \theta \right)
 * 	=
 * 	\frac
 * 	{
 * 		x^{k-1}
 * 		\exp \left( - \frac{ x }{ \theta } \right)
 * 	}
 * 	{
 * 		\Gamma \left( k \right)
 * 		\theta^k
 * 	}
 * \f]
 * where the parameters \f$ k \f$ and \f$ \theta \f$ are called `shape` and
 * `scale`, respectively, and \f$ \Gamma \f$ is the gamma function.
 *
 * For further information, see e.g. @cite Devroye1986, chapter IX.3.
 *
 * @tparam T The underlying data type, which must be a floating-point type.
 */
template<typename T>
class Gamma
{
public:
	/**
	 * The constructor.
	 *
	 * @throw OpenMPCD::InvalidArgumentError
	 *        If `OPENMPCD_DEBUG` is defined, throws if `shape < 0` or
	 *        `scale < 0`.
	 *
	 * @param[in] shape The shape parameter \f$ k > 0 \f$.
	 * @param[in] scale The scale parameter \f$ \theta > 0 \f$.
	 */
	OPENMPCD_CUDA_DEVICE
	Gamma(const T shape, const T scale)
		: gamma_shape_ge_1(shape >= 1 ? shape : 1 + shape, scale),
		  inverse_k(shape >= 1 ? 0 : 1 / shape)
	{
		BOOST_STATIC_ASSERT(boost::is_floating_point<T>::value);

		OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(shape > 0, InvalidArgumentException);
		OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(scale > 0, InvalidArgumentException);
	}

public:
	/**
	 * Generates a random value sampled from the distribution.
	 *
	 * This function implements the algorithm in @cite Marsaglia2000, and
	 * additionally making use of the scaling property of the gamma
	 * distribution, i.e. the fact that if \f$ X \f$ is gamma-distributed with
	 * shape \f$ k \f$ and scale \f$ \theta \f$, then \f$ cX \f$ is
	 * gamma-distributed with shape \f$ k \f$ and scale \f$ c \theta \f$, given
	 * that \f$ c > 0 \f$ (see @cite Devroye1986, chapter IX.3).
	 *
	 * @tparam RNG The random number generator type.
	 *
	 * @param[in] rng The random number generator instance.
	 */
	template<typename RNG>
	OPENMPCD_CUDA_DEVICE
	T operator()(RNG& rng) const
	{
		if(inverse_k == 0)
			return gamma_shape_ge_1(rng);

		Distributions::Uniform0e1e<T> uniform0e1e;

		return gamma_shape_ge_1(rng) * pow(uniform0e1e(rng), inverse_k);
	}

private:
	const Gamma_shape_ge_1<T> gamma_shape_ge_1;
		///< The underlying Gamma distribution with \f$ k \ge 1 \f$.

	const T inverse_k;
		/**< The inverse \f$ k^{-1} \f$ of the shape parameter \f$ k \f$
		     if \f$ 0 < k < 1 \f$, and \f$ 0 \f$ otherwise, i.e.
		     if \f$ k \ge 1 \f$.*/
}; //class Gamma

} //namespace Distributions
} //namespace Random
} //namespace CUDA
} //namespace OpenMPCD


#endif //OPENMPCD_CUDA_RANDOM_DISTRIBUTIONS_GAMMA_HPP
