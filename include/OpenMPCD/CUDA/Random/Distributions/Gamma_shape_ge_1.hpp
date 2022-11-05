/**
 * @file
 * Defines the `OpenMPCD::CUDA::Random::Distributions::Gamma_shape_ge_1` class.
 */

#ifndef OPENMPCD_CUDA_RANDOM_DISTRIBUTIONS_GAMMA_SHAPE_GE_1_HPP
#define OPENMPCD_CUDA_RANDOM_DISTRIBUTIONS_GAMMA_SHAPE_GE_1_HPP

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/Random/Distributions/StandardNormal.hpp>
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
 * The gamma distribution, with shape parameter \f$ k \f$ assumed to satisfy
 * \f$ k \ge 1 \f$.
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
 * where the parameters \f$ k \ge 1 \f$ and \f$ \theta \f$ are called `shape`
 * and `scale`, respectively, and \f$ \Gamma \f$ is the gamma function.
 *
 * For further information, see e.g. @cite Devroye1986, chapter IX.3.
 *
 * @tparam T The underlying data type, which must be a floating-point type.
 */
template<typename T>
class Gamma_shape_ge_1
{
public:
	/**
	 * The constructor.
	 *
	 * @throw OpenMPCD::InvalidArgumentError
	 *        If `OPENMPCD_DEBUG` is defined, throws if `shape < 1` or
	 *        `scale < 0`.
	 *
	 * @param[in] shape The shape parameter \f$ k \ge 1 \f$.
	 * @param[in] scale The scale parameter \f$ \theta > 0 \f$.
	 */
	OPENMPCD_CUDA_DEVICE
	Gamma_shape_ge_1(const T shape, const T scale)
		: k(shape), theta(scale),
		  d(k - T(1.0)/3), c(1 / sqrt(9 * d))
	{
		BOOST_STATIC_ASSERT(boost::is_floating_point<T>::value);

		OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(k >= 1, InvalidArgumentException);
		OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(theta > 0, InvalidArgumentException);

		OPENMPCD_DEBUG_ASSERT(d == shape - T(1.0) / 3);
		OPENMPCD_DEBUG_ASSERT(c == 1 / sqrt(9 * d));
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
		Distributions::Uniform0e1e<T> uniform0e1e;
		Distributions::StandardNormal<T> standardNormal;

		for(;;)
		{
			T x;
			T v;
			do
			{
				x = standardNormal(rng);
				v = 1 + c * x;
			}
			while(v <= 0);

			v = v * v * v;

			const T u = uniform0e1e(rng);

			if(u < 1 - 0.0331 * x * x * x * x)
				return theta * d * v;

			const T criterion = 0.5 * x * x + d * ( 1 - v + log(v) );
			if(log(u) < criterion)
				return theta * d * v;
		}
	}

private:
	const T k;     ///< The shape parameter.
	const T theta; ///< The scale parameter.

	const T d; ///< Precomputed parameter for the implementation.
	const T c; ///< Precomputed parameter for the implementation.
}; //class Gamma_shape_ge_1

} //namespace Distributions
} //namespace Random
} //namespace CUDA
} //namespace OpenMPCD


#endif //OPENMPCD_CUDA_RANDOM_DISTRIBUTIONS_GAMMA_SHAPE_GE_1_HPP
