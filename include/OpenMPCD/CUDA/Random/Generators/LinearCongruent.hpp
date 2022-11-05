/**
 * @file
 * Defines the `OpenMPCD::CUDA::Random::Generators::LinearCongruent` class.
 */

#ifndef OPENMPCD_CUDA_RANDOM_GENERATORS_LINEARCONGRUENT_HPP
#define OPENMPCD_CUDA_RANDOM_GENERATORS_LINEARCONGRUENT_HPP

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>
#include <OpenMPCD/Utility/CompilerDetection.hpp>

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_integral.hpp>


namespace OpenMPCD
{
namespace CUDA
{

/**
 * Contains functionality related to random numbers.
 */
namespace Random
{

/**
 * Contains (pseudo-)random number generators.
 */
namespace Generators
{

/**
 * A linear congruential generator (LCG).
 *
 * Given a starting value \f$ X_0 \f$, which is the seed, the generator produces
 * a sequence of numbers defined by
 * \f[ X_{n+1} = \left( a X_N + c \right) \bmod m \f]
 * where \f$ a \f$ is the `multiplier`, \f$ c \f$ is the `increment`, and
 * \f$ m \f$ is the `modulus`. The first value to be generated is \f$ X_1 \f$.
 *
 * @tparam T          The underlying data type, which must be integral.
 * @tparam multiplier The `multiplier` \f$ 0 < a < m \f$.
 * @tparam increment  The `increment` \f$ 0 \le c < m \f$.
 * @tparam modulus    The `modulus` \f$ 0 < m \f$.
 */
template<
	typename T,
	T multiplier,
	T increment,
	T modulus>
class LinearCongruent
{
public:
	/**
	 * The constructor.
	 *
	 * @throw OpenMPCD::InvalidArgumentError
	 *        If `OPENMPCD_DEBUG` is defined, throws if `seed >= modulus` or
	 *        if `seed < 0`.
	 *
	 * @param[in] seed The starting value \f$ 0 \le X_0 < m \f$.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	LinearCongruent(const T seed) : current(seed)
	{
		BOOST_STATIC_ASSERT(boost::is_integral<T>::value);
		BOOST_STATIC_ASSERT(0 < modulus);
		BOOST_STATIC_ASSERT(0 < multiplier);
		BOOST_STATIC_ASSERT(0 <= increment);
		BOOST_STATIC_ASSERT(multiplier < modulus);
		BOOST_STATIC_ASSERT(increment < modulus);

		OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
			seed > 0, InvalidArgumentException);
		OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
			seed < modulus, InvalidArgumentException);
	}

public:
	/**
	 * Generates the next number.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T operator()()
	{
		current = (multiplier * current + increment) % modulus;
		return current;
	}

private:
	T current; ///< The last value returned.
}; //class LinearCongruent

} //namespace Generators
} //namespace Random
} //namespace CUDA
} //namespace OpenMPCD


#endif //OPENMPCD_CUDA_RANDOM_GENERATORS_LINEARCONGRUENT_HPP
