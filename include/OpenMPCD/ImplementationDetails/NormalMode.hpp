/**
 * @file
 * Implements functionality in the `OpenMPCD::NormalMode` namespace.
 */

#ifndef OPENMPCD_IMPLEMENTATIONDETAILS_NORMALMODE_HPP
#define OPENMPCD_IMPLEMENTATIONDETAILS_NORMALMODE_HPP

#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>

#include <boost/math/constants/constants.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_floating_point.hpp>

namespace OpenMPCD
{
namespace NormalMode
{

template<typename T>
const Vector3D<T> computeNormalCoordinate(
	const unsigned int i, const Vector3D<T>* const vectors,
	const std::size_t N, const T shift)
{
	BOOST_STATIC_ASSERT(boost::is_floating_point<T>::value);
		//non-floating-point `T` is probably a mistake

	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(vectors, NULLPointerException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(N != 0, InvalidArgumentException);
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(i <= N, InvalidArgumentException);

	Vector3D<T> ret(0, 0, 0);

	const T pi = boost::math::constants::pi<T>();
	const T argPart = T(i) * pi / T(N);
	for(std::size_t n = 1; n <= N; ++n)
		ret += vectors[n - 1] * cos(argPart * (n + shift));

	return ret / T(N);
}

} //namespace NormalMode
} //namespace OpenMPCD

#endif //OPENMPCD_IMPLEMENTATIONDETAILS_NORMALMODE_HPP
