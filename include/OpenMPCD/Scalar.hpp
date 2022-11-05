/**
 * @file
 * Defines helper functions for scalar types.
 */

#ifndef OPENMPCD_SCALAR_HPP
#define OPENMPCD_SCALAR_HPP

#include <OpenMPCD/CUDA/Macros.hpp>

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/utility/enable_if.hpp>

#include <complex>

namespace OpenMPCD
{

/**
 * Holds helper functions for scalar types.
 */
namespace Scalar
{
///@{
/**
 * Returns the real part of the given value.
 *
 * @tparam T The type of the value to consider.
 *
 * @param[in] val The value to consider.
 */

template<typename T>
OPENMPCD_CUDA_HOST_AND_DEVICE
typename boost::enable_if<boost::is_floating_point<T>, T>::type
	getRealPart(const T& val)
{
	BOOST_STATIC_ASSERT(boost::is_floating_point<T>::value);

	return val;
}

template<typename T>
OPENMPCD_CUDA_HOST_AND_DEVICE
typename boost::enable_if<boost::is_floating_point<T>, T>::type
	getRealPart(const std::complex<T>& val)
{
	BOOST_STATIC_ASSERT(boost::is_floating_point<T>::value);

	return val.real();
}
///@}

///@{
/**
 * Returns whether the given value is zero.
 *
 * @tparam T The type of the value to consider.
 *
 * @param[in] val The value to consider.
 */

template<typename T> OPENMPCD_CUDA_HOST_AND_DEVICE
typename boost::enable_if<boost::is_floating_point<T>, bool>::type
	isZero(const T& val)
{
	BOOST_STATIC_ASSERT(boost::is_floating_point<T>::value);

	return val == 0;
}

template<typename T> OPENMPCD_CUDA_HOST_AND_DEVICE
typename boost::enable_if<boost::is_floating_point<T>, bool>::type
	isZero(const std::complex<T>& val)
{
	BOOST_STATIC_ASSERT(boost::is_floating_point<T>::value);

	return val == std::complex<T>(0, 0);
}
///@}

} //namespace Scalar
} //namespace OpenMPCD

#endif
