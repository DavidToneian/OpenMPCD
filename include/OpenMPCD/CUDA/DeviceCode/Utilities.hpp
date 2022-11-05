/**
 * @file
 * Defines various CUDA device code utilities.
 */

#ifndef OPENMPCD_CUDA_DEVICECODE_UTILITIES_HPP
#define OPENMPCD_CUDA_DEVICECODE_UTILITIES_HPP

#include <OpenMPCD/CUDA/Macros.hpp>

#include <boost/core/enable_if.hpp>
#include <boost/type_traits/is_floating_point.hpp>
#include <boost/type_traits/is_integral.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace DeviceCode
{
	/**
	 * Atomically adds increment to target.
	 * @param[in] target    The address of the value to increment.
	 * @param[in] increment The value to add to target.
	 */
	OPENMPCD_CUDA_DEVICE
	void atomicAdd(double* const target, const double increment);

	/**
	 * Atomically adds increment to target.
	 * @param[in] target    The address of the value to increment.
	 * @param[in] increment The value to add to target.
	 */
	OPENMPCD_CUDA_DEVICE
	void atomicAdd(float* const target, const float increment);


	/**
	 * The power function.
	 *
	 * @tparam B The type of the `base` argument, which must be an integral
	 *           type.
	 *
	 * @param[in] base     The base.
	 * @param[in] exponent The exponent.
	 */
	template<typename B>
	OPENMPCD_CUDA_HOST_AND_DEVICE
	typename boost::enable_if<boost::is_integral<B>, double>::type
	pow(const B base, const double exponent)
	{
		return ::pow(static_cast<double>(base), exponent);
	}

	/**
	 * The power function.
	 *
	 * @param[in] base     The base.
	 * @param[in] exponent The exponent.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE inline
	double pow(const double base, const double exponent)
	{
		return ::pow(base, exponent);
	}
} //namespace DeviceCode
} //namespace CUDA
} //namespace OpenMPCD

#endif
