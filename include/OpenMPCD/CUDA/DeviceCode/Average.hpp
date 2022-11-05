/**
 * @file
 * Defines CUDA Device functions for averaging.
 */

#ifndef OPENMPCD_CUDA_DEVICECODE_AVERAGE_HPP
#define OPENMPCD_CUDA_DEVICECODE_AVERAGE_HPP

namespace OpenMPCD
{
namespace CUDA
{
namespace DeviceCode
{

/**
 * Kernel to compute the arithmetic mean of the given values.
 *
 * Each value will enter the average with equal weight.
 *
 * @tparam T The type of values.
 *
 * @param[in]  values
 *             The values to take the average of, stored on the CUDA Device.
 *             In total, `values` must hold at least
 *             `numberOfValues * sizeof(T)` bytes.
 * @param[in]  numberOfValues
 *             The number of values to treat.
 * @param[out] output
 *             The CUDA Device pointer to save the result to.
 */
template<typename T>
	__global__ void arithmeticMean_kernel(
		const T* const values,
		const std::size_t numberOfValues,
		T* const output);

/**
 * Computes, on the CUDA Device, the arithmetic mean of the given values.
 *
 * Each value will enter the average with equal weight.
 *
 * @throw OpenMPCD::NULLPointerException
 *        Throws if `values` is `nullptr`.
 * @throw OpenMPCD::InvalidArgumentException
 *        Throws if `numberOfValues == 0`.
 * @throw OpenMPCD::NULLPointerException
 *        Throws if `output` is `nullptr`.
 * @throw OpenMPCD::InvalidArgumentException
 *        If OPENMPCD_DEBUG is defined,
 *        throws if `values` is not a Device pointer.
 * @throw OpenMPCD::InvalidArgumentException
 *        If OPENMPCD_DEBUG is defined,
 *        throws if `output` is not a Device pointer.
 *
 * @tparam T The type of values.
 *
 * @param[in]  values
 *             The values to take the average of, stored on the CUDA Device.
 *             In total, `values` must hold at least
 *             `numberOfValues * sizeof(T)` bytes.
 * @param[in]  numberOfValues
 *             The number of values to treat.
 * @param[out] output
 *             The CUDA Device pointer to save the result to.
 */
template<typename T>
	void arithmeticMean(
		const T* const values,
		const std::size_t numberOfValues,
		T* const output);

} //namespace DeviceCode
} //namespace CUDA
} //namespace OpenMPCD

#include <OpenMPCD/CUDA/DeviceCode/ImplementationDetails/Average.hpp>

#endif //OPENMPCD_CUDA_DEVICECODE_AVERAGE_HPP
