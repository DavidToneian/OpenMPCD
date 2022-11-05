/**
 * @file
 * Declares common mathematical functions.
 */

#ifndef OPENMPCD_UTILITY_MATHEMATICALFUNCTIONS_HPP
#define OPENMPCD_UTILITY_MATHEMATICALFUNCTIONS_HPP

#include <OpenMPCD/CUDA/Macros.hpp>

namespace OpenMPCD
{
namespace Utility
{

/**
 * Defines common mathematical functions.
 */
namespace MathematicalFunctions
{

/**
 * Returns the arc cosine of the argument.
 *
 * @tparam T The data type to use.
 *
 * @param[in] x The argument \f$ x \f$, which must be in the range
 *              \f$ \left[1, 1\right] \f$.
 *
 * @return Returns \f$ \arccos \left( x \right) \f$.
 */
template<typename T>
OPENMPCD_CUDA_HOST_AND_DEVICE
T acos(const T x);

/**
 * Returns the cosine of the argument.
 *
 * @tparam T The data type to use.
 *
 * @param[in] x The argument \f$ x \f$.
 *
 * @return Returns \f$ \cos \left( x \right) \f$.
 */
template<typename T>
OPENMPCD_CUDA_HOST_AND_DEVICE
T cos(const T x);

/**
 * Returns the cosine of the product of the argument and \f$ \pi \f$.
 *
 * @tparam T The data type to use.
 *
 * @param[in] x The argument \f$ x \f$.
 *
 * @return Returns \f$ \cos \left( \pi x \right) \f$.
 */
template<typename T>
OPENMPCD_CUDA_HOST_AND_DEVICE
T cospi(const T x);

/**
 * Returns the sine of the argument.
 *
 * @tparam T The data type to use.
 *
 * @param[in] x The argument \f$ x \f$.
 *
 * @return Returns \f$ \sin \left( x \right) \f$.
 */
template<typename T>
OPENMPCD_CUDA_HOST_AND_DEVICE
T sin(const T x);

/**
 * Returns the sine of the product of the argument and \f$ \pi \f$.
 *
 * @tparam T The data type to use.
 *
 * @param[in] x The argument \f$ x \f$.
 *
 * @return Returns \f$ \sin \left( \pi x \right) \f$.
 */
template<typename T>
OPENMPCD_CUDA_HOST_AND_DEVICE
T sinpi(const T x);


/**
 * Computes both the sine and the cosine of the argument.
 *
 * @throw OpenMPCD::NULLPointerException
 *        If `OPENMPCD_DEBUG` is defined, throws if `s == nullptr` or
 *        `c == nullptr`.
 *
 * @tparam T The data type to use.
 *
 * @param[in]  x
 *             The argument \f$ x \f$.
 * @param[out] s
 *             Where to store the sine of the argument,
 *             \f$ \sin \left( x \right) \f$. Must not be `nullptr`.
 * @param[out] c
 *             Where to store the cosine of the argument,
 *             \f$ \cos \left( x \right) \f$. Must not be `nullptr`.
 */
template<typename T>
OPENMPCD_CUDA_HOST_AND_DEVICE
void sincos(const T x, T* const s, T* const c);

/**
 * Computes both the sine and the cosine of the product of the argument and
 * \f$ \pi \f$.
 *
 * @throw OpenMPCD::NULLPointerException
 *        If `OPENMPCD_DEBUG` is defined, throws if `s == nullptr` or
 *        `c == nullptr`.
 *
 * @tparam T The data type to use.
 *
 * @param[in]  x
 *             The argument \f$ x \f$.
 * @param[out] s
 *             Where to store the sine of the product of the argument and
 *             \f$ \pi \f$, \f$ \sin \left( x \pi \right) \f$.
 *             Must not be `nullptr`.
 * @param[out] c
 *             Where to store the cosine of the product of the argument and
 *             \f$ \pi \f$, \f$ \cos \left( x \pi \right) \f$.
 *             Must not be `nullptr`.
 */
template<typename T>
OPENMPCD_CUDA_HOST_AND_DEVICE
void sincospi(const T x, T* const s, T* const c);


/**
 * Returns the sqaure root of the argument.
 *
 * @tparam T The data type to use.
 *
 * @param[in] x The argument \f$ x \f$.
 *
 * @return Returns \f$ \sqrt{ x } \f$.
 */
template<typename T>
OPENMPCD_CUDA_HOST_AND_DEVICE
T sqrt(const T x);

} //namespace MathematicalFunctions
} //namespace Utility
} //namespace OpenMPCD

#include <OpenMPCD/Utility/ImplementationDetails/MathematicalFunctions.hpp>

#endif //OPENMPCD_UTILITY_MATHEMATICALFUNCTIONS_HPP
