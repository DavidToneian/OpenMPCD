/**
 * @file
 * Defines mathematical constants.
 */

#ifndef OPENMPCD_UTILITY_MATHEMATICALCONSTANTS_HPP
#define OPENMPCD_UTILITY_MATHEMATICALCONSTANTS_HPP

#include <OpenMPCD/CUDA/Macros.hpp>

namespace OpenMPCD
{
namespace Utility
{

/**
 * Holds mathematical constants.
 */
namespace MathematicalConstants
{

/**
 * Returns the value of \f$ \pi \f$.
 *
 * @tparam T The data type to return.
 */
template<typename T>
OPENMPCD_CUDA_HOST_AND_DEVICE
T pi()
{
	return 3.14159265358979323846264338327950288419716939937510582097494459230L;
}

} //namespace MathematicalConstants
} //namespace Utility
} //namespace OpenMPCD

#endif //OPENMPCD_UTILITY_MATHEMATICALCONSTANTS_HPP
