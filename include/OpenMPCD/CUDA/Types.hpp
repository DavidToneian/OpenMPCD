/**
 * @file
 * Defines general types used in OpenMPCD's CUDA module.
 */

#ifndef OPENMPCD_CUDA_TYPES_HPP
#define OPENMPCD_CUDA_TYPES_HPP

#include <OpenMPCD/CUDA/Random/Generators/Philox4x32-10.hpp>

namespace OpenMPCD
{
namespace CUDA
{

typedef
	Random::Generators::Philox4x32_10
	GPURNG; ///< The random number generator type for GPUs.

} //namespace CUDA
} //namespace OpenMPCD

#endif //OPENMPCD_CUDA_TYPES_HPP
