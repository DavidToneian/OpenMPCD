/**
 * @file
 * Defines CUDA constant symbols for OpenMPCD::CUDA::MPCFluid
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_DEVICECODE_SYMBOLS_HPP
#define OPENMPCD_CUDA_MPCFLUID_DEVICECODE_SYMBOLS_HPP

#include <OpenMPCD/Types.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{
namespace DeviceCode
{
	extern __constant__ unsigned int mpcParticleCount; ///< The number of MPC fluid particles.

	extern __constant__ FP omega;
	extern __constant__ FP cos_omegaTimesTimestep;
	extern __constant__ FP sin_omegaTimesTimestep;
} //namespace DeviceCode
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
