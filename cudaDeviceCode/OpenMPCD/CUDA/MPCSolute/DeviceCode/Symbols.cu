/**
 * @file
 * Defines CUDA constant symbols for OpenMPCD::CUDA::MPCSolute
 */

#ifndef OPENMPCD_CUDA_MPCSOLUTE_DEVICECODE_SYMBOLS_HPP
#define OPENMPCD_CUDA_MPCSOLUTE_DEVICECODE_SYMBOLS_HPP

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCSolute
{
namespace DeviceCode
{
	__constant__ unsigned int soluteParticleCount; ///< The number of MPC Solute particles.

} //namespace DeviceCode
} //namespace MPCSolute
} //namespace CUDA
} //namespace OpenMPCD

#endif
