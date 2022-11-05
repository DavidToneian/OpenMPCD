/**
 * @file
 * Defines CUDA Device code for OpenMPCD::CUDA::MPCFluid::Simple
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_DEVICECODE_SIMPLE_HPP
#define OPENMPCD_CUDA_MPCFLUID_DEVICECODE_SIMPLE_HPP

#include <OpenMPCD/Types.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{
namespace DeviceCode
{
	/**
	 * Streams the given simple, independent MPC particle.
	 * No boundary conditions are considered.
	 * @param[in]     workUnitOffset  The number of MPC fluid particles to skip.
	 * @param[in,out] positions       The array of MPC fluid particle positions.
	 * @param[in,out] velocities      The array of MPC fluid particle velocities.
	 */
	__global__ void streamSimpleMPCParticle(
		const unsigned int workUnitOffset,
		MPCParticlePositionType* const positions,
		const MPCParticleVelocityType* const velocities);

} //namespace DeviceCode
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
