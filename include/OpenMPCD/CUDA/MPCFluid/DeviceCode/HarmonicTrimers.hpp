/**
 * @file
 * Defines CUDA Device code for OpenMPCD::CUDA::MPCFluid::HarmonicTrimers
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_DEVICECODE_HARMONICTRIMERS_HPP
#define OPENMPCD_CUDA_MPCFLUID_DEVICECODE_HARMONICTRIMERS_HPP

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
	 * Streams the given trimer via velocity-Verlet integration.
	 * No boundary conditions are considered.
	 * This function assumes that the individual particles in the trimer all have mass 1.
	 * @param[in]     workUnitOffset         The number of trimers to skip.
	 * @param[in,out] positions              The array of MPC fluid particle positions.
	 * @param[in,out] velocities             The array of MPC fluid particle velocities.
	 * @param[in]     reducedSpringConstant1 The spring constant for the spring between particles 1 and 2,
	 *                                       divided by the mass of an individual constituent particle.
	 * @param[in]     reducedSpringConstant2 The spring constant for the spring between particles 2 and 3,
	 *                                       divided by the mass of an individual constituent particle.
	 * @param[in]     timestep               The timestep for an individual velocity-Verlet step.
	 * @param[in]     stepCount              The number of velocity-Verlet steps to perform.
	 */
	__global__ void streamHarmonicTrimerVelocityVerlet(
		const unsigned int workUnitOffset,
		MPCParticlePositionType* const positions,
		MPCParticleVelocityType* const velocities,
		const FP reducedSpringConstant1,
		const FP reducedSpringConstant2,
		const FP timestep,
		const unsigned int stepCount);

} //namespace DeviceCode
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
