/**
 * @file
 * Defines CUDA Device code for OpenMPCD::CUDA::MPCFluid::GaussianRods
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_DEVICECODE_GAUSSIANRODS_HPP
#define OPENMPCD_CUDA_MPCFLUID_DEVICECODE_GAUSSIANRODS_HPP

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
	 * Streams the given Gaussian Rod by applying the velocity-Verlet algorithm.
	 * No boundary conditions are considered.
	 * @param[in]     particle1ID           The ID of the first particle; the partner is the one
	 *                                      with this ID incremented by 1.
	 * @param[in,out] positions             The array of MPC fluid particle positions.
	 * @param[in,out] velocities            The array of MPC fluid particle velocities.
	 * @param[in]     meanBondLength        The mean bond length.
	 * @param[in]     reducedSpringConstant The spring constant, divided by the mass of an
	 *                                      individual spring constituent particle.
	 * @param[in]     timestep              The timestep for an individual velocity-Verlet step.
	 * @param[in]     stepCount             The number of velocity-Verlet steps to perform.
	 */
	__device__ void streamGaussianRodVelocityVerlet(
		const unsigned int particle1ID,
		MPCParticlePositionType* const positions,
		MPCParticleVelocityType* const velocities,
		const FP meanBondLength,
		const FP reducedSpringConstant,
		const FP timestep,
		const unsigned int stepCount);

	/**
	 * Streams the dumbbells by applying the velocity-Verlet algorithm.
	 * No boundary conditions are considered.
	 * @param[in]     workUnitOffset        The number of dumbbells to skip.
	 * @param[in,out] positions             The array of MPC fluid particle positions.
	 * @param[in,out] velocities            The array of MPC fluid particle velocities.
	 * @param[in]     meanBondLength        The mean bond length.
	 * @param[in]     reducedSpringConstant The spring constant, divided by the mass of an
	 *                                      individual spring constituent particle.
	 * @param[in]     timestep              The timestep for an individual velocity-Verlet step.
	 * @param[in]     stepCount             The number of velocity-Verlet steps to perform.
	 */
	__global__ void streamGaussianRodsVelocityVerlet(
		const unsigned int workUnitOffset,
		MPCParticlePositionType* const positions,
		MPCParticleVelocityType* const velocities,
		const FP meanBondLength,
		const FP reducedSpringConstant,
		const FP timestep,
		const unsigned int stepCount);
} //namespace DeviceCode
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
