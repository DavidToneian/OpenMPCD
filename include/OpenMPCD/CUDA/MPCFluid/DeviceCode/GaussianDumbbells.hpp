/**
 * @file
 * Defines CUDA Device code for OpenMPCD::CUDA::MPCFluid::GaussianDumbbells
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_DEVICECODE_GAUSSIANDUMBBELLS_HPP
#define OPENMPCD_CUDA_MPCFLUID_DEVICECODE_GAUSSIANDUMBBELLS_HPP

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
	 * Sets the constant symbols.
	 * @param[in] omega_   The omega factor. (see http://dx.doi.org/10.1063/1.4792196)
	 * @param[in] timestep The timestep for the MPC fluid streaming step.
	 */
	void setGaussianDumbbellSymbols(const FP omega_, const FP timestep);


	/**
	 * Streams the given dumbbell by applying the analytical solution of the equations of motion.
	 * No boundary conditions are considered.
	 * @param[in]     particle1ID The ID of the first particle; the partner is the one with this ID incremented by 1.
	 * @param[in,out] positions   The array of MPC fluid particle positions.
	 * @param[in,out] velocities  The array of MPC fluid particle velocities.
	 */
	__device__ void streamDumbbellAnalytically(
		const unsigned int particle1ID,
		MPCParticlePositionType* const positions,
		MPCParticleVelocityType* const velocities);

	/**
	 * Streams the dumbbells by applying the analytical solution of the equations of motion.
	 * No boundary conditions are considered.
	 * @param[in]     workUnitOffset The number of dumbbells to skip.
	 * @param[in,out] positions      The array of MPC fluid particle positions.
	 * @param[in,out] velocities     The array of MPC fluid particle velocities.
	 */
	__global__ void streamDumbbellsAnalytically(
		const unsigned int workUnitOffset,
		MPCParticlePositionType* const positions,
		MPCParticleVelocityType* const velocities);

	/**
	 * Streams the given dumbbell by applying the velocity-Verlet algorithm.
	 * No boundary conditions are considered.
	 * @param[in]     particle1ID           The ID of the first particle; the partner is the one with this ID incremented by 1.
	 * @param[in,out] positions             The array of MPC fluid particle positions.
	 * @param[in,out] velocities            The array of MPC fluid particle velocities.
	 * @param[in]     reducedSpringConstant The spring constant connecting the two dumbbell contituents,
	 *                                      divided by the mass of each one constituent.
	 * @param[in]     timestep              The timestep for an individual velocity-Verlet step.
	 * @param[in]     stepCount             The number of velocity-Verlet steps to perform.
	 */
	__device__ void streamDumbbellVelocityVerlet(
		const unsigned int particle1ID,
		MPCParticlePositionType* const positions,
		MPCParticleVelocityType* const velocities,
		const FP reducedSpringConstant,
		const FP timestep,
		const unsigned int stepCount);

	/**
	 * Streams the dumbbells by applying the velocity-Verlet algorithm.
	 * No boundary conditions are considered.
	 * @param[in]     workUnitOffset        The number of dumbbells to skip.
	 * @param[in,out] positions             The array of MPC fluid particle positions.
	 * @param[in,out] velocities            The array of MPC fluid particle velocities.
	 * @param[in]     reducedSpringConstant The spring constant connecting the two dumbbell contituents,
	 *                                      divided by the mass of each one constituent.
	 * @param[in]     timestep              The timestep for an individual velocity-Verlet step.
	 * @param[in]     stepCount             The number of velocity-Verlet steps to perform.
	 */
	__global__ void streamDumbbellsVelocityVerlet(
		const unsigned int workUnitOffset,
		MPCParticlePositionType* const positions,
		MPCParticleVelocityType* const velocities,
		const FP reducedSpringConstant,
		const FP timestep,
		const unsigned int stepCount);
} //namespace DeviceCode
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
