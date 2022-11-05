/**
 * @file
 * Defines CUDA Device code for OpenMPCD::CUDA::MPCFluid::GaussianChains
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_DEVICECODE_GAUSSIANCHAINS_HPP
#define OPENMPCD_CUDA_MPCFLUID_DEVICECODE_GAUSSIANCHAINS_HPP

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
	 * Streams the given Gaussian Chain by applying the velocity-Verlet algorithm.
	 * No boundary conditions are considered.
	 * @param[in]     particle1ID           The ID of the first particle; the partner are
	 *                                      obtained by successive incrementing.
	 * @param[in,out] positions             The array of MPC fluid particle positions.
	 * @param[in,out] velocities            The array of MPC fluid particle velocities.
	 * @param[in,out] accelerationBuffer    Buffer used to store initial accelerations during the Velocity
	 *                                      Verlet integration. This has to have at least as many elements
	 *                                      as there are MPC fluid particles.
	 * @param[in]     particlesPerChain     The number of particles per chain.
	 * @param[in]     reducedSpringConstant The spring constant, divided by the mass of an
	 *                                      individual spring constituent particle.
	 * @param[in]     timestep              The timestep for an individual velocity-Verlet step.
	 * @param[in]     stepCount             The number of velocity-Verlet steps to perform.
	 */
	__device__ void streamGaussianChainVelocityVerlet(
		const unsigned int particle1ID,
		MPCParticlePositionType* const positions,
		MPCParticleVelocityType* const velocities,
		FP* const accelerationBuffer,
		const unsigned int particlesPerChain,
		const FP reducedSpringConstant,
		const FP timestep,
		const unsigned int stepCount);

	/**
	 * Streams the dumbbells by applying the velocity-Verlet algorithm.
	 * No boundary conditions are considered.
	 * @param[in]     workUnitOffset        The number of chains to skip.
	 * @param[in,out] positions             The array of MPC fluid particle positions.
	 * @param[in,out] velocities            The array of MPC fluid particle velocities.
	 * @param[in,out] accelerationBuffer    Buffer used to store initial accelerations during the Velocity
	 *                                      Verlet integration. This has to have at least as many elements
	 *                                      as there are MPC fluid particles.
	 * @param[in]     particlesPerChain     The number of particles per chain.
	 * @param[in]     reducedSpringConstant The spring constant, divided by the mass of an
	 *                                      individual spring constituent particle.
	 * @param[in]     timestep              The timestep for an individual velocity-Verlet step.
	 * @param[in]     stepCount             The number of velocity-Verlet steps to perform.
	 */
	__global__ void streamGaussianChainsVelocityVerlet(
		const unsigned int workUnitOffset,
		MPCParticlePositionType* const positions,
		MPCParticleVelocityType* const velocities,
		FP* const accelerationBuffer,
		const unsigned int particlesPerChain,
		const FP reducedSpringConstant,
		const FP timestep,
		const unsigned int stepCount);

	/**
	 * Computes the acceleration experienced by the given particle in the Gaussian Chain.
	 * No boundary conditions are considered.
	 * @param[in] positions             The array of MPC fluid particle positions.
	 * @param[in] firstParticleID       The ID of the first particle in the chain.
	 * @param[in] particleID            The ID of the particle to get the acceleration for.
	 * @param[in] lastParticleID        The ID of the last particle in the chain.
	 * @param[in] reducedSpringConstant The spring constant, divided by the mass of an
	 *                                  individual spring constituent particle.
	 * @param[in] timestep              The timestep for an individual velocity-Verlet step.
	 */
	__device__ const Vector3D<FP> getAccelerationGaussianChainVelocityVerlet(
		MPCParticlePositionType* const positions,
		const unsigned int firstParticleID,
		const unsigned int particleID,
		const unsigned int lastParticleID,
		const FP reducedSpringConstant,
		const FP timestep);

} //namespace DeviceCode
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
