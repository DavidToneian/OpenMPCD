/**
 * @file
 * Defines CUDA Device code for MPC fluids consisting out of MPC-particle chains.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_DEVICECODE_CHAINS_HPP
#define OPENMPCD_CUDA_MPCFLUID_DEVICECODE_CHAINS_HPP

#include <OpenMPCD/Types.hpp>
#include <OpenMPCD/Vector3D.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{
namespace DeviceCode
{
	/**
	 * Saves the center-of-mass velocities of the MPC fluid's chains to the given buffer.
	 *
	 * This function assumes that all chains have the same number of particles,
	 * and that all particles have the same mass.
	 *
	 * This function requires that the
	 * `OpenMPCD::CUDA::MPCFluid::DeviceCode::mpcParticleCount`
	 * symbol has been set, and its value is equal to the `mpcParticleCount`
	 * passed to this function.
	 *
	 * This function assumes that `mpcParticleCount` is an integer multiple of
	 * `chainLength`, and assumes both of them to be non-zero.
	 *
	 * @param[in]  mpcParticleCount The number of individual MPC fluid particles.
	 * @param[in]  chainLength      The number of individual MPC fluid particles per chain.
	 * @param[in]  velocities       The Device array of MPC fluid particle velocities.
	 * @param[out] comVelocities    Device buffer where the center-of-mass velocities are saved.
	 */
	void getCenterOfMassVelocities_chain(
		const unsigned int mpcParticleCount,
		const unsigned int chainLength,
		const MPCParticleVelocityType* const velocities,
		MPCParticleVelocityType* const comVelocities);

	/**
	 * Saves the center-of-mass velocities of the MPC fluid's chains to the given buffer.
	 *
	 * This function assumes that all chains have the same number of particles,
	 * and that all particles have the same mass.
	 *
	 * This function requires that the
	 * `OpenMPCD::CUDA::MPCFluid::DeviceCode::mpcParticleCount`
	 * symbol has been set to a value that is non-zero and an integer multiple
	 * of `chainLength`.
	 *
	 * @param[in]  workUnitOffset The number of chains to skip.
	 * @param[in]  chainLength    The number of individual MPC fluid particles per chain.
	 * @param[in]  velocities     The array of MPC fluid particle velocities.
	 * @param[out] comVelocities  Buffer where the center-of-mass velocities are saved.
	 */
	__global__ void getCenterOfMassVelocities_chain_kernel(
		const unsigned int workUnitOffset,
		const unsigned int chainLength,
		const MPCParticleVelocityType* const velocities,
		MPCParticleVelocityType* const comVelocities);

	/**
	 * Returns the center-of-mass velocity of for the given chain.
	 *
	 * This function assumes that all chains have the same number of particles,
	 * and that all particles have the same mass.
	 *
	 * @param[in] chainID     The ID of the chain.
	 * @param[in] chainLength The number of individual MPC fluid particles per chain.
	 * @param[in] velocities  The array of MPC fluid particle velocities.
	 */
	__device__ Vector3D<MPCParticleVelocityType> getCenterOfMassVelocity_chain(
		const unsigned int chainID,
		const unsigned int chainLength,
		const MPCParticleVelocityType* const velocities);
} //namespace DeviceCode
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
