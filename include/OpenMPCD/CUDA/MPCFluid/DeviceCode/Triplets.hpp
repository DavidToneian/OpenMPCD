/**
 * @file
 * Defines CUDA Device code for MPC fluids consisting out of MPC-particle triplets.
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_DEVICECODE_TRIPLETS_HPP
#define OPENMPCD_CUDA_MPCFLUID_DEVICECODE_TRIPLETS_HPP

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
	 * Saves the center-of-mass velocities of the MPC fluid's triplets to the given buffer.
	 * @param[in]  mpcParticleCount The number of individual MPC fluid particles.
	 * @param[in]  velocities       The Device array of MPC fluid particle velocities.
	 * @param[out] comVelocities    Device buffer where the center-of-mass velocities are saved.
	 */
	void getCenterOfMassVelocities_triplet(
		const unsigned int mpcParticleCount,
		const MPCParticleVelocityType* const velocities,
		MPCParticleVelocityType* const comVelocities);

	/**
	 * Saves the center-of-mass velocities of the MPC fluid's triplets to the given buffer.
	 * @param[in]  workUnitOffset The number of trimers to skip.
	 * @param[in]  velocities     The array of MPC fluid particle velocities.
	 * @param[out] comVelocities  Buffer where the center-of-mass velocities are saved.
	 */
	__global__ void getCenterOfMassVelocities_triplet_kernel(
		const unsigned int workUnitOffset,
		const MPCParticleVelocityType* const velocities,
		MPCParticleVelocityType* const comVelocities);

	/**
	 * Returns the center-of-mass velocity of for the given triplet.
	 * This function assumes that all three constituents have the same mass.
	 * @param[in] tripletID  The ID of the triplet.
	 * @param[in] velocities The array of MPC fluid particle velocities.
	 */
	__device__ Vector3D<MPCParticleVelocityType> getCenterOfMassVelocity_triplet(
		const unsigned int tripletID,
		const MPCParticleVelocityType* const velocities);
} //namespace DeviceCode
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
