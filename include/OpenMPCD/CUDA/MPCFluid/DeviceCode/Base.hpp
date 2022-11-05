/**
 * @file
 * Defines CUDA Device code for OpenMPCD::CUDA::MPCFluid::Base
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_DEVICECODE_BASE_HPP
#define OPENMPCD_CUDA_MPCFLUID_DEVICECODE_BASE_HPP

#include <OpenMPCD/Types.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{
/**
 * Contains CUDA Device code.
 */
namespace DeviceCode
{
	/**
	 * Sets the symbol mpcParticleCount.
	 * @param[in] count The number of MPC fluid particles.
	 */
	void setMPCParticleCountSymbol(const unsigned int count);

	/**
	 * Computes the center of mass of logical entities, where each logical
	 * entity has a common number of constituent particles, each having the same
	 * mass.
	 *
	 * Each thread of this kernel computes one center of mass.
	 *
	 * @param[in]  workUnitOffset
	 *             The number of entities to skip; this is useful in case all
	 *             entities cannot fit in one single kernel call.
	 * @param[in]  positions
	 *             The positions of the individual constituent particles, stored
	 *             on the CUDA Device.
	 *             It is assumed that first, the `x` coordinate of the first
	 *             constituent particle of the first logical entity is stored,
	 *             then its `y` and `z` coordinates; after that, the coordinates
	 *             of the second constituent of the first logical entity,
	 *             and so forth; after the last constituent of the first logical
	 *             entity, the constituents of the second logical entity follow.
	 *             In total, `positions` must hold at least
	 *             `3 * particlesPerLogicalEntity * numberOfLogicalEntities`
	 *             elements.
	 * @param[in]  numberOfLogicalEntities
	 *             The number of logical entities to treat.
	 * @param[in]  particlesPerLogicalEntity
	 *             The number of MPC particles per logical entity, which must
	 *             not be `0`.
	 * @param[out] output
	 *             The CUDA Device buffer to save the coordinates to. It must be
	 *             able to hold at least `3 * numberOfLogicalEntities` elements.
	 *             The first element in the buffer will be the `x`
	 *             coordinate of the center of mass of the first logical entity,
	 *             followed by the `y` and `z` coordinates. After that, the
	 *             second entity's coordinates follow, and so on.
	 */
	__global__ void computeLogicalEntityCentersOfMass(
		const unsigned int workUnitOffset,
		const MPCParticlePositionType* const positions,
		const unsigned int numberOfLogicalEntities,
		const unsigned int particlesPerLogicalEntity,
		MPCParticlePositionType* const output);
} //namespace DeviceCode
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
