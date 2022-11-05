/**
 * @file
 * Declares various functions that are needed in MPCD.
 */

#ifndef OPENMPCD_IMPLEMENTATIONDETAILS_SIMULATION_HPP
#define OPENMPCD_IMPLEMENTATIONDETAILS_SIMULATION_HPP

#include <OpenMPCD/Types.hpp>
#include <OpenMPCD/Vector3D.hpp>

namespace OpenMPCD
{
namespace ImplementationDetails
{

/**
 * Returns the collision cell index for the given position.
 *
 * The collision cell index is determined as follows:
 * The three cartesian coordinates of the given particle position are rounded
 * down towards the nearest integer. This triple of numbers then determine the
 * coordinates of the collision cell the particle lies in. If they are called
 * `cellX`, `cellY`, and `cellZ`, respectively, what is returned is
 * `cellX + cellY * boxSizeX` + cellZ * boxSizeX * boxSizeY`.
 *
 * @param[in] position The position to consider, which must lie in the primary
 *                     simulation volume, i.e. the `x`, `y`, and `z` coordinates
 *                     must be non-negative and smaller than the simulation box
 *                     size along the respective direction.
 * @param[in] boxSizeX The number of collision cells in the primary simulation
 *                     volume along the `x` axis.
 * @param[in] boxSizeY The number of collision cells in the primary simulation
 *                     volume along the `y` axis.
 * @param[in] boxSizeZ The number of collision cells in the primary simulation
 *                     volume along the `z` axis.
 */
unsigned int getCollisionCellIndex(
	const Vector3D<MPCParticlePositionType>& position,
	const unsigned int boxSizeX,
	const unsigned int boxSizeY,
	const unsigned int boxSizeZ);

/**
 * Sorts the given particle into the collision cells, temporarily applying
 * Lees-Edwards boundary conditions.
 *
 * @see LeesEdwardsBoundaryConditions
 *
 * @param[in]     particleID
 *                The particle ID.
 * @param[in]     gridShift_
 *                The grid shift vector to temporarily add to the position.
 * @param[in]     shearRate
 *                The Lees-Edwards shear rate \f$ \dot{\gamma} \f$.
 * @param[in]     mpcTime
 *                The MPC time that has passed since the start of the
 *                simulation.
 * @param[in]     positions
 *                The particle positions.
 * @param[in,out] velocities
 *                The particle velocities, which may be changed due to
 *                Lees-Edwards boundary conditions.
 * @param[out]    velocityCorrections
 *                An array, long enough to store one element per particle; at
 *                position `particleID`, the velocity corrections along the `x`
 *                axis applied due to Lees-Edwards boundary conditions will be
 *                saved.
 * @param[out]    collisionCellIndices
 *                The collision cell indices for the particles.
 * @param[in]     boxSizeX
 *                The number of collision cells in the primary simulation volume
 *                along the `x` axis.
 * @param[in]     boxSizeY
 *                The number of collision cells in the primary simulation volume
 *                along the `y` axis.
 * @param[in]     boxSizeZ
 *                The number of collision cells in the primary simulation volume
 *                along the `z` axis.
 */
void sortIntoCollisionCellsLeesEdwards(
	const unsigned int particleID,
	const MPCParticlePositionType* const gridShift_,
	const FP shearRate,
	const FP mpcTime,
	const MPCParticlePositionType* const positions,
	MPCParticleVelocityType* const velocities,
	MPCParticleVelocityType* const velocityCorrections,
	unsigned int* const collisionCellIndices,
	const unsigned int boxSizeX,
	const unsigned int boxSizeY,
	const unsigned int boxSizeZ);

/**
 * Computes the collision cell mass and momentum contributions by the given
 * particles.
 *
 * @param[in]     particleCount        The number of particles there are.
 * @param[in]     velocities           The particle velocities.
 * @param[in]     collisionCellIndices The collision cell indices for the
 *                                     particles.
 * @param[in,out] collisionCellMomenta The collision cell momenta. This array is
 *                                     expected to be long enough to store `3`
 *                                     coordinates for each collision cell, and
 *                                     is assumed to have each element set to
 *                                     `0` prior to calling this function.
 * @param[in,out] collisionCellMasses  The masses of the collision cells. This
 *                                     array is expected to be long enough to
 *                                     store `1` element for each collision
 *                                     cell, and is assumed to have each element
 *                                     set to `0` prior to calling this
 *                                     function.
 * @param[in]     particleMass         The mass of any one particle.
 */
void collisionCellContributions(
	const unsigned int particleCount,
	const MPCParticleVelocityType* const velocities,
	const unsigned int* const collisionCellIndices,
	MPCParticleVelocityType* const collisionCellMomenta,
	FP* const collisionCellMasses,
	const FP particleMass);

/**
 * Returns the center-of-mass momentum of a collision cell.
 *
 * @param[in] collisionCellIndex   The index of the collision cell.
 * @param[in] collisionCellMomenta The momenta of the collision cells.
 */
Vector3D<MPCParticleVelocityType> getCollisionCellCenterOfMassMomentum(
	const unsigned int collisionCellIndex,
	const MPCParticleVelocityType* const collisionCellMomenta);

/**
 * Returns the center-of-mass velocity of a collision cell.
 *
 * @param[in] collisionCellIndex   The index of the collision cell.
 * @param[in] collisionCellMomenta The momenta of the collision cells.
 * @param[in] collisionCellMasses  The masses of the collision cells.
 */
Vector3D<MPCParticleVelocityType> getCollisionCellCenterOfMassVelocity(
	const unsigned int collisionCellIndex,
	const MPCParticleVelocityType* const collisionCellMomenta,
	const FP* const collisionCellMasses);

/**
 * Applies the first step of the SRD rotation to the given particles.
 *
 * @param[in]     particleCount
 *                The number of particles there are.
 * @param[in,out] velocities
 *                The particle velocities as input; after the function call, the
 *                values in this array will contain the velocities in the
 *                collision cell's center-of-mass frame.
 * @param[in]     particleMass
 *                The mass of any one particle.
 * @param[in]     collisionCellIndices
 *                An array which holds, for each particle, the index of the
 *                collision cell it is currently in.
 * @param[in]     collisionCellMomenta
 *                An array holding, for each collision cell, its three cartesian
 *                momentum coordinates.
 * @param[in]     collisionCellMasses
 *                An array holding, for each collision cell, its total mass
 *                content.
 * @param[in]     collisionCellRotationAxes
 *                An array holding, for each collision cell, the random rotation
 *                axis it is to use during the collision. Each axis consists of
 *                the `x`, `y`, and `z` coordinate of a normalized vector.
 * @param[in]     collisionAngle
 *                The rotation angle to use for the SRD collision.
 * @param[out]    collisionCellFrameInternalKineticEnergies
 *                An array, long enough to hold an element for each collision
 *                cell. The respective elements are set to the sum of the
 *                kinetic energies of the particles in that collision
 *                cell, as measured in that collision cell's center-of-mass
 *                frame. The array is not set to `0` by this function prior to
 *                execution of the algorithm.
 * @param[out]    collisionCellParticleCounts
 *                An array, long enough to hold an element for each collision
 *                cell. The respective elements are set to the number of
 *                particles in that collision cell. The array is assumed to
 *                contain only values `0` prior to calling this function.
 */
void collisionCellStochasticRotationStep1(
	const unsigned int particleCount,
	MPCParticleVelocityType* const velocities,
	const FP particleMass,
	const unsigned int* const collisionCellIndices,
	const MPCParticleVelocityType* const collisionCellMomenta,
	const FP* const collisionCellMasses,
	const MPCParticlePositionType* const collisionCellRotationAxes,
	const FP collisionAngle,
	FP* const collisionCellFrameInternalKineticEnergies,
	unsigned int* const collisionCellParticleCounts);

/**
 * Applies the first step of the SRD rotation to the given particles.
 *
 * @param[in]     particleCount
 *                The number of particles there are.
 * @param[in,out] velocities
 *                The particle new velocities as output; as input, the
 *                values in this array are to contain the rotated velocities in
 *                the collision cell's center-of-mass frame.
 * @param[in]     collisionCellIndices
 *                An array which holds, for each particle, the index of the
 *                collision cell it is currently in.
 * @param[in]     collisionCellMomenta
 *                An array holding, for each collision cell, its three cartesian
 *                momentum coordinates.
 * @param[in]     collisionCellMasses
 *                An array holding, for each collision cell, its total mass
 *                content.
 * @param[in]     collisionCellVelocityScalings
 *                For each collision cell, stores the factor by which to scale
 *                the relative velocities.
 */
void collisionCellStochasticRotationStep2(
	const unsigned int particleCount,
	MPCParticleVelocityType* const velocities,
	const unsigned int* const collisionCellIndices,
	const MPCParticleVelocityType* const collisionCellMomenta,
	const FP* const collisionCellMasses,
	const FP* const collisionCellVelocityScalings);

/**
 * Undoes the velocity corrections applied by
 * `sortIntoCollisionCellsLeesEdwards`.
 *
 * @param[in]     particleCount
 *                The number of particles there are.
 * @param[in,out] velocities
 *                The particle velocities.
 * @param[in]     velocityCorrections
 *                The velocity corrections as returned by
 *                `sortIntoCollisionCellsLeesEdwards`.
 */
void undoLeesEdwardsVelocityCorrections(
	const unsigned int particleCount,
	MPCParticleVelocityType* const velocities,
	const MPCParticleVelocityType* const velocityCorrections);

} //namespace ImplementationDetails
} //namespace OpenMPCD

#endif //OPENMPCD_IMPLEMENTATIONDETAILS_SIMULATION_HPP
