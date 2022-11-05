/**
 * @file
 * Defines CUDA Device code for OpenMPCD::CUDA::Simulation
 */

#ifndef OPENMPCD_CUDA_DEVICECODE_SIMULATION_HPP
#define OPENMPCD_CUDA_DEVICECODE_SIMULATION_HPP

#include <OpenMPCD/CUDA/Types.hpp>
#include <OpenMPCD/Types.hpp>
#include <OpenMPCD/Vector3D.hpp>

namespace OpenMPCD
{
namespace CUDA
{
/**
 * Contains CUDA Device code.
 */
namespace DeviceCode
{

/**
 * Returns whether the given position lies within the primary simulation volume.
 *
 * The primary simulation volume is defined as
 * \f[
 * 	\left[0, L_x\right) \times \left[0, L_y\right) \times \left[0, L_z\right)
 * \f]
 * where \f$ L_x \f$ is the value of `mpcSimulationBoxSizeX`, i.e. the size of
 * the primary simulation volume along the \f$ x \f$ axis, and analogously for
 * \f$ L_y \f$ and `mpcSimulationBoxSizeY`,
 * and \f$ L_z \f$ and `mpcSimulationBoxSizeZ`.
 *
 * This function requires that
 * `OpenMPCD::CUDA::DeviceCode::setSimulationBoxSizeSymbols` has been called.
 *
 * @param[in] position The position to check.
 */
__device__ bool isInPrimarySimulationVolume(
	const Vector3D<MPCParticlePositionType>& position);

/**
 * Returns the collision cell index for the given position.
 *
 * The collision cell index is determined as follows:
 * The three cartesian coordinates of the given particle position are rounded
 * down towards the nearest integer. This triple of numbers then determine the
 * coordinates of the collision cell the particle lies in. If they are called
 * `cellX`, `cellY`, and `cellZ`, respectively, what is returned is
 * `cellX + cellY * boxSizeX` + cellZ * boxSizeX * boxSizeY`, where `boxSizeX`
 * and `boxSizeY` are the number of collision cells in the primary simulation
 * volume along the `x` and `y` directions, respectively.
 *
 * This function requires that
 * `OpenMPCD::CUDA::DeviceCode::setSimulationBoxSizeSymbols` has been called
 * before.
 *
 * @param[in] position The position to consider, which must lie in the primary
 *                     simulation volume, i.e. the `x`, `y`, and `z` coordinates
 *                     must be non-negative and smaller than the simulation box
 *                     size along the respective direction.
 */
__device__ unsigned int getCollisionCellIndex(
	const Vector3D<MPCParticlePositionType>& position);

/**
 * Sorts the MPC particles into the collision cells, temporarily applying
 * Lees-Edwards boundary conditions.
 *
 * This function requires that
 * `OpenMPCD::CUDA::DeviceCode::setSimulationBoxSizeSymbols` and
 * `OpenMPCD::CUDA::DeviceCode::setLeesEdwardsSymbols` have been called before.
 *
 * @see LeesEdwardsBoundaryConditions
 *
 * @param[in]     workUnitOffset       The number of particles to skip.
 * @param[in]     particleCount        The number of particles there are.
 * @param[in]     gridShift_           The grid shift vector to temporarily add
 *                                     to all positions.
 * @param[in]     mpcTime              The MPC time that has passed since the
 *                                     start of the simulation.
 * @param[in]     positions            The MPC fluid particle positions.
 * @param[in,out] velocities           The MPC fluid particle velocities, which
 *                                     may be changed due to Lees-Edwards
 *                                     boundary conditions.
 * @param[out]    velocityCorrections  The velocity corrections applied due to
 *                                     Lees-Edwards boundary conditions, which
 *                                     can be undone by
 *                                     `undoLeesEdwardsVelocityCorrections`; the
 *                                     buffer has to be at least `particleCount`
 *                                     elements long.
 * @param[out]    collisionCellIndices The collision cell indices for the
 *                                     particles.
 */
__global__ void sortParticlesIntoCollisionCellsLeesEdwards(
	const unsigned int workUnitOffset,
	const unsigned int particleCount,
	const MPCParticlePositionType* const gridShift_,
	const FP mpcTime,
	const MPCParticlePositionType* const positions,
	MPCParticleVelocityType* const velocities,
	MPCParticleVelocityType* const velocityCorrections,
	unsigned int* const collisionCellIndices);

/**
 * Sorts the given particle into the collision cells, temporarily applying
 * Lees-Edwards boundary conditions.
 *
 * This function requires that
 * `OpenMPCD::CUDA::DeviceCode::setSimulationBoxSizeSymbols` and
 * `OpenMPCD::CUDA::DeviceCode::setLeesEdwardsSymbols` have been called before.
 *
 * @see LeesEdwardsBoundaryConditions
 *
 * @param[in]     particleID           The particle ID.
 * @param[in]     gridShift_           The grid shift vector to temporarily add
 *                                     to the position.
 * @param[in]     mpcTime              The MPC time that has passed since the
 *                                     start of the simulation.
 * @param[in]     positions            The particle positions.
 * @param[in,out] velocities           The particle velocities, which may be
 *                                     changed due to Lees-Edwards boundary
 *                                     conditions.
 * @param[out]    velocityCorrections  The velocity corrections applied due to
 *                                     Lees-Edwards boundary conditions.
 * @param[out]    collisionCellIndices The collision cell indices for the
 *                                     particles.
 */
__device__ void sortIntoCollisionCellsLeesEdwards(
	const unsigned int particleID,
	const MPCParticlePositionType* const gridShift_,
	const FP mpcTime,
	const MPCParticlePositionType* const positions,
	MPCParticleVelocityType* const velocities,
	MPCParticleVelocityType* const velocityCorrections,
	unsigned int* const collisionCellIndices);

/**
 * Computes the collision cell mass and momentum contributions by the given
 * particles.
 *
 * @param[in]     workUnitOffset       The number of particles to skip.
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
__global__ void collisionCellContributions(
	const unsigned int workUnitOffset,
	const unsigned int particleCount,
	const MPCParticleVelocityType* const velocities,
	const unsigned int* const collisionCellIndices,
	MPCParticleVelocityType* const collisionCellMomenta,
	FP* const collisionCellMasses,
	const FP particleMass);

/**
 * Returns the center-of-mass velocity of a collision cell.
 *
 * @param[in] collisionCellIndex   The index of the collision cell.
 * @param[in] collisionCellMomenta The momenta of the collision cells.
 * @param[in] collisionCellMasses  The masses of the collision cells.
 */
__device__
Vector3D<MPCParticleVelocityType> getCollisionCellCenterOfMassVelocity(
	const unsigned int collisionCellIndex,
	const MPCParticleVelocityType* const collisionCellMomenta,
	const FP* const collisionCellMasses);


/**
 * Resets the collision cell data buffers.
 *
 * This kernel sets the data pointed to by the arguments provided to `0`.
 * One needs to spawn at least `collisionCellCount` threads (possibly using
 * `workUnitOffset`, if one grid does not fit all threads) for this operation to
 * complete semantically.
 *
 * Each pointer argument is required to be non-`nullptr`, and point to at least
 * `collisionCellCount` elements; `collisionCellMomenta` needs to point to at
 * least `3 * collisionCellCount` elements instead.
 *
 * @param[in]  workUnitOffset
 *             The number of collision cells to skip.
 * @param[in]  collisionCellCount
 *             The number of collision cells there are.
 * @param[out] collisionCellParticleCounts
 *             Pointer to the memory where the collision cell particle counts
 *             are stored.
 * @param[out] collisionCellMomenta
 *             Pointer to the memory where the collision cell momenta are
 *             stored (three per collision cell).
 * @param[out] collisionCellFrameInternalKineticEnergies
 *             Pointer to the memory where the collision cell kinetic energies
 *             (in the internal frame) are stored.
 * @param[out] collisionCellMasses
 *             Pointer to the memory where the collision cell masses are stored.
 */
__global__ void resetCollisionCellData(
	const unsigned int workUnitOffset,
	const unsigned int collisionCellCount,
	unsigned int* const collisionCellParticleCounts,
	MPCParticleVelocityType* const collisionCellMomenta,
	FP* const collisionCellFrameInternalKineticEnergies,
	FP* const collisionCellMasses);

/**
 * Generates a new random grid shift vector.
 *
 * This function will draw three Cartesian coordinates from the uniform
 * distribution over \f$ \left( 0, 1 \right] \f$, multiply the results by
 * `gridShiftScale`, and store them in the three elements pointed at by
 * `gridShift`.
 *
 * This kernel must be called with one block and one thread only.
 *
 * @param[out]    gridShift
 *                Where to store the three grid shift coordinates to.
 * @param[in]     gridShiftScale
 *                The scale for the grid shift vector coordinates.
 * @param[in,out] rngs
 *                Pointer to at least one random number generator.
 */
__global__ void generateGridShiftVector(
	MPCParticlePositionType* const gridShift,
	const FP gridShiftScale,
	GPURNG* const rngs);

/**
 * Generates new rotation axes for the collision cells.
 *
 * @param[in]     workUnitOffset
 *                The number of particles to skip.
 * @param[in]     collisionCellCount
 *                The number of collision cells there are.
 * @param[out]    collisionCellRotationAxes
 *                For each collision cell, stores the Cartesian coordinates of
 *                the unit-length axis of rotation.
 * @param[in,out] rngs
 *                The random number generators, of which there must be at least
 *                `collisionCellCount`.
 */
__global__ void generateCollisionCellRotationAxes(
	const unsigned int workUnitOffset,
	const unsigned int collisionCellCount,
	FP* const collisionCellRotationAxes,
	GPURNG* const rngs);

/**
 * Applies the first step of the SRD rotation to the given particles.
 *
 * This function requires that
 * `OpenMPCD::CUDA::DeviceCode::setSRDCollisionAngleSymbol` has been called
 * before.
 *
 * @param[in]     workUnitOffset
 *                The number of particles to skip.
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
 *                axis it is to use during the collision.
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
__global__ void collisionCellStochasticRotationStep1(
	const unsigned int workUnitOffset,
	const unsigned int particleCount,
	MPCParticleVelocityType* const velocities,
	const FP particleMass,
	const unsigned int* const collisionCellIndices,
	const MPCParticleVelocityType* const collisionCellMomenta,
	const FP* const collisionCellMasses,
	const MPCParticlePositionType* const collisionCellRotationAxes,
	FP* const collisionCellFrameInternalKineticEnergies,
	unsigned int* const collisionCellParticleCounts);

/**
 * Generates Maxwell-Boltzmann-Scaling factors for the collision cells.
 *
 * @param[in]     workUnitOffset
 *                The number of particles to skip.
 * @param[in]     collisionCellCount
 *                The number of collision cells there are.
 * @param[in]     collisionCellFrameInternalKineticEnergies
 *                Stores, for each collision cell, the sum of the kinetic
 *                energies of the particles in that collision cell, as measured
 *                in that collision cell's center-of-mass frame
 * @param[in]     collisionCellParticleCounts
 *                The number of particles in the collision cells.
 * @param[out]    collisionCellRelativeVelocityScalings
 *                For each collision cell, stores the factor by which to scale
 *                the relative velocities.
 * @param[in]     bulkThermostatTargetkT
 *                The target temperature for the thermostat.
 * @param[in,out] rngs
 *                The random number generators, of which there must be at least
 *                `collisionCellCount`.
 */
__global__ void generateCollisionCellMBSFactors(
	const unsigned int workUnitOffset,
	const unsigned int collisionCellCount,
	const FP* const collisionCellFrameInternalKineticEnergies,
	unsigned int* const collisionCellParticleCounts,
	FP* const collisionCellRelativeVelocityScalings,
	const FP bulkThermostatTargetkT,
	GPURNG* const rngs);

/**
 * Applies the second step of the SRD rotation to the given particles.
 *
 * This kernel scales each particle's velocity with the corresponding collision
 * cell's scaling factor (stored in `collisionCellVelocityScalings`), and then
 * adds the corresponding collision cell's center-of-mass velocity.
 *
 * @param[in]     workUnitOffset
 *                The number of particles to skip.
 * @param[in]     particleCount
 *                The number of particles there are.
 * @param[in,out] velocities
 *                The new particle velocities as output; as input, the
 *                values in this array are to contain the rotated velocities in
 *                the collision cell's center-of-mass frame.
 * @param[in]     collisionCellIndices
 *                An array which holds, for each particle, the index of the
 *                collision cell it is currently in.
 * @param[in]     collisionCellMomenta
 *                An array holding, for each collision cell, its three Cartesian
 *                momentum coordinates.
 * @param[in]     collisionCellMasses
 *                An array holding, for each collision cell, its total mass
 *                content.
 * @param[in]     collisionCellVelocityScalings
 *                For each collision cell, stores the factor by which to scale
 *                the relative velocities.
 */
__global__ void collisionCellStochasticRotationStep2(
	const unsigned int workUnitOffset,
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
 * @param[in]     workUnitOffset
 *                The number of particles to skip.
 * @param[in]     particleCount
 *                The number of particles there are.
 * @param[in,out] velocities
 *                The particle velocities.
 * @param[in]     velocityCorrections
 *                The velocity corrections as returned by
 *                `sortIntoCollisionCellsLeesEdwards`.
 */
__global__ void undoLeesEdwardsVelocityCorrections(
	const unsigned int workUnitOffset,
	const unsigned int particleCount,
	MPCParticleVelocityType* const velocities,
	const MPCParticleVelocityType* const velocityCorrections);

/**
 * Sets up instances of `GPURNG` in the specified memory location.
 *
 * The individual RNGs will have subsequence numbers (or, alternatively, seeds)
 * set such that the individual instances generate independent streams of random
 * numbers.
 *
 * Only the first thread in this kernel does any work.
 *
 * @param[in]  count
 *             The number of instances to construct.
 * @param[out] location
 *             The location in memory to store the instances in; memory must
 *             have been allocated at this location prior to calling this
 *             function.
 * @param[in]  seed
 *             The seed to use for the RNGs.
 */
__global__
void constructGPURNGs(
	const std::size_t count,
	GPURNG* const location,
	const unsigned long long seed);

/**
 * Destroys instances of `GPURNG` in the specified memory location.
 *
 * Only the first thread in this kernel does any work.
 *
 * @param[in]  count
 *             The number of instances to construct.
 * @param[out] location
 *             The location in memory where there are instances; memory must
 *             be freed manually after this function call.
 */
__global__
void destroyGPURNGs(const std::size_t count, GPURNG* const location);

} //namespace DeviceCode
} //namespace CUDA
} //namespace OpenMPCD

#endif
