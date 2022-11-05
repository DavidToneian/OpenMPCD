#include <OpenMPCD/ImplementationDetails/Simulation.hpp>

#include <OpenMPCD/LeesEdwardsBoundaryConditions.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>
#include <OpenMPCD/RemotelyStoredVector.hpp>

namespace OpenMPCD
{
namespace ImplementationDetails
{

unsigned int getCollisionCellIndex(
	const Vector3D<MPCParticlePositionType>& position,
	const unsigned int boxSizeX,
	const unsigned int boxSizeY,
	const unsigned int boxSizeZ)
{
	const FP x = position.getX();
	const FP y = position.getY();
	const FP z = position.getZ();

	const unsigned int cellX = static_cast<unsigned int>(x);
	const unsigned int cellY = static_cast<unsigned int>(y);
	const unsigned int cellZ = static_cast<unsigned int>(z);

	const unsigned int index =
		cellZ * boxSizeX * boxSizeY +
		cellY * boxSizeX +
		cellX;

	OPENMPCD_DEBUG_ASSERT(index < boxSizeX * boxSizeY * boxSizeZ);

	return index;
}

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
	const unsigned int boxSizeZ)
{
	const RemotelyStoredVector<const MPCParticlePositionType> gridShift(
		gridShift_);

	const RemotelyStoredVector<const MPCParticlePositionType> position(
		positions, particleID);

	OPENMPCD_DEBUG_ASSERT(gridShift.isFinite());
	OPENMPCD_DEBUG_ASSERT(position.isFinite());

	MPCParticleVelocityType velocityCorrection;
	const Vector3D<MPCParticlePositionType> image =
		getImageUnderLeesEdwardsBoundaryConditions(
			position + gridShift,
			mpcTime,
			shearRate,
			boxSizeX, boxSizeY, boxSizeZ,
			&velocityCorrection);

	velocities[3 * particleID + 0] += velocityCorrection;
	velocityCorrections[particleID] = velocityCorrection;

	collisionCellIndices[particleID] =
		getCollisionCellIndex(image, boxSizeX, boxSizeY, boxSizeZ);
}

void collisionCellContributions(
	const unsigned int particleCount,
	const MPCParticleVelocityType* const velocities,
	const unsigned int* const collisionCellIndices,
	MPCParticleVelocityType* const collisionCellMomenta,
	FP* const collisionCellMasses,
	const FP particleMass)
{
	for(unsigned int particleID = 0; particleID < particleCount; ++particleID)
	{
		const unsigned int collisionCellIndex = collisionCellIndices[particleID];

		const RemotelyStoredVector<const MPCParticleVelocityType>
			particleVelocity(velocities, particleID);
		RemotelyStoredVector<MPCParticleVelocityType>
			cellMomentum(collisionCellMomenta, collisionCellIndex);

		OPENMPCD_DEBUG_ASSERT(particleVelocity.isFinite());
		OPENMPCD_DEBUG_ASSERT(cellMomentum.isFinite());

		cellMomentum += particleVelocity * particleMass;
		collisionCellMasses[collisionCellIndex] += particleMass;

		OPENMPCD_DEBUG_ASSERT(cellMomentum.isFinite());
	}
}

Vector3D<MPCParticleVelocityType>
getCollisionCellCenterOfMassMomentum(
	const unsigned int collisionCellIndex,
	const MPCParticleVelocityType* const collisionCellMomenta)
{
	const RemotelyStoredVector<const MPCParticleVelocityType>
		collisionCellMomentum(collisionCellMomenta, collisionCellIndex);

	return collisionCellMomentum;
}

Vector3D<MPCParticleVelocityType>
getCollisionCellCenterOfMassVelocity(
	const unsigned int collisionCellIndex,
	const MPCParticleVelocityType* const collisionCellMomenta,
	const FP* const collisionCellMasses)
{
	const RemotelyStoredVector<const MPCParticleVelocityType>
		collisionCellMomentum(collisionCellMomenta, collisionCellIndex);

	const FP collisionCellMass = collisionCellMasses[collisionCellIndex];

	const Vector3D<MPCParticleVelocityType> ret =
		collisionCellMomentum / collisionCellMass;

	OPENMPCD_DEBUG_ASSERT(ret.isFinite());

	return ret;
}

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
	unsigned int* const collisionCellParticleCounts)
{
	for(unsigned int particleID = 0; particleID < particleCount; ++particleID)
	{
		const unsigned int collisionCellIndex =
			collisionCellIndices[particleID];

		RemotelyStoredVector<MPCParticleVelocityType>
			particleVelocity(velocities, particleID);

		const Vector3D<MPCParticleVelocityType> collisionCellVelocity =
			getCollisionCellCenterOfMassVelocity(
				collisionCellIndex, collisionCellMomenta, collisionCellMasses);

		Vector3D<MPCParticleVelocityType> relativeVelocity =
			particleVelocity - collisionCellVelocity;

		const RemotelyStoredVector<const MPCParticlePositionType>
			rotationAxis(collisionCellRotationAxes, collisionCellIndex);

		particleVelocity =
			relativeVelocity.getRotatedAroundNormalizedAxis(
				rotationAxis, collisionAngle);

		//particleVelocity now contains the velocity *relative* to the
		//center-of-mass frame of the collision cell!

		OPENMPCD_DEBUG_ASSERT(particleVelocity.isFinite());

		collisionCellFrameInternalKineticEnergies[collisionCellIndex] +=
			particleMass * relativeVelocity.getMagnitudeSquared() / 2.0;
		collisionCellParticleCounts[collisionCellIndex] += 1;
	}
}

void collisionCellStochasticRotationStep2(
	const unsigned int particleCount,
	MPCParticleVelocityType* const velocities,
	const unsigned int* const collisionCellIndices,
	const MPCParticleVelocityType* const collisionCellMomenta,
	const FP* const collisionCellMasses,
	const FP* const collisionCellVelocityScalings)
{
	for(unsigned int particleID = 0; particleID < particleCount; ++particleID)
	{
		const unsigned int collisionCellIndex = collisionCellIndices[particleID];

		RemotelyStoredVector<MPCParticleVelocityType>
			particleVelocity(velocities, particleID);

		OPENMPCD_DEBUG_ASSERT(particleVelocity.isFinite());
		OPENMPCD_DEBUG_ASSERT(
			std::isfinite(collisionCellVelocityScalings[collisionCellIndex]));

		particleVelocity *=
			collisionCellVelocityScalings[collisionCellIndex];

		particleVelocity +=
			getCollisionCellCenterOfMassVelocity(
				collisionCellIndex, collisionCellMomenta, collisionCellMasses);

		OPENMPCD_DEBUG_ASSERT(particleVelocity.isFinite());
	}
}

void undoLeesEdwardsVelocityCorrections(
	const unsigned int particleCount,
	MPCParticleVelocityType* const velocities,
	const MPCParticleVelocityType* const velocityCorrections)
{
	for(unsigned int particleID = 0; particleID < particleCount; ++particleID)
	{
		OPENMPCD_DEBUG_ASSERT(std::isfinite(velocities[3 * particleID + 0]));
		OPENMPCD_DEBUG_ASSERT(std::isfinite(velocityCorrections[particleID]));

		velocities[3 * particleID + 0] -= velocityCorrections[particleID];

		OPENMPCD_DEBUG_ASSERT(std::isfinite(velocities[3 * particleID + 0]));
	}
}

} //namespace ImplementationDetails
} //namespace OpenMPCD
