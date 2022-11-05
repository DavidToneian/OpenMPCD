#include <OpenMPCD/CUDA/DeviceCode/Simulation.hpp>

#include <OpenMPCD/CUDA/DeviceCode/LeesEdwardsBoundaryConditions.hpp>
#include <OpenMPCD/CUDA/DeviceCode/Symbols.hpp>
#include <OpenMPCD/CUDA/DeviceCode/Utilities.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Symbols.hpp>
#include <OpenMPCD/CUDA/Random/Distributions/Gamma_shape_ge_1.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>
#include <OpenMPCD/RemotelyStoredVector.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace DeviceCode
{

__device__ bool isInPrimarySimulationVolume(
	const Vector3D<MPCParticlePositionType>& position)
{
	if(position.getX() < 0 || position.getX() >= mpcSimulationBoxSizeX)
		return false;

	if(position.getY() < 0 || position.getY() >= mpcSimulationBoxSizeY)
		return false;

	if(position.getZ() < 0 || position.getZ() >= mpcSimulationBoxSizeZ)
		return false;

	return true;
}

__device__ unsigned int getCollisionCellIndex(
	const Vector3D<MPCParticlePositionType>& position)
{
	const FP x = position.getX();
	const FP y = position.getY();
	const FP z = position.getZ();

	const unsigned int cellX = static_cast<unsigned int>(x);
	const unsigned int cellY = static_cast<unsigned int>(y);
	const unsigned int cellZ = static_cast<unsigned int>(z);

	const unsigned int index =
		cellZ * mpcSimulationBoxSizeX * mpcSimulationBoxSizeY +
		cellY * mpcSimulationBoxSizeX +
		cellX;

	OPENMPCD_DEBUG_ASSERT(index < collisionCellCount);

	return index;
}

__global__ void sortParticlesIntoCollisionCellsLeesEdwards(
	const unsigned int workUnitOffset,
	const unsigned int particleCount,
	const MPCParticlePositionType* const gridShift_,
	const FP mpcTime,
	const MPCParticlePositionType* const positions,
	MPCParticleVelocityType* const velocities,
	MPCParticleVelocityType* const velocityCorrections,
	unsigned int* const collisionCellIndices)
{
	const unsigned int particleID = blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;

	if(particleID >= particleCount)
		return;

	sortIntoCollisionCellsLeesEdwards(
		particleID, gridShift_, mpcTime, positions, velocities,
		velocityCorrections, collisionCellIndices);
}

__device__ void sortIntoCollisionCellsLeesEdwards(
	const unsigned int particleID,
	const MPCParticlePositionType* const gridShift_,
	const FP mpcTime,
	const MPCParticlePositionType* const positions,
	MPCParticleVelocityType* const velocities,
	MPCParticleVelocityType* const velocityCorrections,
	unsigned int* const collisionCellIndices)
{
	const RemotelyStoredVector<const MPCParticlePositionType> gridShift(gridShift_);

	const RemotelyStoredVector<const MPCParticlePositionType> position(positions, particleID);

	#ifdef OPENMPCD_CUDA_DEBUG
		if(!gridShift.isFinite())
		{
			printf(
				"%s %g %g %g (Particle ID: %u)\n",
				"Bad gridShift given in OpenMPCD::CUDA::DeviceCode::sortIntoCollisionCellsLeesEdwards:",
				gridShift.getX(), gridShift.getY(), gridShift.getZ(), particleID);
		}

		if(!position.isFinite())
		{
			printf(
				"%s %g %g %g (Particle ID: %u)\n",
				"Bad position given in OpenMPCD::CUDA::DeviceCode::sortIntoCollisionCellsLeesEdwards:",
				position.getX(), position.getY(), position.getZ(), particleID);
		}
	#endif

	MPCParticleVelocityType tmp;
	const Vector3D<MPCParticlePositionType> image =
		getImageUnderLeesEdwardsBoundaryConditions(
			mpcTime, position+gridShift, tmp);

	velocities[3 * particleID + 0] += tmp;
	velocityCorrections[particleID] = tmp;

	collisionCellIndices[particleID] = getCollisionCellIndex(image);
}

__global__ void collisionCellContributions(
	const unsigned int workUnitOffset,
	const unsigned int particleCount,
	const MPCParticleVelocityType* const velocities,
	const unsigned int* const collisionCellIndices,
	MPCParticleVelocityType* const collisionCellMomenta,
	FP* const collisionCellMasses,
	const FP particleMass)
{
	const unsigned int particleID = blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;
	if(particleID >= particleCount)
		return;

	const unsigned int collisionCellIndex = collisionCellIndices[particleID];

	const RemotelyStoredVector<const MPCParticleVelocityType> particleVelocity(velocities, particleID);
	RemotelyStoredVector<MPCParticleVelocityType> cellMomentum(collisionCellMomenta, collisionCellIndex);

	#ifdef OPENMPCD_CUDA_DEBUG
		if(!particleVelocity.isFinite())
			printf(	"particleVelocity in "
					"OpenMPCD::CUDA::DeviceCode::collisionCellFluidContributions"
					" is not finite. "
					"Values: %g %g %g\n",
					particleVelocity.getX(), particleVelocity.getY(), particleVelocity.getZ());

		if(!cellMomentum.isFinite())
			printf(	"cellMomentum before increment in "
					"OpenMPCD::CUDA::DeviceCode::collisionCellFluidContributions"
					" is not finite. "
					"Values: %g %g %g\n",
					cellMomentum.getX(), cellMomentum.getY(), cellMomentum.getZ());
	#endif

	cellMomentum.atomicAdd(particleVelocity * particleMass);
	atomicAdd(&collisionCellMasses[collisionCellIndex], particleMass);

	#ifdef OPENMPCD_CUDA_DEBUG
		if(!cellMomentum.isFinite())
			printf(	"cellMomentum after increment in "
					"OpenMPCD::CUDA::DeviceCode::collisionCellFluidContributions"
					" is not finite. "
					"Values: %g %g %g\n",
					cellMomentum.getX(), cellMomentum.getY(), cellMomentum.getZ());
	#endif
}

__device__ Vector3D<MPCParticleVelocityType>
getCollisionCellCenterOfMassVelocity(
	const unsigned int collisionCellIndex,
	const MPCParticleVelocityType* const collisionCellMomenta,
	const FP* const collisionCellMasses)
{
	const RemotelyStoredVector<const MPCParticleVelocityType>
		collisionCellMomentum(collisionCellMomenta, collisionCellIndex);

	const FP collisionCellMass = collisionCellMasses[collisionCellIndex];

	const Vector3D<MPCParticleVelocityType> ret = collisionCellMomentum / collisionCellMass;

	#ifdef OPENMPCD_CUDA_DEBUG
		if(!ret.isFinite())
			printf(	"ret at the end of "
					"OpenMPCD::CUDA::DeviceCode::getCollisionCellCenterOfMassVelocity"
					" is not finite. "
					"Values: %g %g %g\n"
					"collisionCellMomentum: %g %g %g\n"
					"collisionCellMass: %g\n",
					ret.getX(), ret.getY(), ret.getZ(),
					collisionCellMomentum.getX(), collisionCellMomentum.getY(), collisionCellMomentum.getZ(),
					collisionCellMass);
	#endif

	return ret;
}

__global__ void generateGridShiftVector(
	MPCParticlePositionType* const gridShift,
	const FP gridShiftScale,
	GPURNG* const rngs)
{
	OPENMPCD_DEBUG_ASSERT(gridDim.x == 1);
	OPENMPCD_DEBUG_ASSERT(gridDim.y == 1);
	OPENMPCD_DEBUG_ASSERT(gridDim.z == 1);
	OPENMPCD_DEBUG_ASSERT(blockDim.x == 1);
	OPENMPCD_DEBUG_ASSERT(blockDim.y == 1);
	OPENMPCD_DEBUG_ASSERT(blockDim.z == 1);

	CUDA::Random::Distributions::Uniform0e1i<MPCParticlePositionType> shiftDist;

	gridShift[0] = shiftDist(rngs[0]) * gridShiftScale;
	gridShift[1] = shiftDist(rngs[0]) * gridShiftScale;
	gridShift[2] = shiftDist(rngs[0]) * gridShiftScale;
}

__global__ void generateCollisionCellRotationAxes(
	const unsigned int workUnitOffset,
	const unsigned int collisionCellCount,
	FP* const collisionCellRotationAxes,
	GPURNG* const rngs)
{
	const unsigned int collisionCellID =
		blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;
	if(collisionCellID >= collisionCellCount)
		return;

	RemotelyStoredVector<MPCParticlePositionType> axis(
		collisionCellRotationAxes, collisionCellID);

	axis =
		Vector3D<MPCParticlePositionType>::getRandomUnitVector(
			rngs[collisionCellID]);
}

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
	unsigned int* const collisionCellParticleCounts)
{
	const unsigned int particleID = blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;
	if(particleID >= particleCount)
		return;

	const unsigned int collisionCellIndex = collisionCellIndices[particleID];

	RemotelyStoredVector<MPCParticleVelocityType>
		particleVelocity(velocities, particleID);

	const Vector3D<MPCParticleVelocityType> collisionCellVelocity =
		getCollisionCellCenterOfMassVelocity(collisionCellIndex, collisionCellMomenta, collisionCellMasses);

	const Vector3D<MPCParticleVelocityType> relativeVelocity = particleVelocity - collisionCellVelocity;

	const RemotelyStoredVector<const MPCParticlePositionType>
		rotationAxis(collisionCellRotationAxes, collisionCellIndex);

	particleVelocity =
		relativeVelocity.getRotatedAroundNormalizedAxis(rotationAxis, srdCollisionAngle);

	#ifdef OPENMPCD_CUDA_DEBUG
		if(!particleVelocity.isFinite())
			printf(	"particleVelocity at the end of OpenMPCD::CUDA::DeviceCode::collisionCellFluidRotationStep1 is not finite. "
					"Values: %g %g %g\nrelativeVelocity: %g %g %g\n",
					particleVelocity.getX(), particleVelocity.getY(), particleVelocity.getZ(),
					relativeVelocity.getX(), relativeVelocity.getY(), relativeVelocity.getZ());
	#endif

	atomicAdd(
		&collisionCellFrameInternalKineticEnergies[collisionCellIndex],
		particleMass * relativeVelocity.getMagnitudeSquared() / 2.0);
	::atomicAdd(&collisionCellParticleCounts[collisionCellIndex], 1U);
}

__global__ void generateCollisionCellMBSFactors(
	const unsigned int workUnitOffset,
	const unsigned int collisionCellCount,
	const FP* const collisionCellFrameInternalKineticEnergies,
	unsigned int* const collisionCellParticleCounts,
	FP* const collisionCellRelativeVelocityScalings,
	const FP bulkThermostatTargetkT,
	GPURNG* const rngs)
{
	const unsigned int collisionCellID =
		blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;
	if(collisionCellID >= collisionCellCount)
		return;

	const unsigned int numberOfParticles =
		collisionCellParticleCounts[collisionCellID];

	if(
		numberOfParticles < 2 ||
		collisionCellFrameInternalKineticEnergies[collisionCellID] == 0)
	{
		collisionCellRelativeVelocityScalings[collisionCellID] = 1;
		return;
	}

	const FP fOver2 = 1.5 * (numberOfParticles - 1);
	Random::Distributions::Gamma_shape_ge_1<FP> dist(
		fOver2, bulkThermostatTargetkT);

	const FP alphaSquared =
		dist(rngs[collisionCellID]) /
		collisionCellFrameInternalKineticEnergies[collisionCellID];

	collisionCellRelativeVelocityScalings[collisionCellID] = sqrt(alphaSquared);
}

__global__ void collisionCellStochasticRotationStep2(
	const unsigned int workUnitOffset,
	const unsigned int particleCount,
	MPCParticleVelocityType* const velocities,
	const unsigned int* const collisionCellIndices,
	const MPCParticleVelocityType* const collisionCellMomenta,
	const FP* const collisionCellMasses,
	const FP* const collisionCellVelocityScalings)
{
	const unsigned int particleID = blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;
	if(particleID >= particleCount)
		return;

	const unsigned int collisionCellIndex = collisionCellIndices[particleID];

	RemotelyStoredVector<MPCParticleVelocityType>
		particleVelocity(velocities, particleID);

	#ifdef OPENMPCD_CUDA_DEBUG
		if(!particleVelocity.isFinite())
		{
			printf(
				"%s %g %g %g (Particle ID: %u)\n",
				"Bad particleVelocity given in "
					"OpenMPCD::CUDA::DeviceCode::"
					"collisionCellStochasticRotationStep2:",
				particleVelocity.getX(), particleVelocity.getY(), particleVelocity.getZ(), particleID);
		}

		if(!isfinite(collisionCellVelocityScalings[collisionCellIndex]))
		{
			printf(
				"%s %g (Particle ID: %u, Collision Cell Index: %u)\n",
				"Bad collisionCellVelocityScalings given in "
					"OpenMPCD::CUDA::DeviceCode::"
					"collisionCellStochasticRotationStep2:",
				collisionCellVelocityScalings[collisionCellIndex],
				particleID,
				collisionCellIndex);
		}
	#endif

	particleVelocity *= collisionCellVelocityScalings[collisionCellIndex];
	particleVelocity += getCollisionCellCenterOfMassVelocity(collisionCellIndex, collisionCellMomenta, collisionCellMasses);

	#ifdef OPENMPCD_CUDA_DEBUG
		if(!particleVelocity.isFinite())
		{
			printf(
				"%s %g %g %g (Particle ID: %u)\n",
				"Bad particleVelocity calculated in OpenMPCD::CUDA::DeviceCode::collisionCellFluidRotationStep2:",
				particleVelocity.getX(), particleVelocity.getY(), particleVelocity.getZ(), particleID);
		}
	#endif
}

__global__ void undoLeesEdwardsVelocityCorrections(
	const unsigned int workUnitOffset,
	const unsigned int particleCount,
	MPCParticleVelocityType* const velocities,
	const MPCParticleVelocityType* const velocityCorrections)
{
	const unsigned int particleID = blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;
	if(particleID >= particleCount)
		return;

	#ifdef OPENMPCD_CUDA_DEBUG
		if(!isfinite(velocities[3 * particleID + 0]))
		{
			printf(
				"%s %g (Particle ID: %u)\n",
				"Bad velocity given in OpenMPCD::CUDA::DeviceCode::undoLeesEdwardsVelocityCorrections:",
				velocities[3 * particleID + 0], particleID);
		}

		if(!isfinite(velocityCorrections[particleID]))
		{
			printf(
				"%s %g (Particle ID: %u)\n",
				"Bad correction given in OpenMPCD::CUDA::DeviceCode::undoLeesEdwardsVelocityCorrections:",
				velocityCorrections[particleID], particleID);
		}
	#endif

	velocities[3 * particleID + 0] -= velocityCorrections[particleID];

	#ifdef OPENMPCD_CUDA_DEBUG
		if(!isfinite(velocities[3 * particleID + 0]))
		{
			printf(
				"%s %g (Particle ID: %u)\n",
				"Bad velocity calculated in OpenMPCD::CUDA::DeviceCode::undoLeesEdwardsVelocityCorrections:",
				velocities[3 * particleID + 0], particleID);
		}
	#endif
}

__global__
void constructGPURNGs(
	const std::size_t count,
	GPURNG* const location,
	const unsigned long long seed)
{
	const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadID != 0)
		return;

	for(std::size_t i = 0; i < count; ++i)
		new(location + i) GPURNG(seed, i);
}

__global__
void destroyGPURNGs(const std::size_t count, GPURNG* const location)
{
	const unsigned int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if(threadID != 0)
		return;

	for(std::size_t i = 0; i < count; ++i)
		location[i].~GPURNG();
}

} //namespace DeviceCode
} //namespace CUDA
} //namespace OpenMPCD
