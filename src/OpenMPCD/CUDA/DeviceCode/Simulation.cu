/**
 * @file
 * Defines the functions declared in `OpenMPCD/CUDA/DeviceCode/Simulation.hpp`.
 */

#include <OpenMPCD/CUDA/DeviceCode/Simulation.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace DeviceCode
{

__global__ void resetCollisionCellData(
	const unsigned int workUnitOffset,
	const unsigned int collisionCellCount,
	unsigned int* const collisionCellParticleCounts,
	MPCParticleVelocityType* const collisionCellMomenta,
	FP* const collisionCellFrameInternalKineticEnergies,
	FP* const collisionCellMasses)
{
	const unsigned int collisionCellID =
		blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;

	if(collisionCellID >= collisionCellCount)
		return;

	OPENMPCD_DEBUG_ASSERT(collisionCellParticleCounts != NULL);
	OPENMPCD_DEBUG_ASSERT(collisionCellMomenta != NULL);
	OPENMPCD_DEBUG_ASSERT(collisionCellFrameInternalKineticEnergies != NULL);
	OPENMPCD_DEBUG_ASSERT(collisionCellMasses != NULL);

	collisionCellParticleCounts[collisionCellID] = 0;

	collisionCellMomenta[3 * collisionCellID + 0] = 0;
	collisionCellMomenta[3 * collisionCellID + 1] = 0;
	collisionCellMomenta[3 * collisionCellID + 2] = 0;

	collisionCellFrameInternalKineticEnergies[collisionCellID] = 0;

	collisionCellMasses[collisionCellID] = 0;
}

} //namespace DeviceCode
} //namespace CUDA
} //namespace OpenMPCD
