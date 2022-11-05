#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Base.hpp>

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Symbols.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>
#include <OpenMPCD/RemotelyStoredVector.hpp>

void OpenMPCD::CUDA::MPCFluid::DeviceCode::setMPCParticleCountSymbol(const unsigned int count)
{
	cudaMemcpyToSymbol(mpcParticleCount, &count, sizeof(count));
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{
namespace DeviceCode
{

__global__ void computeLogicalEntityCentersOfMass(
	const unsigned int workUnitOffset,
	const MPCParticlePositionType* const positions,
	const unsigned int numberOfLogicalEntities,
	const unsigned int particlesPerLogicalEntity,
	MPCParticlePositionType* const output)
{
	OPENMPCD_DEBUG_ASSERT(positions);
	OPENMPCD_DEBUG_ASSERT(particlesPerLogicalEntity > 0);
	OPENMPCD_DEBUG_ASSERT(output);


	const std::size_t entityID =
		blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;

	if(entityID >= numberOfLogicalEntities)
		return;

	RemotelyStoredVector<MPCParticlePositionType> result(output, entityID);

	result.setX(0);
	result.setY(0);
	result.setZ(0);

	for(unsigned int p = 0; p < particlesPerLogicalEntity; ++p)
	{
		const RemotelyStoredVector<const MPCParticlePositionType>
			pos(positions, entityID * particlesPerLogicalEntity + p);

		result += pos;
	}

	result /= MPCParticlePositionType(particlesPerLogicalEntity);
}

} //namespace DeviceCode
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD
