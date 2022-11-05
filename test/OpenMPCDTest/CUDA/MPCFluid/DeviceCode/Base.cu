/**
 * @file
 * Tests `OpenMPCD/CUDA/MPCFluid/DeviceCode/Base.hpp`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Base.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>

SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::DeviceCode::computeLogicalEntityCentersOfMass`",
	"[CUDA]")
{
	static const unsigned int particleCount = 2 * 3 * 4 * 5 * 6;

	using
		OpenMPCD::CUDA::MPCFluid::DeviceCode::computeLogicalEntityCentersOfMass;

	OpenMPCD::MPCParticlePositionType hostPositions[3 * particleCount];
	for(unsigned int p = 0; p < particleCount; ++p)
	{
		hostPositions[3 * p + 0] = 0.1 * p;
		hostPositions[3 * p + 1] = - 0.2 * p;
		hostPositions[3 * p + 2] = 1.23 + 1.0 / p;
	}

	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	OpenMPCD::MPCParticlePositionType* devicePositions;
	dmm.allocateMemory(&devicePositions, 3 * particleCount);
	dmm.copyElementsFromHostToDevice(
		hostPositions, devicePositions, 3 * particleCount);

	OpenMPCD::MPCParticlePositionType* result;
	dmm.allocateMemory(&result, 3 * particleCount);


	OpenMPCD::MPCParticlePositionType expected[3 * particleCount] = {0};

	for(unsigned int ppe = 1; ppe <= 6; ++ppe)
	{ //ppe = particles per entity
		const unsigned int entityCount = particleCount / ppe;
		REQUIRE(entityCount * ppe == particleCount);

		OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(2 * entityCount)
			computeLogicalEntityCentersOfMass<<<gridSize, blockSize>>>(
				workUnitOffset, devicePositions, entityCount, ppe, result);
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		for(unsigned int e = 0; e < entityCount; ++e)
		{
			expected[3 * e + 0] = 0;
			expected[3 * e + 1] = 0;
			expected[3 * e + 2] = 0;

			for(unsigned int p = 0; p < ppe; ++p)
			{
				expected[3 * e + 0] += hostPositions[3 * (e * ppe + p) + 0];
				expected[3 * e + 1] += hostPositions[3 * (e * ppe + p) + 1];
				expected[3 * e + 2] += hostPositions[3 * (e * ppe + p) + 2];
			}

			expected[3 * e + 0] /= ppe;
			expected[3 * e + 1] /= ppe;
			expected[3 * e + 2] /= ppe;
		}

		REQUIRE(
			dmm.elementMemoryEqualOnHostAndDevice(
				expected, result, 3 * particleCount));
	}
}
