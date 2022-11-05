/**
 * @file
 * Tests functionality in
 * `OpenMPCD/CUDA/DeviceCode/LeesEdwardsBoundaryConditions.hpp`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/DeviceCode/LeesEdwardsBoundaryConditions.hpp>

#include <OpenMPCD/CUDA/DeviceCode/Simulation.hpp>
#include <OpenMPCD/CUDA/DeviceCode/Symbols.hpp>
#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>

SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::setLeesEdwardsSymbols`",
	"[CUDA]")
{
	using namespace OpenMPCD;
	using namespace OpenMPCD::CUDA::DeviceCode;

	static const FP shearRates[] = {0.1, 0.0, -1};
	static const std::size_t shearRatesCount =
		sizeof(shearRates) / sizeof(shearRates[0]);

	static const unsigned int sizes[] = {1, 3, 20};
	static const std::size_t sizesCount = sizeof(sizes) / sizeof(sizes[0]);

	for(std::size_t shearIndex = 0; shearIndex < shearRatesCount; ++shearIndex)
	{
		for(std::size_t sizeIndex = 0; sizeIndex < sizesCount; ++sizeIndex)
		{
			const FP shearRate = shearRates[shearIndex];
			const unsigned int simBoxY = sizes[sizeIndex];

			setLeesEdwardsSymbols(shearRate, simBoxY);

			FP readLeesEdwardsRelativeLayerVelocity;

			cudaMemcpyFromSymbol(
				&readLeesEdwardsRelativeLayerVelocity,
				leesEdwardsRelativeLayerVelocity,
				sizeof(leesEdwardsRelativeLayerVelocity),
				0,
				cudaMemcpyDeviceToHost);
			OPENMPCD_CUDA_THROW_ON_ERROR;

			const FP expected = shearRate * simBoxY;
			REQUIRE(readLeesEdwardsRelativeLayerVelocity == expected);
		}
	}
}


__global__ void getImageUnderLeesEdwardsBoundaryConditions_test_kernel(
	const OpenMPCD::FP mpcTime,
	const OpenMPCD::MPCParticlePositionType posX,
	const OpenMPCD::MPCParticlePositionType posY,
	const OpenMPCD::MPCParticlePositionType posZ,
	OpenMPCD::MPCParticlePositionType* const ret,
	OpenMPCD::MPCParticleVelocityType* const velocityCorrection)
{
	using namespace OpenMPCD;
	using namespace OpenMPCD::CUDA::DeviceCode;

	const Vector3D<MPCParticlePositionType> position(posX, posY, posZ);
	const Vector3D<MPCParticlePositionType> image =
		getImageUnderLeesEdwardsBoundaryConditions(
			mpcTime,
			position,
			*velocityCorrection);
	ret[0] = image.getX();
	ret[1] = image.getY();
	ret[2] = image.getZ();
}
static void getImageUnderLeesEdwardsBoundaryConditions_test(
	const OpenMPCD::FP shearRate,
	const unsigned int (&simBox)[3],
	const OpenMPCD::FP mpcTime,
	const OpenMPCD::MPCParticlePositionType (&pos)[3])
{
	using namespace OpenMPCD;
	using namespace OpenMPCD::CUDA;
	using namespace OpenMPCD::CUDA::DeviceCode;

	setLeesEdwardsSymbols(shearRate, simBox[1]);
	setSimulationBoxSizeSymbols(simBox[0], simBox[1], simBox[2]);

	DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	MPCParticlePositionType* ret;
	MPCParticleVelocityType* velocityCorrection;
	dmm.allocateMemory(&ret, 3);
	dmm.allocateMemory(&velocityCorrection, 1);


	getImageUnderLeesEdwardsBoundaryConditions_test_kernel<<<1, 1>>>(
		mpcTime,
		pos[0], pos[1], pos[2],
		ret, velocityCorrection);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	MPCParticlePositionType expectedReturnValue[3] = {pos[0], pos[1], pos[2]};
	MPCParticleVelocityType expectedVelocityCorrection = 0;
	int yLayer = 0;

	while(expectedReturnValue[1] < 0)
	{
		expectedReturnValue[1] += simBox[1];
		--yLayer;
		expectedVelocityCorrection += shearRate * simBox[1];
	}
	while(expectedReturnValue[1] >= simBox[1])
	{
		expectedReturnValue[1] -= simBox[1];
		++yLayer;
		expectedVelocityCorrection -= shearRate * simBox[1];
	}

	expectedReturnValue[0] -= yLayer * shearRate * simBox[1] * mpcTime;

	for(std::size_t coord = 0; coord < 3; ++coord)
	{
		if(coord == 1)
			continue;

		while(expectedReturnValue[coord] < 0)
			expectedReturnValue[coord] += simBox[coord];
		while(expectedReturnValue[coord] >= simBox[coord])
			expectedReturnValue[coord] -= simBox[coord];
	}


	//sanity checks
	for(std::size_t coord = 0; coord < 3; ++coord)
	{
		REQUIRE(expectedReturnValue[coord] >= 0);
		REQUIRE(expectedReturnValue[coord] < simBox[coord]);
	}


	MPCParticlePositionType returnValueOnHost[3];
	dmm.copyElementsFromDeviceToHost(
		ret, returnValueOnHost, 3);

	for(std::size_t coord = 0; coord < 3; ++coord)
	{
		REQUIRE(returnValueOnHost[coord] >= 0);
		REQUIRE(returnValueOnHost[coord] < simBox[coord]);

		if(returnValueOnHost[coord] != Approx(expectedReturnValue[coord]))
		{
			REQUIRE(
				fabs(returnValueOnHost[coord] - expectedReturnValue[coord])
				==
				Approx(simBox[coord]));
		}
	}

	FP velocityCorrectionOnHost;
	dmm.copyElementsFromDeviceToHost(
		velocityCorrection, &velocityCorrectionOnHost, 1);
	REQUIRE(velocityCorrectionOnHost == Approx(expectedVelocityCorrection));
}
SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::getImageUnderLeesEdwardsBoundaryConditions`",
	"[CUDA]")
{
	using namespace OpenMPCD;

	static const FP shearRates[] = {-3.4, -0.2, 0, 0.1, 5.8};
	static const unsigned int simBoxSizes[][3] =
		{ {1, 1, 1}, {2, 3, 3}, {2, 3, 4} };
	static FP mpcTimes[] = {-3.1415, 0, 1, 12.34, 100};
	static const MPCParticlePositionType positions[][3] =
		{ {0, 0, 0}, {0.4, 0.4, 0.4}, {-20, 9.3, 5.1}, {300, -400.5, 500} };

	#define ITERATE_INDICES(index, array) \
		for( \
			std::size_t index = 0; \
			index < sizeof(array) / sizeof(array[0]); \
			++index)

	ITERATE_INDICES(i, shearRates)
	ITERATE_INDICES(j, simBoxSizes)
	ITERATE_INDICES(k, mpcTimes)
	ITERATE_INDICES(l, positions)
	{
		getImageUnderLeesEdwardsBoundaryConditions_test(
			shearRates[i],
			simBoxSizes[j],
			mpcTimes[k],
			positions[l]);
	}


	#undef ITERATE_INDICES
}

