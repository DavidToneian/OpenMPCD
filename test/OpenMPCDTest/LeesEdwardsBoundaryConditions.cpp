/**
 * @file
 * Tests functionality in `OpenMPCD/LeesEdwardsBoundaryConditions.hpp`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/LeesEdwardsBoundaryConditions.hpp>

static void getImageUnderLeesEdwardsBoundaryConditions_test(
	const OpenMPCD::FP shearRate,
	const unsigned int (&simBox)[3],
	const OpenMPCD::FP mpcTime,
	const OpenMPCD::MPCParticlePositionType (&pos)[3])
{
	using namespace OpenMPCD;

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


	const Vector3D<MPCParticlePositionType> position(
		pos[0], pos[1], pos[2]);

	MPCParticleVelocityType resultVelocityCorrection;
	const Vector3D<MPCParticlePositionType> resultPositionV =
		getImageUnderLeesEdwardsBoundaryConditions(
			position, mpcTime, shearRate,
			simBox[0], simBox[1], simBox[2],
			&resultVelocityCorrection);

	const MPCParticleVelocityType resultPosition[3] =
		{
			resultPositionV.getX(),
			resultPositionV.getY(),
			resultPositionV.getZ()
		};


	for(std::size_t coord = 0; coord < 3; ++coord)
	{
		REQUIRE(resultPosition[coord] >= 0);
		REQUIRE(resultPosition[coord] < simBox[coord]);

		if(resultPosition[coord] != Approx(expectedReturnValue[coord]))
		{
			REQUIRE(
				fabs(resultPosition[coord] - expectedReturnValue[coord])
				==
				Approx(simBox[coord]));
		}
	}

	REQUIRE(resultVelocityCorrection == Approx(expectedVelocityCorrection));

	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			getImageUnderLeesEdwardsBoundaryConditions(
				position, mpcTime, shearRate,
				simBox[0], simBox[1], simBox[2],
				NULL),
			OpenMPCD::NULLPointerException);
	#endif
}
SCENARIO(
	"`OpenMPCD::getImageUnderLeesEdwardsBoundaryConditions`",
	"")
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

