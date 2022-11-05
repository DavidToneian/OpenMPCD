/**
 * @file
 * Tests functionality in `OpenMPCD/CUDA/DeviceCode/Symbols.hpp`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/DeviceCode/Symbols.hpp>

#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>

SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::setSimulationBoxSizeSymbols`",
	"[CUDA]")
{
	using namespace OpenMPCD::CUDA;
	using namespace OpenMPCD::CUDA::DeviceCode;

	static const unsigned int boxX = 4;
	static const unsigned int boxY = 5;
	static const unsigned int boxZ = 6;

	setSimulationBoxSizeSymbols(boxX, boxY, boxZ);

	unsigned int tmp;

	cudaMemcpyFromSymbol(
		&tmp, mpcSimulationBoxSizeX, sizeof(tmp), 0, cudaMemcpyDeviceToHost);
	REQUIRE(tmp == boxX);

	cudaMemcpyFromSymbol(
		&tmp, mpcSimulationBoxSizeY, sizeof(tmp), 0, cudaMemcpyDeviceToHost);
	REQUIRE(tmp == boxY);

	cudaMemcpyFromSymbol(
		&tmp, mpcSimulationBoxSizeZ, sizeof(tmp), 0, cudaMemcpyDeviceToHost);
	REQUIRE(tmp == boxZ);

	cudaMemcpyFromSymbol(
		&tmp, collisionCellCount, sizeof(tmp), 0, cudaMemcpyDeviceToHost);
	REQUIRE(tmp == boxX * boxY * boxZ);

	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			setSimulationBoxSizeSymbols(1, 1, 0),
			OpenMPCD::InvalidArgumentException);

		REQUIRE_THROWS_AS(
			setSimulationBoxSizeSymbols(1, 0, 1),
			OpenMPCD::InvalidArgumentException);

		REQUIRE_THROWS_AS(
			setSimulationBoxSizeSymbols(0, 1, 1),
			OpenMPCD::InvalidArgumentException);

		REQUIRE_THROWS_AS(
			setSimulationBoxSizeSymbols(1, 0, 0),
			OpenMPCD::InvalidArgumentException);

		REQUIRE_THROWS_AS(
			setSimulationBoxSizeSymbols(0, 1, 0),
			OpenMPCD::InvalidArgumentException);

		REQUIRE_THROWS_AS(
			setSimulationBoxSizeSymbols(0, 0, 1),
			OpenMPCD::InvalidArgumentException);

		REQUIRE_THROWS_AS(
			setSimulationBoxSizeSymbols(0, 0, 0),
			OpenMPCD::InvalidArgumentException);
	#endif
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::setMPCStreamingTimestep`",
	"[CUDA]")
{
	using namespace OpenMPCD::CUDA;
	using namespace OpenMPCD::CUDA::DeviceCode;

	static const OpenMPCD::FP timestep = 0.1234;

	setMPCStreamingTimestep(timestep);

	OpenMPCD::FP tmp;

	cudaMemcpyFromSymbol(
		&tmp, streamingTimestep, sizeof(tmp), 0, cudaMemcpyDeviceToHost);
	REQUIRE(tmp == timestep);

	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			setMPCStreamingTimestep(0),
			OpenMPCD::InvalidArgumentException);
	#endif
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::setSRDCollisionAngleSymbol`",
	"[CUDA]")
{
	using namespace OpenMPCD::CUDA;
	using namespace OpenMPCD::CUDA::DeviceCode;

	static const OpenMPCD::FP angle = 1.234;

	setSRDCollisionAngleSymbol(angle);

	OpenMPCD::FP tmp;

	cudaMemcpyFromSymbol(
		&tmp, srdCollisionAngle, sizeof(tmp), 0, cudaMemcpyDeviceToHost);
	REQUIRE(tmp == angle);
}
