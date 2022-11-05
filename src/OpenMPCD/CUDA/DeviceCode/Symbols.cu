/**
 * @file
 * Defines the symbols declared in `OpenMPCD/CUDA/DeviceCode/Symbols.hpp`.
 */

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/Types.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace DeviceCode
{

__constant__ unsigned int mpcSimulationBoxSizeX;
__constant__ unsigned int mpcSimulationBoxSizeY;
__constant__ unsigned int mpcSimulationBoxSizeZ;

__constant__ unsigned int collisionCellCount;

__constant__ FP streamingTimestep;

__constant__ FP srdCollisionAngle;


void setSimulationBoxSizeSymbols(
	const unsigned int simBoxSizeX, const unsigned int simBoxSizeY,
	const unsigned int simBoxSizeZ)
{
	#ifdef OPENMPCD_DEBUG
		if(simBoxSizeX == 0)
			OPENMPCD_THROW(InvalidArgumentException, "`simBoxSizeX`");
		if(simBoxSizeY == 0)
			OPENMPCD_THROW(InvalidArgumentException, "`simBoxSizeY`");
		if(simBoxSizeZ == 0)
			OPENMPCD_THROW(InvalidArgumentException, "`simBoxSizeZ`");
	#endif

	cudaMemcpyToSymbol(mpcSimulationBoxSizeX, &simBoxSizeX, sizeof(simBoxSizeX));
	OPENMPCD_CUDA_THROW_ON_ERROR;
	cudaMemcpyToSymbol(mpcSimulationBoxSizeY, &simBoxSizeY, sizeof(simBoxSizeY));
	OPENMPCD_CUDA_THROW_ON_ERROR;
	cudaMemcpyToSymbol(mpcSimulationBoxSizeZ, &simBoxSizeZ, sizeof(simBoxSizeZ));
	OPENMPCD_CUDA_THROW_ON_ERROR;

	const unsigned int collisionCellCount_ = simBoxSizeX * simBoxSizeY * simBoxSizeZ;
	cudaMemcpyToSymbol(collisionCellCount, &collisionCellCount_, sizeof(collisionCellCount_));
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

void setMPCStreamingTimestep(const FP timestep)
{
	#ifdef OPENMPCD_DEBUG
		if(timestep == 0)
			OPENMPCD_THROW(InvalidArgumentException, "`timestep`");
	#endif

	cudaMemcpyToSymbol(streamingTimestep, &timestep, sizeof(timestep));
	OPENMPCD_CUDA_THROW_ON_ERROR;
}


void setSRDCollisionAngleSymbol(const FP angle)
{
	cudaMemcpyToSymbol(srdCollisionAngle, &angle, sizeof(angle));
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

} //namespace DeviceCode
} //namespace CUDA
} //namespace OpenMPCD
