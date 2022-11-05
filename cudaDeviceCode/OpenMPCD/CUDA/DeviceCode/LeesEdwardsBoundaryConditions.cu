#include <OpenMPCD/CUDA/DeviceCode/LeesEdwardsBoundaryConditions.hpp>

#include <OpenMPCD/CUDA/Exceptions.hpp>
#include <OpenMPCD/CUDA/DeviceCode/Symbols.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>

__constant__ OpenMPCD::FP OpenMPCD::CUDA::DeviceCode::leesEdwardsRelativeLayerVelocity;

void OpenMPCD::CUDA::DeviceCode::setLeesEdwardsSymbols(const FP shearRate, const unsigned int simBoxY)
{
	const FP relativeLayerVelocity = shearRate * simBoxY;

	cudaMemcpyToSymbol(leesEdwardsRelativeLayerVelocity, &relativeLayerVelocity, sizeof(relativeLayerVelocity));
	OPENMPCD_CUDA_THROW_ON_ERROR;
}

__device__ const OpenMPCD::Vector3D<OpenMPCD::MPCParticlePositionType>
	OpenMPCD::CUDA::DeviceCode::getImageUnderLeesEdwardsBoundaryConditions(
		const FP mpcTime,
		const Vector3D<MPCParticlePositionType>& position,
		MPCParticleVelocityType& velocityCorrection)
{
	const FP layerDisplacement = leesEdwardsRelativeLayerVelocity * mpcTime; //delrx in Kowalik and Winkler's paper

	FP x = position.getX();
	FP y = position.getY();
	FP z = position.getZ();

	#ifdef OPENMPCD_CUDA_DEBUG
		if(isnan(x) || isnan(y) || isnan(z))
		{
			printf(
				"%s %g %g %g\n",
				"Bad coordinates passed to "
					"OpenMPCD::CUDA::DeviceCode::getImageUnderLeesEdwardsBoundaryConditions: ",
				x, y, z);
		}
	#endif

	const double layerY = floor(y / mpcSimulationBoxSizeY);
		//< cory in Kowalik and Winkler's paper

	const double layerZ = floor(z / mpcSimulationBoxSizeZ);

	x -= layerY * layerDisplacement;

	const double layerX = floor(x / mpcSimulationBoxSizeX);

	x -= layerX * mpcSimulationBoxSizeX;
	y -= layerY * mpcSimulationBoxSizeY;
	z -= layerZ * mpcSimulationBoxSizeZ;

	#ifdef OPENMPCD_CUDA_DEBUG
		if(!isfinite(x) || !isfinite(y) || !isfinite(z))
		{
			printf(
				"%s %g %g %g\n",
				"Bad coordinates computed in "
					"OpenMPCD::CUDA::DeviceCode::getImageUnderLeesEdwardsBoundaryConditions: ",
				x, y, z);
		}
	#endif

	#ifdef OPENMPCD_CUDA_DEBUG
		if(x < 0)
		{
			printf(
				"%s %g %g %g\n%s %g %g\n",
				"Bad coordinates computed in "
					"OpenMPCD::CUDA::DeviceCode::getImageUnderLeesEdwardsBoundaryConditions:",
				x, y, z,
				"Original coordinates:",
				position.getX(), position.getY(), position.getZ());
		}
	#endif

	OPENMPCD_DEBUG_ASSERT(x >= 0);
	OPENMPCD_DEBUG_ASSERT(x < mpcSimulationBoxSizeX);
	OPENMPCD_DEBUG_ASSERT(y >= 0);
	OPENMPCD_DEBUG_ASSERT(y < mpcSimulationBoxSizeY);
	OPENMPCD_DEBUG_ASSERT(z >= 0);
	OPENMPCD_DEBUG_ASSERT(z < mpcSimulationBoxSizeZ);

	velocityCorrection = -layerY * leesEdwardsRelativeLayerVelocity;

	return Vector3D<MPCParticlePositionType>(x, y, z);
}
