#include <OpenMPCD/LeesEdwardsBoundaryConditions.hpp>

#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>

const OpenMPCD::Vector3D<OpenMPCD::MPCParticlePositionType>
	OpenMPCD::getImageUnderLeesEdwardsBoundaryConditions(
		const Vector3D<MPCParticlePositionType>& position,
		const FP mpcTime,
		const FP shearRate,
		const unsigned int simBoxX, const unsigned int simBoxY, const unsigned int simBoxZ,
		MPCParticleVelocityType* const velocityCorrection)
{
	const FP relativeLayerVelocity = shearRate * simBoxY;
	const FP layerDisplacement = relativeLayerVelocity * mpcTime; //delrx in Kowalik and Winkler's paper

	FP x = position.getX();
	FP y = position.getY();
	FP z = position.getZ();

	const double layerY = floor(y / simBoxY);
		//< cory in Kowalik and Winkler's paper

	const double layerZ = floor(z / simBoxZ);

	x -= layerY * layerDisplacement;

	const double layerX = floor(x / simBoxX);

	x -= layerX * simBoxX;
	y -= layerY * simBoxY;
	z -= layerZ * simBoxZ;

	#ifdef OPENMPCD_DEBUG
		if(velocityCorrection == NULL)
			OPENMPCD_THROW(NULLPointerException, "velocityCorrection");
	#endif

	*velocityCorrection = - layerY * relativeLayerVelocity;

	OPENMPCD_DEBUG_ASSERT(x >= 0 && x < simBoxX);
	OPENMPCD_DEBUG_ASSERT(y >= 0 && y < simBoxY);
	OPENMPCD_DEBUG_ASSERT(z >= 0 && z < simBoxZ);

	return Vector3D<MPCParticlePositionType>(x, y, z);
}
