#include <OpenMPCD/CUDA/BoundaryCondition/LeesEdwards.hpp>

#include <OpenMPCD/CUDA/DeviceCode/LeesEdwardsBoundaryConditions.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace BoundaryCondition
{

LeesEdwards::LeesEdwards(
	const Configuration::Setting& config,
	const unsigned int simulationBoxSizeX,
	const unsigned int simulationBoxSizeY,
	const unsigned int simulationBoxSizeZ)
	: primarySimulationVolume(
		simulationBoxSizeX * simulationBoxSizeY * simulationBoxSizeZ)
{
	if(config.getChildCount() != 1)
	{
		OPENMPCD_THROW(
			InvalidConfigurationException,
			"Not the right amount of configuration settings.");
	}

	config.read("shearRate", &shearRate);


	DeviceCode::setLeesEdwardsSymbols(shearRate, simulationBoxSizeY);
}

FP LeesEdwards::getTotalAvailableVolume() const
{
	return primarySimulationVolume;
}

} //namespace BoundaryCondition
} //namespace CUDA
} //namespace OpenMPCD
