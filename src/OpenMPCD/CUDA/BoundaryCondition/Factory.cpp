#include <OpenMPCD/CUDA/BoundaryCondition/Factory.hpp>

#include <OpenMPCD/CUDA/BoundaryCondition/LeesEdwards.hpp>

#include <iostream>

namespace OpenMPCD
{
namespace CUDA
{
namespace BoundaryCondition
{

BoundaryCondition::Base* Factory::getInstance(
	const Configuration::Setting& config,
	const unsigned int simulationBoxSizeX,
	const unsigned int simulationBoxSizeY,
	const unsigned int simulationBoxSizeZ)
{
	if(config.getChildCount() == 0)
	{
		OPENMPCD_THROW(
			InvalidConfigurationException,
			"No boundary condition configured.");
	}

	if(config.getChildCount() == 2)
	{
		if(	config.read<std::string>("type") != "Lees-Edwards"
			||
			!config.has("shearRate"))
		{
			OPENMPCD_THROW(
				InvalidConfigurationException,
				"Boundary condition configuration malformed.");
		}

		std::cout << "Warning: This style of boundary condition configuration ";
		std::cout << "is deprecated.\n";

		Configuration newConfig;
		newConfig.set("LeesEdwards.shearRate", config.read<FP>("shearRate"));
		return new LeesEdwards(
			newConfig.getSetting("LeesEdwards"),
			simulationBoxSizeX, simulationBoxSizeY, simulationBoxSizeZ);
	}

	if(config.getChildCount() != 1)
	{
		OPENMPCD_THROW(
			InvalidConfigurationException,
			"Boundary condition configuration malformed.");
	}


	if(config.has("LeesEdwards"))
	{
		return new LeesEdwards(
			config.getSetting("LeesEdwards"),
			simulationBoxSizeX, simulationBoxSizeY, simulationBoxSizeZ);
	}

	OPENMPCD_THROW(
		InvalidConfigurationException,
		"Unknown boundary condition configured.");
}
} //namespace BoundaryCondition
} //namespace CUDA
} //namespace OpenMPCD
