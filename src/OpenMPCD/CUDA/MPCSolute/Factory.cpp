#include <OpenMPCD/CUDA/MPCSolute/Factory.hpp>

#include <OpenMPCD/CUDA/MPCSolute/StarPolymers.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <cmath>

using namespace OpenMPCD;
using namespace OpenMPCD::CUDA;

MPCSolute::Base<MPCParticlePositionType, MPCParticleVelocityType>*
MPCSolute::Factory::getInstance(
	CUDA::Simulation* const sim,
	const Configuration& config,
	RNG& rng)
{
	if(!config.has("solute"))
		return NULL;

	Configuration::Setting soluteGroup = config.getSetting("solute");

	if(soluteGroup.getChildCount() == 0)
		return NULL;

	if(soluteGroup.getChildCount() != 1)
	{
		OPENMPCD_THROW(
			UnimplementedException,
			"Currently, only a single type of solutes is supported");
	}

	if(soluteGroup.has("StarPolymers"))
	{
		return
			new MPCSolute::StarPolymers<
				MPCParticlePositionType, MPCParticleVelocityType>(
					soluteGroup.getSetting("StarPolymers"),
					sim->getBoundaryConditions());
	}

	OPENMPCD_THROW(UnimplementedException, "Unknown solute type");
}

