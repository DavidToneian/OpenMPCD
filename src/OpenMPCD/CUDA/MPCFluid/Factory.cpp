#include <OpenMPCD/CUDA/MPCFluid/Factory.hpp>

#include <OpenMPCD/CUDA/MPCFluid/GaussianChains.hpp>
#include <OpenMPCD/CUDA/MPCFluid/GaussianDumbbells.hpp>
#include <OpenMPCD/CUDA/MPCFluid/GaussianRods.hpp>
#include <OpenMPCD/CUDA/MPCFluid/HarmonicTrimers.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Simple.hpp>
#include <OpenMPCD/Exceptions.hpp>

using namespace OpenMPCD;
using namespace OpenMPCD::CUDA;

MPCFluid::Base* MPCFluid::Factory::getInstance(CUDA::Simulation* const sim,
			                                   const Configuration& config,
			                                   const unsigned int count,
			                                   RNG& rng)
{
	const FP mpcTimestep = config.read<FP>("mpc.timestep");

	if(!config.has("mpc.fluid"))
		return NULL;

	if(config.has("mpc.fluid.simple"))
		return new MPCFluid::Simple(sim, count, mpcTimestep, rng, sim->getDeviceMemoryManager());

	if(config.has("mpc.fluid.dumbbell"))
	{
		const FP bulkThermostatTargetkT = config.read<FP>("bulkThermostat.targetkT");
		const FP shearRate = config.read<FP>("boundaryConditions.LeesEdwards.shearRate");

		return new MPCFluid::GaussianDumbbells(
			sim, count, mpcTimestep, rng, sim->getDeviceMemoryManager(),
			bulkThermostatTargetkT, shearRate);
	}

	if(config.has("mpc.fluid.gaussianChains"))
	{
		return new MPCFluid::GaussianChains(
			sim, count, mpcTimestep, rng, sim->getDeviceMemoryManager());
	}

	if(config.has("mpc.fluid.gaussianRods"))
	{
		return new MPCFluid::GaussianRods(
			sim, count, mpcTimestep, rng, sim->getDeviceMemoryManager());
	}

	if(config.has("mpc.fluid.harmonicTrimers"))
		return new MPCFluid::HarmonicTrimers(sim, count, mpcTimestep, rng, sim->getDeviceMemoryManager());

	OPENMPCD_THROW(InvalidConfigurationException, "Invalid MPC fluid type.");
}
