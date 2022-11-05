#include <OpenMPCD/CUDA/Instrumentation.hpp>

#include <OpenMPCD/AnalyticQuantities.hpp>
#include <OpenMPCD/AnalyticQuantitiesGaussianDumbbell.hpp>
#include <OpenMPCD/CUDA/BoundaryCondition/LeesEdwards.hpp>
#include <OpenMPCD/CUDA/Device.hpp>
#include <OpenMPCD/LeesEdwardsBoundaryConditions.hpp>
#include <OpenMPCD/Profiling/CodeRegion.hpp>
#include <OpenMPCD/Utility/HostInformation.hpp>

#include <boost/filesystem.hpp>

#include <fstream>
#include <sstream>

using namespace OpenMPCD;

CUDA::Instrumentation::Instrumentation(
	const Simulation* const sim, const unsigned int rngSeed_,
	const std::string& gitRevision_)
		: simulation(sim), rngSeed(rngSeed_), gitRevision(gitRevision_),

		  constructionTimeUTC(
		      Utility::HostInformation::getCurrentUTCTimeAsString()),

		  autosave(false),

		  measurementCount(0)
{
	const Configuration config = sim->getConfiguration();

	if(!config.has("instrumentation"))
		return;

	const Configuration::Setting setting = config.getSetting("instrumentation");

	std::set<std::string> knownSettings;
	knownSettings.insert("velocityMagnitudeHistogram");
	knownSettings.insert("densityProfile");
	knownSettings.insert("flowProfile");

	knownSettings.insert("dumbbellBondLengthHistogram");
	knownSettings.insert("dumbbellBondLengthSquaredHistogram");
	knownSettings.insert("dumbbellBondXHistogram");
	knownSettings.insert("dumbbellBondYHistogram");
	knownSettings.insert("dumbbellBondZHistogram");
	knownSettings.insert("dumbbellBondXXHistogram");
	knownSettings.insert("dumbbellBondYYHistogram");
	knownSettings.insert("dumbbellBondZZHistogram");
	knownSettings.insert("dumbbellBondXYHistogram");
	knownSettings.insert("dumbbellBondAngleWithFlowDirectionHistogram");
	knownSettings.insert("dumbbellBondXYAngleWithFlowDirectionHistogram");
	knownSettings.insert("fourierTransformedVelocity");
	knownSettings.insert("gaussianChains");
	knownSettings.insert("gaussianRods");
	knownSettings.insert("harmonicTrimers");
	knownSettings.insert("logicalEntityMeanSquareDisplacement");
	knownSettings.insert("normalModeAutocorrelation");
	knownSettings.insert("selfDiffusionCoefficient");
	knownSettings.insert("velocityAutocorrelation");

	{
		std::string offender;
		if(!setting.childrenHaveNamesInCollection(knownSettings, &offender))
		{
			OPENMPCD_THROW(
				InvalidConfigurationException,
				std::string("Unknown setting in `instrumentation`: ") +
					offender);
		}
	}


	if(!sim->hasMPCFluid())
		return;

	if(setting.has("velocityMagnitudeHistogram"))
	{
		velocityMagnitudeHistogram.reset(
			new Histogram("velocityMagnitudeHistogram", config));
	}

	if(setting.has("densityProfile"))
	{
		densityProfile.reset(
			new DensityProfile(
				simulation->getSimulationBoxSizeX(),
				simulation->getSimulationBoxSizeY(),
				simulation->getSimulationBoxSizeZ(),
				setting.getSetting("densityProfile")));
	}

	if(setting.has("flowProfile"))
	{
		flowProfile.reset(
			new FlowProfile<MPCParticleVelocityType>(
				simulation->getSimulationBoxSizeX(),
				simulation->getSimulationBoxSizeY(),
				simulation->getSimulationBoxSizeZ(),
				setting.getSetting("flowProfile")));
	}
}

CUDA::Instrumentation::~Instrumentation()
{
	if(autosave)
		save(autosave_rundir);
}

void CUDA::Instrumentation::measure()
{
	const Profiling::CodeRegion codeRegion("measurement");

	measureMPCFluid();
	measureMPCSolute();

	++measurementCount;
}

void CUDA::Instrumentation::save(const std::string& rundir) const
{
	if(velocityMagnitudeHistogram)
	{
		velocityMagnitudeHistogram->save(
			rundir + "/velocityMagnitudeHistogram.data");
	}

	if(densityProfile)
		densityProfile->saveToFile(rundir + "/densityProfile.data");

	if(flowProfile)
		flowProfile->saveToFile(rundir + "/flowProfile.data");

	if(totalFluidVelocityMagnitudeVSTime)
	{
		totalFluidVelocityMagnitudeVSTime->save(
			rundir + "/totalFluidVelocityMagnitudeVSTime.data", false);
	}

	if(simulation->hasMPCFluid())
		simulation->getMPCFluid().getInstrumentation().save(rundir);

	if(simulation->hasSolute())
	{
		if(simulation->getMPCSolute().hasInstrumentation())
			simulation->getMPCSolute().getInstrumentation().save(rundir);
	}


	{
		std::stringstream ss;
		ss << "numberOfCompletedSweeps: ";
		ss << simulation->getNumberOfCompletedSweeps() << "\n";

		ss << "executingHost: \"";
		ss << Utility::HostInformation::getHostname() << "\"\n";

		Device device;
		ss << "DefaultCUDADevicePCIAddress: \"";
		ss << device.getPCIAddressString() << "\"\n";

		ss << "startTimeUTC: \"";
		ss << constructionTimeUTC << "\"\n";

		ss << "endTimeUTC: \"";
		ss << Utility::HostInformation::getCurrentUTCTimeAsString() << "\"\n";

		const std::string outname=rundir+"/metadata.txt";
		std::ofstream dst(outname.c_str(), std::ios::trunc);
		dst << ss.str();
	}


	saveStaticData(rundir);
}

void CUDA::Instrumentation::measureMPCFluid()
{
	if(!simulation->hasMPCFluid())
		return;

	const Profiling::CodeRegion codeRegion("fluid measurement");


	const BoundaryCondition::LeesEdwards* const leesEdwardsBoundaryConditions =
		dynamic_cast<const BoundaryCondition::LeesEdwards*>(
			simulation->getBoundaryConditions());

	if(leesEdwardsBoundaryConditions == NULL)
	{
		OPENMPCD_THROW(
			UnimplementedException,
			"Non-Lees-Edwards currently not supported");
	}


	if(flowProfile)
		flowProfile->newSweep();


	const MPCFluid::Base& mpcFluid = simulation->getMPCFluid();

	const unsigned int mpcParticleCount = mpcFluid.getParticleCount();

	mpcFluid.fetchFromDevice();

	Vector3D<MPCParticleVelocityType> totalFluidVelocity(0, 0, 0);

	for(unsigned int i=0; i<mpcParticleCount; ++i)
	{
		const RemotelyStoredVector<const MPCParticlePositionType> position =
			mpcFluid.getPosition(i);
		const RemotelyStoredVector<const MPCParticleVelocityType> velocity =
			mpcFluid.getVelocity(i);

		if(velocityMagnitudeHistogram)
			velocityMagnitudeHistogram->fill(velocity.getMagnitude());

		MPCParticleVelocityType velocityCorrection;
		const Vector3D<MPCParticlePositionType> image =
			getImageUnderLeesEdwardsBoundaryConditions(
				position,
				simulation->getMPCTime(),
				leesEdwardsBoundaryConditions->getShearRate(),
				simulation->getSimulationBoxSizeX(),
				simulation->getSimulationBoxSizeY(),
				simulation->getSimulationBoxSizeZ(),
				&velocityCorrection);

		Vector3D<MPCParticleVelocityType> correctedVelocity = velocity;
		correctedVelocity.addToX(velocityCorrection);

		if(densityProfile)
		{
			const std::size_t x =
				static_cast<std::size_t>(
					image.getX() * densityProfile->getCellSubdivisionsX());
			const std::size_t y =
				static_cast<std::size_t>(
					image.getY() * densityProfile->getCellSubdivisionsY());
			const std::size_t z =
				static_cast<std::size_t>(
					image.getZ() * densityProfile->getCellSubdivisionsZ());

			densityProfile->add(x, y, z, mpcFluid.getParticleMass());
		}

		if(flowProfile)
		{
			const std::size_t x =
				static_cast<std::size_t>(
					image.getX() * flowProfile->getCellSubdivisionsX());
			const std::size_t y =
				static_cast<std::size_t>(
					image.getY() * flowProfile->getCellSubdivisionsY());
			const std::size_t z =
				static_cast<std::size_t>(
					image.getZ() * flowProfile->getCellSubdivisionsZ());

			flowProfile->add(x, y, z, correctedVelocity);
		}

		totalFluidVelocity += correctedVelocity;
	}

	if(densityProfile)
		densityProfile->incrementFillCount();

	if(totalFluidVelocityMagnitudeVSTime)
	{
		totalFluidVelocityMagnitudeVSTime->addPoint(
			simulation->getMPCTime(), totalFluidVelocity.getMagnitude());
	}

	mpcFluid.getInstrumentation().measure();
}

void CUDA::Instrumentation::measureMPCSolute()
{
	if(!simulation->hasSolute())
		return;

	if(!simulation->getMPCSolute().hasInstrumentation())
		return;

	const Profiling::CodeRegion codeRegion("solute measurement");

	simulation->getMPCSolute().getInstrumentation().measure();
}

void CUDA::Instrumentation::saveStaticData(const std::string& rundir) const
{
	{
		std::stringstream ss;
		ss<<rngSeed;

		const std::string outname=rundir+"/rngSeed.txt";
		std::ofstream dst(outname.c_str(), std::ios::trunc);
		dst<<ss.str();
	}

	if(!boost::filesystem::exists(rundir + "/config.txt"))
		simulation->getConfiguration().writeToFile(rundir + "/config.txt");

	if(!gitRevision.empty())
	{
		const std::string outname=rundir+"/git-revision";
		std::ofstream dst(outname.c_str(), std::ios::trunc);
		dst<<gitRevision;
	}
}
