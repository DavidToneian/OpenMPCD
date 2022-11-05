#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/GaussianDumbbells.hpp>

#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/FourierTransformedVelocity/Doublets.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/VelocityAutocorrelation/Doublets.hpp>

using namespace OpenMPCD::CUDA::MPCFluid::Instrumentation;

GaussianDumbbells::GaussianDumbbells(
	const Simulation* const sim, DeviceMemoryManager* const devMemMgr,
	const MPCFluid::GaussianDumbbells* const mpcFluid_)
		: Base(sim->getConfiguration(), mpcFluid_),
		  simulation(sim), mpcFluid(mpcFluid_),
		  dumbbellBondLengthHistogram("dumbbellBondLengthHistogram", sim->getConfiguration()),
		  dumbbellBondLengthSquaredHistogram("dumbbellBondLengthSquaredHistogram", sim->getConfiguration()),

		  dumbbellBondXHistogram("dumbbellBondXHistogram", sim->getConfiguration()),
		  dumbbellBondYHistogram("dumbbellBondYHistogram", sim->getConfiguration()),
		  dumbbellBondZHistogram("dumbbellBondZHistogram", sim->getConfiguration()),

		  dumbbellBondXXHistogram("dumbbellBondXXHistogram", sim->getConfiguration()),
		  dumbbellBondYYHistogram("dumbbellBondYYHistogram", sim->getConfiguration()),
		  dumbbellBondZZHistogram("dumbbellBondZZHistogram", sim->getConfiguration()),

		  dumbbellBondXYHistogram("dumbbellBondXYHistogram", sim->getConfiguration()),

		  dumbbellBondAngleWithFlowDirectionHistogram("dumbbellBondAngleWithFlowDirectionHistogram", sim->getConfiguration()),
		  dumbbellBondXYAngleWithFlowDirectionHistogram("dumbbellBondXYAngleWithFlowDirectionHistogram", sim->getConfiguration())
{
	fourierTransformedVelocity = new FourierTransformedVelocity::Doublets(sim, devMemMgr, mpcFluid_);
	velocityAutocorrelation    = new VelocityAutocorrelation::Doublets(sim, devMemMgr, mpcFluid_);
}

void GaussianDumbbells::measureSpecific()
{
	static const Vector3D<MPCParticleVelocityType> flowDirection(1, 0, 0);

	FP averageBondXX = 0;
	FP averageBondYY = 0;
	FP averageBondZZ = 0;
	for(unsigned int i=0; i<mpcFluid->getParticleCount(); i+=2)
	{
		const RemotelyStoredVector<const MPCParticlePositionType> r_1 = mpcFluid->getPosition(i);
		const RemotelyStoredVector<const MPCParticlePositionType> r_2 = mpcFluid->getPosition(i+1);

		const Vector3D<MPCParticlePositionType> R = r_2 - r_1;

		dumbbellBondLengthHistogram.fill(R.magnitude());
		dumbbellBondLengthSquaredHistogram.fill(R.magnitudeSquared());

		dumbbellBondXHistogram.fill(R.getX());
		dumbbellBondYHistogram.fill(R.getY());
		dumbbellBondZHistogram.fill(R.getZ());

		const FP bondXX = R.getX()*R.getX();
		const FP bondYY = R.getY()*R.getY();
		const FP bondZZ = R.getZ()*R.getZ();

		dumbbellBondXXHistogram.fill(bondXX);
		dumbbellBondYYHistogram.fill(bondYY);
		dumbbellBondZZHistogram.fill(bondZZ);

		dumbbellBondXYHistogram.fill(R.getX()*R.getY());

		averageBondXX += bondXX;
		averageBondYY += bondYY;
		averageBondZZ += bondZZ;

		const Vector3D<MPCParticlePositionType> R_xy(R.getX(), R.getY(), 0);
		dumbbellBondAngleWithFlowDirectionHistogram.fill(R.getAngle(flowDirection));
		dumbbellBondXYAngleWithFlowDirectionHistogram.fill(R_xy.getAngle(flowDirection));
	}

	const unsigned int dumbbellCount = mpcFluid->getParticleCount() / 2;
	averageBondXX /= dumbbellCount;
	averageBondYY /= dumbbellCount;
	averageBondZZ /= dumbbellCount;

	dumbbellAverageBondXXVSTime.addPoint(simulation->getMPCTime(), averageBondXX);
	dumbbellAverageBondYYVSTime.addPoint(simulation->getMPCTime(), averageBondYY);
	dumbbellAverageBondZZVSTime.addPoint(simulation->getMPCTime(), averageBondZZ);
}

void GaussianDumbbells::saveSpecific(const std::string& rundir) const
{
	dumbbellBondLengthHistogram.save(rundir+"/dumbbellBondLengthHistogram.data");
	dumbbellBondLengthSquaredHistogram.save(rundir+"/dumbbellBondLengthSquaredHistogram.data");

	dumbbellBondXHistogram.save(rundir+"/dumbbellBondXHistogram.data");
	dumbbellBondYHistogram.save(rundir+"/dumbbellBondYHistogram.data");
	dumbbellBondZHistogram.save(rundir+"/dumbbellBondZHistogram.data");

	dumbbellBondXXHistogram.save(rundir+"/dumbbellBondXXHistogram.data");
	dumbbellBondYYHistogram.save(rundir+"/dumbbellBondYYHistogram.data");
	dumbbellBondZZHistogram.save(rundir+"/dumbbellBondZZHistogram.data");

	dumbbellBondXYHistogram.save(rundir+"/dumbbellBondXYHistogram.data");

	dumbbellAverageBondXXVSTime.save(rundir+"/dumbbellAverageBondXXVSTime.data", false);
	dumbbellAverageBondYYVSTime.save(rundir+"/dumbbellAverageBondYYVSTime.data", false);
	dumbbellAverageBondZZVSTime.save(rundir+"/dumbbellAverageBondZZVSTime.data", false);

	dumbbellBondAngleWithFlowDirectionHistogram.save(rundir+"/dumbbellBondAngleWithFlowDirectionHistogram.data");
	dumbbellBondXYAngleWithFlowDirectionHistogram.save(rundir+"/dumbbellBondXYAngleWithFlowDirectionHistogram.data");
}
