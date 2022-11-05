#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/HarmonicTrimers.hpp>

#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/FourierTransformedVelocity/Triplets.hpp>
#include <OpenMPCD/CUDA/MPCFluid/Instrumentation/VelocityAutocorrelation/Triplets.hpp>

using namespace OpenMPCD;
using namespace OpenMPCD::CUDA::MPCFluid::Instrumentation;

HarmonicTrimers::HarmonicTrimers(
	const Simulation* const sim, DeviceMemoryManager* const devMemMgr,
	const MPCFluid::HarmonicTrimers* const mpcFluid_)
		: Base(sim->getConfiguration(), mpcFluid_),
		  simulation(sim), mpcFluid(mpcFluid_),

		  bond1LengthHistogram("harmonicTrimers.bond1LengthHistogram", sim->getConfiguration()),
		  bond2LengthHistogram("harmonicTrimers.bond2LengthHistogram", sim->getConfiguration()),

		  bond1LengthSquaredHistogram("harmonicTrimers.bond1LengthSquaredHistogram", sim->getConfiguration()),
		  bond2LengthSquaredHistogram("harmonicTrimers.bond2LengthSquaredHistogram", sim->getConfiguration()),

		  bond1XXHistogram("harmonicTrimers.bond1XXHistogram", sim->getConfiguration()),
		  bond2XXHistogram("harmonicTrimers.bond2XXHistogram", sim->getConfiguration()),

		  bond1YYHistogram("harmonicTrimers.bond1YYHistogram", sim->getConfiguration()),
		  bond2YYHistogram("harmonicTrimers.bond2YYHistogram", sim->getConfiguration()),

		  bond1ZZHistogram("harmonicTrimers.bond1ZZHistogram", sim->getConfiguration()),
		  bond2ZZHistogram("harmonicTrimers.bond2ZZHistogram", sim->getConfiguration()),

		  bond1XYHistogram("harmonicTrimers.bond1XYHistogram", sim->getConfiguration()),
		  bond2XYHistogram("harmonicTrimers.bond2XYHistogram", sim->getConfiguration()),

		  bond1XYAngleWithFlowDirectionHistogram("harmonicTrimers.bond1XYAngleWithFlowDirectionHistogram", sim->getConfiguration()),
		  bond2XYAngleWithFlowDirectionHistogram("harmonicTrimers.bond2XYAngleWithFlowDirectionHistogram", sim->getConfiguration()),

		  trimerCenterOfMassesSnapshotTime(-1)
{
	sim->getConfiguration().read(
		"instrumentation.selfDiffusionCoefficient.measurementTime",
		&selfDiffusionCoefficientMeasurementTime);

	fourierTransformedVelocity = new FourierTransformedVelocity::Triplets(sim, devMemMgr, mpcFluid_);
	velocityAutocorrelation    = new VelocityAutocorrelation::Triplets(sim, devMemMgr, mpcFluid_);
}

void HarmonicTrimers::measureSpecific()
{
	static const Vector3D<MPCParticleVelocityType> flowDirection(1, 0, 0);

	for(unsigned int i=0; i<mpcFluid->getParticleCount(); i+=3)
	{
		const RemotelyStoredVector<const MPCParticlePositionType> r_1 = mpcFluid->getPosition(i+0);
		const RemotelyStoredVector<const MPCParticlePositionType> r_2 = mpcFluid->getPosition(i+1);
		const RemotelyStoredVector<const MPCParticlePositionType> r_3 = mpcFluid->getPosition(i+2);

		const Vector3D<MPCParticlePositionType> R_1 = r_2 - r_1;
		const Vector3D<MPCParticlePositionType> R_2 = r_3 - r_2;

		bond1LengthHistogram.fill(R_1.getMagnitude());
		bond2LengthHistogram.fill(R_2.getMagnitude());

		bond1LengthSquaredHistogram.fill(R_1.getMagnitudeSquared());
		bond2LengthSquaredHistogram.fill(R_2.getMagnitudeSquared());

		const FP xx1 = R_1.getX()*R_1.getX();
		const FP yy1 = R_1.getY()*R_1.getY();
		const FP zz1 = R_1.getZ()*R_1.getZ();
		const FP xy1 = R_1.getX()*R_1.getY();

		const FP xx2 = R_2.getX()*R_2.getX();
		const FP yy2 = R_2.getY()*R_2.getY();
		const FP zz2 = R_2.getZ()*R_2.getZ();
		const FP xy2 = R_1.getX()*R_2.getY();

		bond1XXHistogram.fill(xx1);
		bond2XXHistogram.fill(xx2);

		bond1YYHistogram.fill(yy1);
		bond2YYHistogram.fill(yy2);

		bond1ZZHistogram.fill(zz1);
		bond2ZZHistogram.fill(zz2);

		bond1XYHistogram.fill(xy1);
		bond2XYHistogram.fill(xy2);

		const Vector3D<MPCParticlePositionType> R_1_xy(R_1.getX(), R_1.getY(), 0);
		const Vector3D<MPCParticlePositionType> R_2_xy(R_2.getX(), R_2.getY(), 0);

		bond1XYAngleWithFlowDirectionHistogram.fill(R_1_xy.getAngle(flowDirection));
		bond2XYAngleWithFlowDirectionHistogram.fill(R_2_xy.getAngle(flowDirection));
	}


	const unsigned int trimerCount = getTrimerCount();

	if(trimerCenterOfMassesSnapshotTime < 0)
	{
		trimerCenterOfMassesSnapshot.reserve(trimerCount);

		for(unsigned int i=0; i<trimerCount; ++i)
			trimerCenterOfMassesSnapshot.push_back(getTrimerCenterOfMass(i));

		trimerCenterOfMassesSnapshotTime = simulation->getMPCTime();
	}

	const FP timeSinceLastCenterOfMassSnapshot = simulation->getMPCTime() - trimerCenterOfMassesSnapshotTime;
	if(timeSinceLastCenterOfMassSnapshot >= selfDiffusionCoefficientMeasurementTime)
	{
		FP accumulatedSquares = 0;
		for(unsigned int i=0; i<trimerCount; ++i)
		{
			const Vector3D<MPCParticlePositionType>& oldPosition = trimerCenterOfMassesSnapshot[i];
			const Vector3D<MPCParticlePositionType> newPosition = getTrimerCenterOfMass(i);

			const Vector3D<MPCParticlePositionType> difference = newPosition - oldPosition;

			accumulatedSquares += difference.getMagnitudeSquared();

			trimerCenterOfMassesSnapshot[i] = newPosition;
		}

		const FP selfDiffusionCoefficient =
			accumulatedSquares / (6 * timeSinceLastCenterOfMassSnapshot * trimerCount);

		selfDiffusionCoefficients.addPoint(trimerCenterOfMassesSnapshotTime, selfDiffusionCoefficient);

		trimerCenterOfMassesSnapshotTime = simulation->getMPCTime();
	}
}

void HarmonicTrimers::saveSpecific(const std::string& rundir) const
{
	bond1LengthHistogram.save(rundir+"/harmonicTrimers/bond1/lengthHistogram.data");
	bond2LengthHistogram.save(rundir+"/harmonicTrimers/bond2/lengthHistogram.data");

	bond1LengthSquaredHistogram.save(rundir+"/harmonicTrimers/bond1/lengthSquaredHistogram.data");
	bond2LengthSquaredHistogram.save(rundir+"/harmonicTrimers/bond2/lengthSquaredHistogram.data");

	bond1XXHistogram.save(rundir+"/harmonicTrimers/bond1/XXHistogram.data");
	bond2XXHistogram.save(rundir+"/harmonicTrimers/bond2/XXHistogram.data");

	bond1YYHistogram.save(rundir+"/harmonicTrimers/bond1/YYHistogram.data");
	bond2YYHistogram.save(rundir+"/harmonicTrimers/bond2/YYHistogram.data");

	bond1ZZHistogram.save(rundir+"/harmonicTrimers/bond1/ZZHistogram.data");
	bond2ZZHistogram.save(rundir+"/harmonicTrimers/bond2/ZZHistogram.data");

	bond1XYHistogram.save(rundir+"/harmonicTrimers/bond1/XYHistogram.data");
	bond2XYHistogram.save(rundir+"/harmonicTrimers/bond2/XYHistogram.data");

	bond1XYHistogram.save(rundir+"/harmonicTrimers/bond1/XYHistogram.data");
	bond2XYHistogram.save(rundir+"/harmonicTrimers/bond2/XYHistogram.data");

	bond1XYAngleWithFlowDirectionHistogram.save(rundir+"/harmonicTrimers/bond1/angleWithFlowDirectionHistogram.data");
	bond2XYAngleWithFlowDirectionHistogram.save(rundir+"/harmonicTrimers/bond2/angleWithFlowDirectionHistogram.data");

	selfDiffusionCoefficients.save(rundir+"/selfDiffusionCoefficient.data", false);
}

const Vector3D<MPCParticlePositionType> HarmonicTrimers::getTrimerCenterOfMass(const unsigned int trimerID) const
{
	#ifdef OPENMPCD_DEBUG
		if(trimerID >= getTrimerCount())
			OPENMPCD_THROW(OutOfBoundsException, "trimerID");
	#endif

	const unsigned int particle1ID = trimerID * 3;

	const RemotelyStoredVector<const MPCParticlePositionType> r_1 = mpcFluid->getPosition(particle1ID + 0);
	const RemotelyStoredVector<const MPCParticlePositionType> r_2 = mpcFluid->getPosition(particle1ID + 1);
	const RemotelyStoredVector<const MPCParticlePositionType> r_3 = mpcFluid->getPosition(particle1ID + 2);

	return (r_1 + r_2 + r_3) / 3.0;
}
