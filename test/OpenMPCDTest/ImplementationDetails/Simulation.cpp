/**
 * @file
 * Tests functionality in `OpenMPCD/ImplementationDetails/Simulation.hpp`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/ImplementationDetails/Simulation.hpp>
#include <OpenMPCD/LeesEdwardsBoundaryConditions.hpp>
#include <OpenMPCD/RemotelyStoredVector.hpp>

#include <boost/scoped_array.hpp>

SCENARIO(
	"`OpenMPCD::ImplementationDetails::getCollisionCellIndex`",
	"")
{
	using namespace OpenMPCD;
	using namespace OpenMPCD::ImplementationDetails;

	static const unsigned int boxSizes[][3] =
		{ {1, 2, 3}, {4, 5, 6} };
	static const unsigned int maxFrac = 10;

	for(
		std::size_t boxSizeIndex = 0;
		boxSizeIndex < sizeof(boxSizes)/sizeof(boxSizes[0]);
		++boxSizeIndex)
	{
		const unsigned int (&boxSize)[3] = boxSizes[boxSizeIndex];

		for(unsigned int fracX = 0; fracX < maxFrac; ++fracX)
		{
			for(unsigned int fracY = 0; fracY < maxFrac; ++fracY)
			{
				for(unsigned int fracZ = 0; fracZ < maxFrac; ++fracZ)
				{
					const FP x = (1.0 * fracX) / maxFrac;
					const FP y = (1.0 * fracY) / maxFrac;
					const FP z = (1.0 * fracZ) / maxFrac;

					const unsigned int result =
						getCollisionCellIndex(
							Vector3D<MPCParticlePositionType>(x, y, z),
							boxSize[0], boxSize[1], boxSize[2]);

					const unsigned int expected =
						static_cast<unsigned int>(x) +
						static_cast<unsigned int>(y) * boxSize[0] +
						static_cast<unsigned int>(z) * boxSize[0] * boxSize[1];

					REQUIRE(result == expected);
				}
			}
		}
	}
}


SCENARIO(
	"`OpenMPCD::ImplementationDetails::sortIntoCollisionCellsLeesEdwards`",
	"")
{
	using namespace OpenMPCD;

	static const unsigned int boxSize[3] = {1, 2, 3};
	static const std::size_t particleCount = 123;
	static const FP shearRate = 0.123;
	static const MPCParticlePositionType gridShift[3] = {0.1, 0.2, 0.3};
	static const FP mpcTime = 12.34;

	RNG rng;
	boost::random::uniform_01<MPCParticlePositionType> posDist;
	boost::random::uniform_01<MPCParticleVelocityType> velDist;

	boost::scoped_array<MPCParticlePositionType> positions(
		new MPCParticlePositionType[3 * particleCount]);
	boost::scoped_array<MPCParticleVelocityType> velocities(
		new MPCParticleVelocityType[3 * particleCount]);

	for(std::size_t particle = 0; particle < particleCount; ++particle)
	{
		positions[particle * 3 + 0] = (posDist(rng) - 0.5) * boxSize[0] * 4;
		positions[particle * 3 + 1] = (posDist(rng) - 0.5) * boxSize[1] * 4;
		positions[particle * 3 + 2] = (posDist(rng) - 0.5) * boxSize[2] * 4;

		velocities[particle * 3 + 0] = (velDist(rng) - 0.5) * 10;
		velocities[particle * 3 + 1] = (velDist(rng) - 0.5) * 10;
		velocities[particle * 3 + 2] = (velDist(rng) - 0.5) * 10;
	}


	boost::scoped_array<MPCParticleVelocityType> velocityCorrections(
		new MPCParticleVelocityType[particleCount]);
	boost::scoped_array<unsigned int> collisionCellIndices(
		new unsigned int[particleCount]);

	const Vector3D<MPCParticlePositionType> gridShiftVector(
		gridShift[0], gridShift[1], gridShift[2]);
	for(std::size_t particle = 0; particle < particleCount; ++particle)
	{
		const RemotelyStoredVector<MPCParticlePositionType>
			position(positions.get(), particle);

		MPCParticleVelocityType expectedVelocityCorrection;
		const Vector3D<MPCParticlePositionType> image =
			OpenMPCD::getImageUnderLeesEdwardsBoundaryConditions(
				position + gridShiftVector,
				mpcTime,
				shearRate,
				boxSize[0], boxSize[1], boxSize[2],
				&expectedVelocityCorrection);

		const MPCParticleVelocityType oldVX = velocities[3 * particle + 0];
		const MPCParticleVelocityType oldVY = velocities[3 * particle + 1];
		const MPCParticleVelocityType oldVZ = velocities[3 * particle + 2];

		ImplementationDetails::sortIntoCollisionCellsLeesEdwards(
			particle,
			gridShift,
			shearRate,
			mpcTime,
			positions.get(),
			velocities.get(),
			velocityCorrections.get(),
			collisionCellIndices.get(),
			boxSize[0], boxSize[1], boxSize[2]);

		const unsigned int expectedCollisionCellIndex =
			OpenMPCD::ImplementationDetails::getCollisionCellIndex(
				image, boxSize[0], boxSize[1], boxSize[2]);

		const MPCParticleVelocityType velocityCorrection =
			velocityCorrections[particle];

		REQUIRE(velocityCorrection == expectedVelocityCorrection);
		REQUIRE(velocities[3 * particle + 0] == oldVX + velocityCorrection);
		REQUIRE(velocities[3 * particle + 1] == oldVY);
		REQUIRE(velocities[3 * particle + 2] == oldVZ);
		REQUIRE(collisionCellIndices[particle] == expectedCollisionCellIndex);
	}
}


SCENARIO(
	"`OpenMPCD::ImplementationDetails::collisionCellContributions`",
	"")
{
	using namespace OpenMPCD;

	static const unsigned int boxSize[3] = {1, 2, 3};
	static const std::size_t particleCount = 123;
	static const FP shearRate = 0.123;
	static const MPCParticlePositionType gridShift[3] = {0.1, 0.2, 0.3};
	static const FP mpcTime = 12.34;
	static const FP particleMass = 1.5;

	RNG rng;
	boost::random::uniform_01<MPCParticlePositionType> posDist;
	boost::random::uniform_01<MPCParticleVelocityType> velDist;

	boost::scoped_array<MPCParticlePositionType> positions(
		new MPCParticlePositionType[3 * particleCount]);
	boost::scoped_array<MPCParticleVelocityType> velocities(
		new MPCParticleVelocityType[3 * particleCount]);
	boost::scoped_array<MPCParticleVelocityType> velocityCorrections(
		new MPCParticleVelocityType[particleCount]);
	boost::scoped_array<unsigned int> collisionCellIndices(
		new unsigned int[particleCount]);

	for(std::size_t particle = 0; particle < particleCount; ++particle)
	{
		positions[particle * 3 + 0] = (posDist(rng) - 0.5) * boxSize[0] * 4;
		positions[particle * 3 + 1] = (posDist(rng) - 0.5) * boxSize[1] * 4;
		positions[particle * 3 + 2] = (posDist(rng) - 0.5) * boxSize[2] * 4;

		velocities[particle * 3 + 0] = (velDist(rng) - 0.5) * 10;
		velocities[particle * 3 + 1] = (velDist(rng) - 0.5) * 10;
		velocities[particle * 3 + 2] = (velDist(rng) - 0.5) * 10;
	}

	const Vector3D<MPCParticlePositionType> gridShiftVector(
		gridShift[0], gridShift[1], gridShift[2]);
	for(std::size_t particle = 0; particle < particleCount; ++particle)
	{
		ImplementationDetails::sortIntoCollisionCellsLeesEdwards(
			particle,
			gridShift,
			shearRate,
			mpcTime,
			positions.get(),
			velocities.get(),
			velocityCorrections.get(),
			collisionCellIndices.get(),
			boxSize[0], boxSize[1], boxSize[2]);
	}


	const std::size_t collisionCellCount =
		boxSize[0] * boxSize[1] * boxSize[2];

	boost::scoped_array<MPCParticleVelocityType> collisionCellMomenta(
		new MPCParticleVelocityType[3 * collisionCellCount]);
	boost::scoped_array<FP> collisionCellMasses(
		new FP[collisionCellCount]);

	for(std::size_t ccIndex = 0; ccIndex < collisionCellCount; ++ccIndex)
	{
		collisionCellMomenta[3 * ccIndex + 0] = 0;
		collisionCellMomenta[3 * ccIndex + 1] = 0;
		collisionCellMomenta[3 * ccIndex + 2] = 0;
		collisionCellMasses[ccIndex] = 0;
	}

	ImplementationDetails::collisionCellContributions(
		particleCount,
		velocities.get(),
		collisionCellIndices.get(),
		collisionCellMomenta.get(),
		collisionCellMasses.get(),
		particleMass);

	for(std::size_t ccIndex = 0; ccIndex < collisionCellCount; ++ccIndex)
	{
		Vector3D<MPCParticleVelocityType> momentum(0, 0, 0);
		FP mass = 0;

		for(unsigned int p = 0; p < particleCount; ++p)
		{
			if(collisionCellIndices[p] != ccIndex)
				continue;

			const RemotelyStoredVector<MPCParticleVelocityType> velocity(
				velocities.get(), p);

			momentum += velocity * particleMass;
			mass += particleMass;
		}

		REQUIRE(collisionCellMomenta[3 * ccIndex + 0] == momentum.getX());
		REQUIRE(collisionCellMomenta[3 * ccIndex + 1] == momentum.getY());
		REQUIRE(collisionCellMomenta[3 * ccIndex + 2] == momentum.getZ());
		REQUIRE(collisionCellMasses[ccIndex] == mass);
	}
}


SCENARIO(
	"`OpenMPCD::ImplementationDetails::getCollisionCellCenterOfMassMomentum`",
	"")
{
	using namespace OpenMPCD;

	static const unsigned int collisionCellCount = 10;

	RNG rng;
	boost::random::uniform_01<MPCParticleVelocityType> velDist;

	boost::scoped_array<MPCParticleVelocityType> momenta(
		new MPCParticleVelocityType[3 * collisionCellCount]);

	for(std::size_t cc = 0; cc < collisionCellCount; ++cc)
	{
		momenta[cc * 3 + 0] = (velDist(rng) - 0.5) * 500;
		momenta[cc * 3 + 1] = (velDist(rng) - 0.5) * 500;
		momenta[cc * 3 + 2] = (velDist(rng) - 0.5) * 500;
	}

	for(std::size_t cc = 0; cc < collisionCellCount; ++cc)
	{
		const Vector3D<MPCParticleVelocityType> momentum =
			ImplementationDetails::getCollisionCellCenterOfMassMomentum(
				cc, momenta.get());

		RemotelyStoredVector<MPCParticleVelocityType> expected(
			momenta.get(), cc);
		REQUIRE(momentum == expected);
	}
}


SCENARIO(
	"`OpenMPCD::ImplementationDetails::getCollisionCellCenterOfMassVelocity`",
	"")
{
	using namespace OpenMPCD;

	static const unsigned int collisionCellCount = 10;

	RNG rng;
	boost::random::uniform_01<MPCParticleVelocityType> velDist;
	boost::random::uniform_01<FP> massDist;

	boost::scoped_array<MPCParticleVelocityType> momenta(
		new MPCParticleVelocityType[3 * collisionCellCount]);
	boost::scoped_array<FP> masses(
		new FP[collisionCellCount]);

	for(std::size_t cc = 0; cc < collisionCellCount; ++cc)
	{
		momenta[cc * 3 + 0] = (velDist(rng) - 0.5) * 500;
		momenta[cc * 3 + 1] = (velDist(rng) - 0.5) * 500;
		momenta[cc * 3 + 2] = (velDist(rng) - 0.5) * 500;
		masses[cc] = massDist(rng) * 20;
	}

	for(std::size_t cc = 0; cc < collisionCellCount; ++cc)
	{
		const Vector3D<MPCParticleVelocityType> velocity =
			ImplementationDetails::getCollisionCellCenterOfMassVelocity(
				cc, momenta.get(), masses.get());

		RemotelyStoredVector<MPCParticleVelocityType> momentum(
			momenta.get(), cc);
		REQUIRE(velocity == momentum / masses[cc]);
	}
}


SCENARIO(
	"`OpenMPCD::ImplementationDetails::collisionCellStochasticRotationStep1`",
	"")
{
	using namespace OpenMPCD;

	static const unsigned int boxSize[3] = {1, 2, 3};
	static const std::size_t particleCount = 123;
	static const FP shearRate = 0.123;
	static const MPCParticlePositionType gridShift[3] = {0.1, 0.2, 0.3};
	static const FP mpcTime = 12.34;
	static const FP particleMass = 1.5;
	static const FP collisionAngle = 2;

	RNG rng;
	boost::random::uniform_01<MPCParticlePositionType> posDist;
	boost::random::uniform_01<MPCParticleVelocityType> velDist;

	boost::scoped_array<MPCParticlePositionType> positions(
		new MPCParticlePositionType[3 * particleCount]);
	boost::scoped_array<MPCParticleVelocityType> velocities(
		new MPCParticleVelocityType[3 * particleCount]);
	boost::scoped_array<MPCParticleVelocityType> velocityCorrections(
		new MPCParticleVelocityType[particleCount]);
	boost::scoped_array<unsigned int> collisionCellIndices(
		new unsigned int[particleCount]);

	for(std::size_t particle = 0; particle < particleCount; ++particle)
	{
		positions[particle * 3 + 0] = (posDist(rng) - 0.5) * boxSize[0] * 4;
		positions[particle * 3 + 1] = (posDist(rng) - 0.5) * boxSize[1] * 4;
		positions[particle * 3 + 2] = (posDist(rng) - 0.5) * boxSize[2] * 4;

		velocities[particle * 3 + 0] = (velDist(rng) - 0.5) * 10;
		velocities[particle * 3 + 1] = (velDist(rng) - 0.5) * 10;
		velocities[particle * 3 + 2] = (velDist(rng) - 0.5) * 10;
	}

	const Vector3D<MPCParticlePositionType> gridShiftVector(
		gridShift[0], gridShift[1], gridShift[2]);
	for(std::size_t particle = 0; particle < particleCount; ++particle)
	{
		ImplementationDetails::sortIntoCollisionCellsLeesEdwards(
			particle,
			gridShift,
			shearRate,
			mpcTime,
			positions.get(),
			velocities.get(),
			velocityCorrections.get(),
			collisionCellIndices.get(),
			boxSize[0], boxSize[1], boxSize[2]);
	}


	const std::size_t collisionCellCount =
		boxSize[0] * boxSize[1] * boxSize[2];

	boost::scoped_array<MPCParticleVelocityType> collisionCellMomenta(
		new MPCParticleVelocityType[3 * collisionCellCount]);
	boost::scoped_array<FP> collisionCellMasses(
		new FP[collisionCellCount]);

	for(std::size_t ccIndex = 0; ccIndex < collisionCellCount; ++ccIndex)
	{
		collisionCellMomenta[3 * ccIndex + 0] = 0;
		collisionCellMomenta[3 * ccIndex + 1] = 0;
		collisionCellMomenta[3 * ccIndex + 2] = 0;
		collisionCellMasses[ccIndex] = 0;
	}

	ImplementationDetails::collisionCellContributions(
		particleCount,
		velocities.get(),
		collisionCellIndices.get(),
		collisionCellMomenta.get(),
		collisionCellMasses.get(),
		particleMass);



	boost::scoped_array<MPCParticleVelocityType> oldVelocities(
		new MPCParticleVelocityType[3 * particleCount]);
	memcpy(
		oldVelocities.get(),
		velocities.get(),
		3 * particleCount * sizeof(oldVelocities[0]));

	boost::scoped_array<MPCParticlePositionType> collisionCellRotationAxes(
		new MPCParticlePositionType[3 * collisionCellCount]);
	boost::scoped_array<FP> collisionCellFrameInternalKineticEnergies(
		new FP[collisionCellCount]);
	boost::scoped_array<unsigned int> collisionCellParticleCounts(
		new unsigned int[collisionCellCount]);
	boost::scoped_array<FP> expectedFrameInternalKineticEnergies(
		new FP[collisionCellCount]);
	boost::scoped_array<unsigned int> expectedParticleCounts(
		new unsigned int[collisionCellCount]);

	for(std::size_t ccIndex = 0; ccIndex < collisionCellCount; ++ccIndex)
	{
		RemotelyStoredVector<MPCParticlePositionType> axis(
			collisionCellRotationAxes.get(), ccIndex);

		axis = Vector3D<MPCParticlePositionType>::getRandomUnitVector(rng);

		collisionCellFrameInternalKineticEnergies[ccIndex] = 0;
		collisionCellParticleCounts[ccIndex] = 0;
		expectedFrameInternalKineticEnergies[ccIndex] = 0;
		expectedParticleCounts[ccIndex] = 0;
	}

	ImplementationDetails::collisionCellStochasticRotationStep1(
		particleCount,
		velocities.get(),
		particleMass,
		collisionCellIndices.get(),
		collisionCellMomenta.get(),
		collisionCellMasses.get(),
		collisionCellRotationAxes.get(),
		collisionAngle,
		collisionCellFrameInternalKineticEnergies.get(),
		collisionCellParticleCounts.get());

	for(std::size_t particle = 0; particle < particleCount; ++particle)
	{
		const RemotelyStoredVector<MPCParticleVelocityType> oldVelocity(
			oldVelocities.get(), particle);

		const unsigned int ccIndex = collisionCellIndices[particle];

		const RemotelyStoredVector<MPCParticlePositionType> axis(
			collisionCellRotationAxes.get(), ccIndex);

		const Vector3D<MPCParticleVelocityType> ccVelocity =
			ImplementationDetails::getCollisionCellCenterOfMassVelocity(
				ccIndex, collisionCellMomenta.get(), collisionCellMasses.get());

		const Vector3D<MPCParticleVelocityType> oldRelativeVelocity =
			oldVelocity - ccVelocity;

		const Vector3D<MPCParticleVelocityType> newRelativeVelocity =
			oldRelativeVelocity.getRotatedAroundNormalizedAxis(
				axis, collisionAngle);

		REQUIRE(velocities[3 * particle + 0] == newRelativeVelocity.getX());
		REQUIRE(velocities[3 * particle + 1] == newRelativeVelocity.getY());
		REQUIRE(velocities[3 * particle + 2] == newRelativeVelocity.getZ());

		expectedFrameInternalKineticEnergies[ccIndex] +=
			particleMass * newRelativeVelocity.getMagnitudeSquared() / 2.0;
		expectedParticleCounts[ccIndex] += 1;
	}

	for(std::size_t ccIndex = 0; ccIndex < collisionCellCount; ++ccIndex)
	{
		REQUIRE(
			collisionCellFrameInternalKineticEnergies[ccIndex]
			==
			Approx(expectedFrameInternalKineticEnergies[ccIndex]));

		REQUIRE(
			collisionCellParticleCounts[ccIndex]
			==
			expectedParticleCounts[ccIndex]);
	}
}


SCENARIO(
	"`OpenMPCD::ImplementationDetails::collisionCellStochasticRotationStep2`",
	"")
{
	using namespace OpenMPCD;

	static const unsigned int boxSize[3] = {1, 2, 3};
	static const std::size_t particleCount = 123;
	static const FP shearRate = 0.123;
	static const MPCParticlePositionType gridShift[3] = {0.1, 0.2, 0.3};
	static const FP mpcTime = 12.34;
	static const FP particleMass = 1.5;
	static const FP collisionAngle = 2;

	RNG rng;
	boost::random::uniform_01<MPCParticlePositionType> posDist;
	boost::random::uniform_01<MPCParticleVelocityType> velDist;
	boost::random::uniform_01<FP> fpDist;

	boost::scoped_array<MPCParticlePositionType> positions(
		new MPCParticlePositionType[3 * particleCount]);
	boost::scoped_array<MPCParticleVelocityType> velocities(
		new MPCParticleVelocityType[3 * particleCount]);
	boost::scoped_array<MPCParticleVelocityType> velocityCorrections(
		new MPCParticleVelocityType[particleCount]);
	boost::scoped_array<unsigned int> collisionCellIndices(
		new unsigned int[particleCount]);

	for(std::size_t particle = 0; particle < particleCount; ++particle)
	{
		positions[particle * 3 + 0] = (posDist(rng) - 0.5) * boxSize[0] * 4;
		positions[particle * 3 + 1] = (posDist(rng) - 0.5) * boxSize[1] * 4;
		positions[particle * 3 + 2] = (posDist(rng) - 0.5) * boxSize[2] * 4;

		velocities[particle * 3 + 0] = (velDist(rng) - 0.5) * 10;
		velocities[particle * 3 + 1] = (velDist(rng) - 0.5) * 10;
		velocities[particle * 3 + 2] = (velDist(rng) - 0.5) * 10;
	}

	const Vector3D<MPCParticlePositionType> gridShiftVector(
		gridShift[0], gridShift[1], gridShift[2]);
	for(std::size_t particle = 0; particle < particleCount; ++particle)
	{
		ImplementationDetails::sortIntoCollisionCellsLeesEdwards(
			particle,
			gridShift,
			shearRate,
			mpcTime,
			positions.get(),
			velocities.get(),
			velocityCorrections.get(),
			collisionCellIndices.get(),
			boxSize[0], boxSize[1], boxSize[2]);
	}


	const std::size_t collisionCellCount =
		boxSize[0] * boxSize[1] * boxSize[2];

	boost::scoped_array<MPCParticleVelocityType> collisionCellMomenta(
		new MPCParticleVelocityType[3 * collisionCellCount]);
	boost::scoped_array<FP> collisionCellMasses(
		new FP[collisionCellCount]);

	for(std::size_t ccIndex = 0; ccIndex < collisionCellCount; ++ccIndex)
	{
		collisionCellMomenta[3 * ccIndex + 0] = 0;
		collisionCellMomenta[3 * ccIndex + 1] = 0;
		collisionCellMomenta[3 * ccIndex + 2] = 0;
		collisionCellMasses[ccIndex] = 0;
	}

	ImplementationDetails::collisionCellContributions(
		particleCount,
		velocities.get(),
		collisionCellIndices.get(),
		collisionCellMomenta.get(),
		collisionCellMasses.get(),
		particleMass);


	boost::scoped_array<MPCParticlePositionType> collisionCellRotationAxes(
		new MPCParticlePositionType[3 * collisionCellCount]);
	boost::scoped_array<FP> collisionCellSquaredRelativeMomenta(
		new FP[collisionCellCount]);
	boost::scoped_array<unsigned int> collisionCellParticleCounts(
		new unsigned int[collisionCellCount]);
	boost::scoped_array<FP> collisionCellVelocityScalings(
		new FP[collisionCellCount]);

	for(std::size_t ccIndex = 0; ccIndex < collisionCellCount; ++ccIndex)
	{
		RemotelyStoredVector<MPCParticlePositionType> axis(
			collisionCellRotationAxes.get(), ccIndex);

		axis = Vector3D<MPCParticlePositionType>::getRandomUnitVector(rng);

		collisionCellSquaredRelativeMomenta[ccIndex] = 0;
		collisionCellParticleCounts[ccIndex] = 0;
		collisionCellVelocityScalings[ccIndex] = (fpDist(rng) + 0.01) * 2;
	}

	ImplementationDetails::collisionCellStochasticRotationStep1(
		particleCount,
		velocities.get(),
		particleMass,
		collisionCellIndices.get(),
		collisionCellMomenta.get(),
		collisionCellMasses.get(),
		collisionCellRotationAxes.get(),
		collisionAngle,
		collisionCellSquaredRelativeMomenta.get(),
		collisionCellParticleCounts.get());

	boost::scoped_array<MPCParticleVelocityType> rotatedRelativeVelocities(
		new MPCParticleVelocityType[3 * particleCount]);
	memcpy(
		rotatedRelativeVelocities.get(),
		velocities.get(),
		3 * particleCount * sizeof(rotatedRelativeVelocities[0]));

	ImplementationDetails::collisionCellStochasticRotationStep2(
		particleCount,
		velocities.get(),
		collisionCellIndices.get(),
		collisionCellMomenta.get(),
		collisionCellMasses.get(),
		collisionCellVelocityScalings.get());

	for(std::size_t particle = 0; particle < particleCount; ++particle)
	{
		const unsigned int ccIndex = collisionCellIndices[particle];

		const RemotelyStoredVector<MPCParticleVelocityType>
			rotatedRelativeVelocity(
					rotatedRelativeVelocities.get(), particle);

		const Vector3D<MPCParticleVelocityType> scaledRelativeVelocity =
			rotatedRelativeVelocity * collisionCellVelocityScalings[ccIndex];

		const Vector3D<MPCParticleVelocityType> ccVelocity =
			ImplementationDetails::getCollisionCellCenterOfMassVelocity(
				ccIndex, collisionCellMomenta.get(), collisionCellMasses.get());

		const Vector3D<MPCParticleVelocityType> newVelocity =
			ccVelocity + scaledRelativeVelocity;

		REQUIRE(velocities[3 * particle + 0] == newVelocity.getX());
		REQUIRE(velocities[3 * particle + 1] == newVelocity.getY());
		REQUIRE(velocities[3 * particle + 2] == newVelocity.getZ());
	}
}


SCENARIO(
	"`OpenMPCD::ImplementationDetails::undoLeesEdwardsVelocityCorrections`",
	"")
{
	using namespace OpenMPCD;

	static const unsigned int boxSize[3] = {1, 2, 3};
	static const std::size_t particleCount = 123;
	static const FP shearRate = 0.123;
	static const MPCParticlePositionType gridShift[3] = {0.1, 0.2, 0.3};
	static const FP mpcTime = 12.34;

	RNG rng;
	boost::random::uniform_01<MPCParticlePositionType> posDist;
	boost::random::uniform_01<MPCParticleVelocityType> velDist;

	boost::scoped_array<MPCParticlePositionType> positions(
		new MPCParticlePositionType[3 * particleCount]);
	boost::scoped_array<MPCParticleVelocityType> velocities(
		new MPCParticleVelocityType[3 * particleCount]);

	for(std::size_t particle = 0; particle < particleCount; ++particle)
	{
		positions[particle * 3 + 0] = (posDist(rng) - 0.5) * boxSize[0] * 4;
		positions[particle * 3 + 1] = (posDist(rng) - 0.5) * boxSize[1] * 4;
		positions[particle * 3 + 2] = (posDist(rng) - 0.5) * boxSize[2] * 4;

		velocities[particle * 3 + 0] = (velDist(rng) - 0.5) * 10;
		velocities[particle * 3 + 1] = (velDist(rng) - 0.5) * 10;
		velocities[particle * 3 + 2] = (velDist(rng) - 0.5) * 10;
	}

	boost::scoped_array<MPCParticleVelocityType> oldVelocities(
		new MPCParticleVelocityType[3 * particleCount]);
	memcpy(
		oldVelocities.get(),
		velocities.get(),
		3 * particleCount * sizeof(oldVelocities[0]));

	boost::scoped_array<MPCParticleVelocityType> velocityCorrections(
		new MPCParticleVelocityType[particleCount]);
	boost::scoped_array<unsigned int> collisionCellIndices(
		new unsigned int[particleCount]);

	for(std::size_t particle = 0; particle < particleCount; ++particle)
	{
		ImplementationDetails::sortIntoCollisionCellsLeesEdwards(
			particle,
			gridShift,
			shearRate,
			mpcTime,
			positions.get(),
			velocities.get(),
			velocityCorrections.get(),
			collisionCellIndices.get(),
			boxSize[0], boxSize[1], boxSize[2]);
	}

	ImplementationDetails::undoLeesEdwardsVelocityCorrections(
		particleCount,
		velocities.get(),
		velocityCorrections.get());

	for(std::size_t coord = 0; coord < 3 * particleCount; ++coord)
	{
		REQUIRE(velocities[coord] == Approx(oldVelocities[coord]));
	}
}
