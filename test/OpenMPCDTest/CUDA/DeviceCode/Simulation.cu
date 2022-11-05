/**
 * @file
 * Tests functionality in `OpenMPCD/CUDA/DeviceCode/Simulation.hpp`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/DeviceCode/Simulation.hpp>

#include <OpenMPCD/CUDA/DeviceCode/LeesEdwardsBoundaryConditions.hpp>
#include <OpenMPCD/CUDA/DeviceCode/Symbols.hpp>
#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/Random/Distributions/Uniform0e1e.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/ImplementationDetails/Simulation.hpp>
#include <OpenMPCD/LeesEdwardsBoundaryConditions.hpp>
#include <OpenMPCD/RemotelyStoredVector.hpp>

#include <boost/random/uniform_01.hpp>
#include <boost/scoped_array.hpp>

using namespace OpenMPCD;
using namespace OpenMPCD::CUDA;
using namespace OpenMPCD::CUDA::DeviceCode;


static __global__ void test_isInPrimarySimulationVolume(
	const OpenMPCD::MPCParticlePositionType* const positions,
	bool* const results,
	const std::size_t count)
{
	using namespace OpenMPCD;
	using OpenMPCD::CUDA::DeviceCode::isInPrimarySimulationVolume;

	for(std::size_t i = 0; i < count; ++i)
	{
		const RemotelyStoredVector<const MPCParticlePositionType> position(
			positions, i);
		results[i] = isInPrimarySimulationVolume(position);
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::isInPrimarySimulationVolume`",
	"[CUDA]")
{
	static const unsigned int boxSize[3] = {3, 4, 5};
	static const std::size_t positionCount = 1000;

	setSimulationBoxSizeSymbols(boxSize[0], boxSize[1], boxSize[2]);

	DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	RNG rng;
	boost::random::uniform_01<MPCParticlePositionType> posDist;

	boost::scoped_array<MPCParticlePositionType> positions(
		new MPCParticlePositionType[3 * positionCount]);

	boost::scoped_array<bool> expected(new bool[positionCount]);

	for(std::size_t i = 0; i < positionCount; ++i)
	{
		positions[i * 3 + 0] = (posDist(rng) - 0.5) * boxSize[0] * 4;
		positions[i * 3 + 1] = (posDist(rng) - 0.5) * boxSize[1] * 4;
		positions[i * 3 + 2] = (posDist(rng) - 0.5) * boxSize[2] * 4;

		bool e = true;
		for(std::size_t coord = 0; coord < 3; ++coord)
		{
			if(positions[i * 3 + coord] < 0)
				e = false;
			if(positions[i * 3 + coord] >= boxSize[coord])
				e = false;
		}
		expected[i] = e;
	}

	MPCParticlePositionType* d_positions;
	dmm.allocateMemory(&d_positions, 3 * positionCount);
	dmm.copyElementsFromHostToDevice(positions.get(), d_positions,
			3 * positionCount);

	bool* d_results;
	dmm.allocateMemory(&d_results, positionCount);


	test_isInPrimarySimulationVolume<<<1, 1>>>(
		d_positions, d_results, positionCount);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			expected.get(), d_results, positionCount));
}


static void __global__ getCollisionCellIndex_test_kernel(
	const FP x, const FP y, const FP z,
	unsigned int* const ret)
{
	*ret = getCollisionCellIndex(Vector3D<MPCParticlePositionType>(x, y, z));
}
static void getCollisionCellIndex_test(
	const unsigned int (&boxSize)[3],
	const FP x, const FP y, const FP z)
{
	DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	setSimulationBoxSizeSymbols(boxSize[0], boxSize[1], boxSize[2]);

	unsigned int* ret;
	dmm.allocateMemory(&ret, 1);

	getCollisionCellIndex_test_kernel<<<1, 1>>>(x, y, z, ret);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	const unsigned int expected =
		OpenMPCD::ImplementationDetails::getCollisionCellIndex(
			Vector3D<MPCParticlePositionType>(x, y, z),
			boxSize[0], boxSize[1], boxSize[2]);

	REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(&expected, ret, 1));
}
SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::getCollisionCellIndex`",
	"[CUDA]")
{
	static const unsigned int boxSizes[][3] =
		{ {1, 2, 3}, {4, 5, 6} };
	static const unsigned int maxFrac = 10;

	for(
		std::size_t boxSizeIndex = 0;
		boxSizeIndex < sizeof(boxSizes)/sizeof(boxSizes[0]);
		++boxSizeIndex)
	{
		for(unsigned int fracX = 0; fracX < maxFrac; ++fracX)
		{
			for(unsigned int fracY = 0; fracY < maxFrac; ++fracY)
			{
				for(unsigned int fracZ = 0; fracZ < maxFrac; ++fracZ)
				{
					const FP x = (1.0 * fracX) / maxFrac;
					const FP y = (1.0 * fracY) / maxFrac;
					const FP z = (1.0 * fracZ) / maxFrac;

					getCollisionCellIndex_test(boxSizes[boxSizeIndex], x, y, z);
				}
			}
		}
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::sortParticlesIntoCollisionCellsLeesEdwards`",
	"[CUDA]")
{
	static const unsigned int boxSize[3] = {1, 2, 3};
	static const std::size_t particleCount = 123;
	static const FP shearRate = 0.123;
	static const MPCParticlePositionType gridShift[3] = {0.1, 0.2, 0.3};
	static const FP mpcTime = 12.34;

	setSimulationBoxSizeSymbols(boxSize[0], boxSize[1], boxSize[2]);
	setLeesEdwardsSymbols(shearRate, boxSize[1]);

	DeviceMemoryManager dmm;
	dmm.setAutofree(true);

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

	MPCParticlePositionType* d_gridShift;
	dmm.allocateMemory(&d_gridShift, 3);
	dmm.copyElementsFromHostToDevice(gridShift, d_gridShift, 3);

	MPCParticlePositionType* d_positions;
	dmm.allocateMemory(&d_positions, 3 * particleCount);
	dmm.copyElementsFromHostToDevice(positions.get(), d_positions,
			3 * particleCount);

	MPCParticleVelocityType* d_velocities;
	dmm.allocateMemory(&d_velocities, 3 * particleCount);
	dmm.copyElementsFromHostToDevice(
		velocities.get(), d_velocities, 3 * particleCount);

	MPCParticleVelocityType* d_velocityCorrections;
	dmm.allocateMemory(&d_velocityCorrections, particleCount);

	unsigned int* d_collisionCellIndices;
	dmm.allocateMemory(&d_collisionCellIndices, particleCount);


	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(particleCount)
		DeviceCode::sortParticlesIntoCollisionCellsLeesEdwards <<<gridSize, blockSize>>> (
			workUnitOffset,
			particleCount,
			d_gridShift,
			mpcTime,
			d_positions,
			d_velocities,
			d_velocityCorrections,
			d_collisionCellIndices);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	boost::scoped_array<MPCParticleVelocityType> expectedVelocityCorrections(
		new MPCParticleVelocityType[particleCount]);
	boost::scoped_array<unsigned int> expectedCollisionCellIndices(
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
			expectedVelocityCorrections.get(),
			expectedCollisionCellIndices.get(),
			boxSize[0], boxSize[1], boxSize[2]);
	}

	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			velocities.get(), d_velocities, 3 * particleCount));
	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			expectedVelocityCorrections.get(), d_velocityCorrections,
			particleCount));
	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			expectedCollisionCellIndices.get(), d_collisionCellIndices,
			particleCount));
}


static __global__ void sortIntoCollisionCellsLeesEdwards_test_kernel(
	const unsigned int particleID,
	const MPCParticlePositionType* const gridShift_,
	const FP mpcTime,
	const MPCParticlePositionType* const positions,
	MPCParticleVelocityType* const velocities,
	MPCParticleVelocityType* const velocityCorrections,
	unsigned int* const collisionCellIndices)
{
	sortIntoCollisionCellsLeesEdwards(
		particleID,
		gridShift_,
		mpcTime,
		positions,
		velocities,
		velocityCorrections,
		collisionCellIndices);
}
SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::sortIntoCollisionCellsLeesEdwards`",
	"[CUDA]")
{
	static const unsigned int boxSize[3] = {1, 2, 3};
	static const std::size_t particleCount = 123;
	static const FP shearRate = 0.123;
	static const MPCParticlePositionType gridShift[3] = {0.1, 0.2, 0.3};
	static const FP mpcTime = 12.34;

	setSimulationBoxSizeSymbols(boxSize[0], boxSize[1], boxSize[2]);
	setLeesEdwardsSymbols(shearRate, boxSize[1]);

	DeviceMemoryManager dmm;
	dmm.setAutofree(true);

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

	MPCParticlePositionType* d_gridShift;
	dmm.allocateMemory(&d_gridShift, 3);
	dmm.copyElementsFromHostToDevice(gridShift, d_gridShift, 3);

	MPCParticlePositionType* d_positions;
	dmm.allocateMemory(&d_positions, 3 * particleCount);
	dmm.copyElementsFromHostToDevice(positions.get(), d_positions,
			3 * particleCount);

	MPCParticleVelocityType* d_velocities;
	dmm.allocateMemory(&d_velocities, 3 * particleCount);
	dmm.copyElementsFromHostToDevice(
		velocities.get(), d_velocities, 3 * particleCount);

	MPCParticleVelocityType* d_velocityCorrections;
	dmm.allocateMemory(&d_velocityCorrections, particleCount);

	unsigned int* d_collisionCellIndices;
	dmm.allocateMemory(&d_collisionCellIndices, particleCount);


	boost::scoped_array<MPCParticleVelocityType> expectedVelocityCorrections(
		new MPCParticleVelocityType[particleCount]);
	boost::scoped_array<unsigned int> expectedCollisionCellIndices(
		new unsigned int[particleCount]);

	for(std::size_t particle = 0; particle < particleCount; ++particle)
	{
		sortIntoCollisionCellsLeesEdwards_test_kernel<<<1, 1>>>(
			particle,
			d_gridShift,
			mpcTime,
			d_positions,
			d_velocities,
			d_velocityCorrections,
			d_collisionCellIndices);

		ImplementationDetails::sortIntoCollisionCellsLeesEdwards(
			particle,
			gridShift,
			shearRate,
			mpcTime,
			positions.get(),
			velocities.get(),
			expectedVelocityCorrections.get(),
			expectedCollisionCellIndices.get(),
			boxSize[0], boxSize[1], boxSize[2]);
	}
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			velocities.get(), d_velocities, 3 * particleCount));
	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			expectedVelocityCorrections.get(), d_velocityCorrections,
			particleCount));
	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			expectedCollisionCellIndices.get(), d_collisionCellIndices,
			particleCount));
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::collisionCellContributions`",
	"[CUDA]")
{
	static const unsigned int boxSize[3] = {1, 2, 3};
	static const std::size_t particleCount = 123;
	static const FP shearRate = 0.123;
	static const MPCParticlePositionType gridShift[3] = {0.1, 0.2, 0.3};
	static const FP mpcTime = 12.34;
	static const FP particleMass = 1.5;

	setSimulationBoxSizeSymbols(boxSize[0], boxSize[1], boxSize[2]);
	setLeesEdwardsSymbols(shearRate, boxSize[1]);

	const unsigned int collisionCellCount =
		boxSize[0] * boxSize[1] * boxSize[2];

	DeviceMemoryManager dmm;
	dmm.setAutofree(true);

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

	MPCParticlePositionType* d_gridShift;
	dmm.allocateMemory(&d_gridShift, 3);
	dmm.copyElementsFromHostToDevice(gridShift, d_gridShift, 3);

	MPCParticlePositionType* d_positions;
	dmm.allocateMemory(&d_positions, 3 * particleCount);
	dmm.copyElementsFromHostToDevice(positions.get(), d_positions,
			3 * particleCount);

	MPCParticleVelocityType* d_velocities;
	dmm.allocateMemory(&d_velocities, 3 * particleCount);
	dmm.copyElementsFromHostToDevice(
		velocities.get(), d_velocities, 3 * particleCount);

	MPCParticleVelocityType* d_velocityCorrections;
	dmm.allocateMemory(&d_velocityCorrections, particleCount);

	unsigned int* d_collisionCellIndices;
	dmm.allocateMemory(&d_collisionCellIndices, particleCount);

	MPCParticleVelocityType* d_collisionCellMomenta;
	dmm.allocateMemory(&d_collisionCellMomenta, 3 * collisionCellCount);
	dmm.zeroMemory(d_collisionCellMomenta, 3 * collisionCellCount);

	FP* d_collisionCellMasses;
	dmm.allocateMemory(&d_collisionCellMasses, collisionCellCount);
	dmm.zeroMemory(d_collisionCellMasses, collisionCellCount);


	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(particleCount)
		DeviceCode::sortParticlesIntoCollisionCellsLeesEdwards <<<gridSize, blockSize>>> (
			workUnitOffset,
			particleCount,
			d_gridShift,
			mpcTime,
			d_positions,
			d_velocities,
			d_velocityCorrections,
			d_collisionCellIndices);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(particleCount)
		DeviceCode::collisionCellContributions <<<gridSize, blockSize>>> (
			workUnitOffset,
			particleCount,
			d_velocities,
			d_collisionCellIndices,
			d_collisionCellMomenta,
			d_collisionCellMasses,
			particleMass);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;



	boost::scoped_array<MPCParticleVelocityType> velocityCorrections(
		new MPCParticleVelocityType[3 * particleCount]);
	boost::scoped_array<unsigned int> collisionCellIndices(
		new unsigned int[particleCount]);

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

	boost::scoped_array<MPCParticleVelocityType>
		collisionCellMomenta_fromDevice(
				new MPCParticleVelocityType[3 * collisionCellCount]);
	dmm.copyElementsFromDeviceToHost(
		d_collisionCellMomenta,
		collisionCellMomenta_fromDevice.get(),
		3 * collisionCellCount);
	for(std::size_t coord = 0; coord < 3 * collisionCellCount; ++coord)
	{
		REQUIRE(
			collisionCellMomenta_fromDevice[coord]
			==
			Approx(collisionCellMomenta[coord]));
	}

	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			collisionCellMasses.get(),
			d_collisionCellMasses,
			collisionCellCount));
}


static __global__ void getCollisionCellCenterOfMassVelocity_test_kernel(
	const unsigned int collisionCellIndex,
	const MPCParticleVelocityType* const collisionCellMomenta,
	const FP* const collisionCellMasses,
	MPCParticleVelocityType* const ret)
{
	const Vector3D<MPCParticleVelocityType> result =
		getCollisionCellCenterOfMassVelocity(
			collisionCellIndex,
			collisionCellMomenta,
			collisionCellMasses);

	ret[0] = result.getX();
	ret[1] = result.getY();
	ret[2] = result.getZ();
}
SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::getCollisionCellCenterOfMassVelocity`",
	"[CUDA]")
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

	DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	MPCParticleVelocityType* d_momenta;
	dmm.allocateMemory(&d_momenta, 3 * collisionCellCount);
	dmm.copyElementsFromHostToDevice(
		momenta.get(), d_momenta, 3 * collisionCellCount);

	FP* d_masses;
	dmm.allocateMemory(&d_masses, collisionCellCount);
	dmm.copyElementsFromHostToDevice(
		masses.get(), d_masses, collisionCellCount);

	MPCParticleVelocityType* d_result;
	dmm.allocateMemory(&d_result, 3);
	for(std::size_t cc = 0; cc < collisionCellCount; ++cc)
	{
		const Vector3D<MPCParticleVelocityType> velocity =
			ImplementationDetails::getCollisionCellCenterOfMassVelocity(
				cc, momenta.get(), masses.get());

		getCollisionCellCenterOfMassVelocity_test_kernel<<<1, 1>>>(
			cc,
			d_momenta,
			d_masses,
			d_result);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		const MPCParticleVelocityType vel[3] =
			{velocity.getX(), velocity.getY(), velocity.getZ()};
		REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(vel, d_result, 3));
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::resetCollisionCellData`",
	"[CUDA]")
{
	static const unsigned int collisionCellCount = 1000;

	DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	unsigned int collisionCellParticleCounts[collisionCellCount];
	MPCParticleVelocityType collisionCellMomenta[collisionCellCount];
	FP collisionCellFrameInternalKineticEnergies[collisionCellCount];
	FP collisionCellMasses[collisionCellCount];

	memset(
		collisionCellParticleCounts,
		0xFF,
		sizeof(collisionCellParticleCounts));
	memset(
		collisionCellMomenta,
		0xFF,
		sizeof(collisionCellMomenta));
	memset(
		collisionCellFrameInternalKineticEnergies,
		0xFF,
		sizeof(collisionCellFrameInternalKineticEnergies));
	memset(
		collisionCellMasses,
		0xFF,
		sizeof(collisionCellMasses));


	unsigned int* d_collisionCellParticleCounts;
	MPCParticleVelocityType* d_collisionCellMomenta;
	FP* d_collisionCellFrameInternalKineticEnergies;
	FP* d_collisionCellMasses;

	dmm.allocateMemory(&d_collisionCellParticleCounts, collisionCellCount);
	dmm.allocateMemory(&d_collisionCellMomenta, 3 * collisionCellCount);
	dmm.allocateMemory(
		&d_collisionCellFrameInternalKineticEnergies, collisionCellCount);
	dmm.allocateMemory(&d_collisionCellMasses, collisionCellCount);


	dmm.copyElementsFromHostToDevice(
		collisionCellParticleCounts,
		d_collisionCellParticleCounts,
		collisionCellCount);
	dmm.copyElementsFromHostToDevice(
		collisionCellMomenta,
		d_collisionCellMomenta,
		3 * collisionCellCount);
	dmm.copyElementsFromHostToDevice(
		collisionCellFrameInternalKineticEnergies,
		d_collisionCellFrameInternalKineticEnergies,
		collisionCellCount);
	dmm.copyElementsFromHostToDevice(
		collisionCellMasses,
		d_collisionCellMasses,
		collisionCellCount);

	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	memset(
		collisionCellParticleCounts,
		0,
		sizeof(collisionCellParticleCounts));
	memset(
		collisionCellMomenta,
		0,
		sizeof(collisionCellMomenta));
	memset(
		collisionCellFrameInternalKineticEnergies,
		0,
		sizeof(collisionCellFrameInternalKineticEnergies));
	memset(
		collisionCellMasses,
		0,
		sizeof(collisionCellMasses));


	REQUIRE_FALSE(
		dmm.elementMemoryEqualOnHostAndDevice(
			collisionCellParticleCounts,
			d_collisionCellParticleCounts,
			collisionCellCount));
	REQUIRE_FALSE(
		dmm.elementMemoryEqualOnHostAndDevice(
			collisionCellMomenta,
			d_collisionCellMomenta,
			3 * collisionCellCount));
	REQUIRE_FALSE(
		dmm.elementMemoryEqualOnHostAndDevice(
			collisionCellFrameInternalKineticEnergies,
			d_collisionCellFrameInternalKineticEnergies,
			collisionCellCount));
	REQUIRE_FALSE(
		dmm.elementMemoryEqualOnHostAndDevice(
			collisionCellMasses,
			d_collisionCellMasses,
			collisionCellCount));

	OPENMPCD_CUDA_LAUNCH_WORKUNITS_SIZES_BEGIN(collisionCellCount, 32, 16)
		DeviceCode::resetCollisionCellData <<<gridSize, blockSize>>> (
			workUnitOffset,
			collisionCellCount,
			d_collisionCellParticleCounts,
			d_collisionCellMomenta,
			d_collisionCellFrameInternalKineticEnergies,
			d_collisionCellMasses);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			collisionCellParticleCounts,
			d_collisionCellParticleCounts,
			collisionCellCount));
	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			collisionCellMomenta,
			d_collisionCellMomenta,
			3 * collisionCellCount));
	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			collisionCellFrameInternalKineticEnergies,
			d_collisionCellFrameInternalKineticEnergies,
			collisionCellCount));
	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			collisionCellMasses,
			d_collisionCellMasses,
			collisionCellCount));
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::generateGridShiftVector`",
	"[CUDA]")
{
	static const unsigned long long seed = 31415;
	static const FP gridShiftScale = 0.1234;

	static const MPCParticlePositionType expected[] =
	{
		0.11189421862636227122944632128564990125596523284912109375,
		0.1011516357910586527690810498825157992541790008544921875,
		0.0437378121651010587367380821888218633830547332763671875
	};

	DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	GPURNG* rngs;
	dmm.allocateMemory(&rngs, 1);

	DeviceCode::constructGPURNGs <<<1, 1>>> (
		1, rngs, seed);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	MPCParticlePositionType* output;
	dmm.allocateMemory(&output, 3);

	DeviceCode::generateGridShiftVector <<<1, 1>>> (
		output, gridShiftScale, rngs);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	DeviceCode::destroyGPURNGs <<<1, 1>>> (
		1,
		rngs);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(expected, output, 3));
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::generateCollisionCellRotationAxes`",
	"[CUDA]")
{
	static const unsigned long long seed = 31415;
	static const unsigned int collisionCellCount = 10;

	static const FP expected[] =
	{
		-0.56404642679802063920391219653538428246974945068359375,
		-0.52249985302665258313226104291970841586589813232421875,
		0.639410300201296077915458226925693452358245849609375,
		0.744461163902356393151649172068573534488677978515625,
		-0.260658000598798622604590491391718387603759765625,
		-0.61468283054351335881193563182023353874683380126953125,
		0.64925262155891960702547294204123318195343017578125,
		0.1516934079264512791329622132252552546560764312744140625,
		0.745291985325569950049384715384803712368011474609375,
		-0.315647416390887947112986466891015879809856414794921875,
		-0.86983005293616499908893047177116386592388153076171875,
		-0.37916010804780853060691470091114751994609832763671875,
		0.410400333071300027487637862577685154974460601806640625,
		0.1999856767283017255909527420953963883221149444580078125,
		-0.88970629744792140147779946346418000757694244384765625,
		0.1192954626234116466410029033795581199228763580322265625,
		0.1555693794626804571379352637450210750102996826171875,
		0.980595105418675228037272972869686782360076904296875,
		-0.455578574728317420294843032024800777435302734375,
		-0.6214716399719801831480481268954463303089141845703125,
		0.63735481716156527909333817660808563232421875,
		0.799245270598502255410267025581561028957366943359375,
		0.401238957044814215091577125349431298673152923583984375,
		0.4474531224335311918594015878625214099884033203125,
		-0.3247205378745252613015281895059160888195037841796875,
		0.532986303095885460834324476309120655059814453125,
		-0.78133358624511972667647796697565354406833648681640625,
		0.353380017246333333336139048697077669203281402587890625,
		0.70591096468238501149272678958368487656116485595703125,
		-0.61385036723306252159915175070636905729770660400390625
	};

	DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	GPURNG* rngs;
	dmm.allocateMemory(&rngs, collisionCellCount);

	DeviceCode::constructGPURNGs <<<1, 1>>> (
		collisionCellCount, rngs, seed);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	FP* output;
	dmm.allocateMemory(&output, 3 * collisionCellCount);

	OPENMPCD_CUDA_LAUNCH_WORKUNITS_SIZES_BEGIN(collisionCellCount, 2, 2)
		DeviceCode::generateCollisionCellRotationAxes <<<gridSize, blockSize>>> (
			workUnitOffset,
			collisionCellCount,
			output,
			rngs);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	DeviceCode::destroyGPURNGs <<<1, 1>>> (
		collisionCellCount,
		rngs);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			expected, output, 3 * collisionCellCount));

	for(unsigned int i = 0; i < collisionCellCount; ++i)
	{
		const OpenMPCD::RemotelyStoredVector<const FP> v(expected, i);
		REQUIRE(v.getMagnitudeSquared() == Approx(1));
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::collisionCellStochasticRotationStep1`",
	"[CUDA]")
{
	static const unsigned int boxSize[3] = {1, 2, 3};
	static const std::size_t particleCount = 123;
	static const FP shearRate = 0.123;
	static const MPCParticlePositionType gridShift[3] = {0.1, 0.2, 0.3};
	static const FP mpcTime = 12.34;
	static const FP particleMass = 1.5;
	static const FP collisionAngle = 2;

	setSimulationBoxSizeSymbols(boxSize[0], boxSize[1], boxSize[2]);
	setLeesEdwardsSymbols(shearRate, boxSize[1]);
	setSRDCollisionAngleSymbol(collisionAngle);

	const unsigned int collisionCellCount =
		boxSize[0] * boxSize[1] * boxSize[2];

	DeviceMemoryManager dmm;
	dmm.setAutofree(true);

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

	boost::scoped_array<MPCParticlePositionType> collisionCellRotationAxes(
		new MPCParticlePositionType[3 * collisionCellCount]);
	boost::scoped_array<FP> collisionCellFrameInternalKineticEnergies(
		new FP[collisionCellCount]);
	boost::scoped_array<unsigned int> collisionCellParticleCounts(
		new unsigned int[collisionCellCount]);

	for(std::size_t ccIndex = 0; ccIndex < collisionCellCount; ++ccIndex)
	{
		RemotelyStoredVector<MPCParticlePositionType> axis(
			collisionCellRotationAxes.get(), ccIndex);

		axis = Vector3D<MPCParticlePositionType>::getRandomUnitVector(rng);

		collisionCellFrameInternalKineticEnergies[ccIndex] = 0;
		collisionCellParticleCounts[ccIndex] = 0;
	}


	MPCParticlePositionType* d_gridShift;
	dmm.allocateMemory(&d_gridShift, 3);
	dmm.copyElementsFromHostToDevice(gridShift, d_gridShift, 3);

	MPCParticlePositionType* d_positions;
	dmm.allocateMemory(&d_positions, 3 * particleCount);
	dmm.copyElementsFromHostToDevice(positions.get(), d_positions,
			3 * particleCount);

	MPCParticleVelocityType* d_velocities;
	dmm.allocateMemory(&d_velocities, 3 * particleCount);
	dmm.copyElementsFromHostToDevice(
		velocities.get(), d_velocities, 3 * particleCount);

	MPCParticleVelocityType* d_velocityCorrections;
	dmm.allocateMemory(&d_velocityCorrections, particleCount);

	unsigned int* d_collisionCellIndices;
	dmm.allocateMemory(&d_collisionCellIndices, particleCount);

	MPCParticleVelocityType* d_collisionCellMomenta;
	dmm.allocateMemory(&d_collisionCellMomenta, 3 * collisionCellCount);
	dmm.zeroMemory(d_collisionCellMomenta, 3 * collisionCellCount);

	FP* d_collisionCellMasses;
	dmm.allocateMemory(&d_collisionCellMasses, collisionCellCount);
	dmm.zeroMemory(d_collisionCellMasses, collisionCellCount);


	MPCParticlePositionType* d_collisionCellRotationAxes;
	dmm.allocateMemory(&d_collisionCellRotationAxes, 3 * collisionCellCount);
	dmm.copyElementsFromHostToDevice(
		collisionCellRotationAxes.get(),
		d_collisionCellRotationAxes,
		3 * collisionCellCount);

	FP* d_collisionCellFrameInternalKineticEnergies;
	dmm.allocateMemory(
		&d_collisionCellFrameInternalKineticEnergies, collisionCellCount);
	dmm.zeroMemory(
		d_collisionCellFrameInternalKineticEnergies, collisionCellCount);

	unsigned int* d_collisionCellParticleCounts;
	dmm.allocateMemory(&d_collisionCellParticleCounts, collisionCellCount);
	dmm.zeroMemory(d_collisionCellParticleCounts, collisionCellCount);


	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(particleCount)
		DeviceCode::sortParticlesIntoCollisionCellsLeesEdwards <<<gridSize, blockSize>>> (
			workUnitOffset,
			particleCount,
			d_gridShift,
			mpcTime,
			d_positions,
			d_velocities,
			d_velocityCorrections,
			d_collisionCellIndices);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(particleCount)
		DeviceCode::collisionCellContributions <<<gridSize, blockSize>>> (
			workUnitOffset,
			particleCount,
			d_velocities,
			d_collisionCellIndices,
			d_collisionCellMomenta,
			d_collisionCellMasses,
			particleMass);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(particleCount)
		DeviceCode::collisionCellStochasticRotationStep1 <<<gridSize, blockSize>>> (
			workUnitOffset,
			particleCount,
			d_velocities,
			particleMass,
			d_collisionCellIndices,
			d_collisionCellMomenta,
			d_collisionCellMasses,
			d_collisionCellRotationAxes,
			d_collisionCellFrameInternalKineticEnergies,
			d_collisionCellParticleCounts);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;



	boost::scoped_array<MPCParticleVelocityType> velocityCorrections(
		new MPCParticleVelocityType[3 * particleCount]);
	boost::scoped_array<unsigned int> collisionCellIndices(
		new unsigned int[particleCount]);

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

	boost::scoped_array<MPCParticleVelocityType> velocities_fromDevice(
		new MPCParticleVelocityType[3 * particleCount]);
	boost::scoped_array<FP>
		collisionCellFrameInternalKineticEnergies_fromDevice(
				new FP[collisionCellCount]);
	boost::scoped_array<unsigned int> collisionCellParticleCounts_fromDevice(
		new unsigned int[collisionCellCount]);

	dmm.copyElementsFromDeviceToHost(
		d_velocities,
		velocities_fromDevice.get(),
		3 * particleCount);
	dmm.copyElementsFromDeviceToHost(
		d_collisionCellFrameInternalKineticEnergies,
		collisionCellFrameInternalKineticEnergies_fromDevice.get(),
		collisionCellCount);
	dmm.copyElementsFromDeviceToHost(
		d_collisionCellParticleCounts,
		collisionCellParticleCounts_fromDevice.get(),
		collisionCellCount);

	for(std::size_t coord = 0; coord < 3 * particleCount; ++coord)
	{
		REQUIRE(velocities_fromDevice[coord] == Approx(velocities[coord]));
	}

	for(std::size_t cc = 0; cc < collisionCellCount; ++cc)
	{
		REQUIRE(
			collisionCellFrameInternalKineticEnergies_fromDevice[cc]
			==
			Approx(collisionCellFrameInternalKineticEnergies[cc]));

		REQUIRE(
			collisionCellParticleCounts_fromDevice[cc]
			==
			collisionCellParticleCounts[cc]);
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::generateCollisionCellMBSFactors`",
	"[CUDA]")
{
	static const unsigned long long seed = 31415;
	static const unsigned int collisionCellCount = 10;

	static const FP expected[] =
	{
		1,
		1,
		0.5667240800742143758128577246679924428462982177734375,
		0.471466605899246438138305848042364232242107391357421875,
		1.23298728900925258500365089275874197483062744140625,
		1.1360214077185515346712918471894226968288421630859375,
		1.1093786762055477534971714703715406358242034912109375,
		1.4027808063605096133841243499773554503917694091796875,
		1.81889628545309367524396293447352945804595947265625,
		1.8435364744197209052600783252273686230182647705078125
	};

	static const FP kineticEnergies[] =
	{
		10, 10.5,
		11, 11.5,
		12, 12.5,
		13, 13.5,
		14, 14.5
	};

	static const unsigned int particleCounts[] =
	{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9
	};

	static const FP bulkThermostatTargetkT = 2.71828;

	DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	GPURNG* rngs;
	dmm.allocateMemory(&rngs, collisionCellCount);

	DeviceCode::constructGPURNGs <<<1, 1>>> (
		collisionCellCount, rngs, seed);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	FP* output;
	dmm.allocateMemory(&output, collisionCellCount);

	FP* d_kineticEnergies;
	dmm.allocateMemory(&d_kineticEnergies, collisionCellCount);
	dmm.copyElementsFromHostToDevice(
		kineticEnergies, d_kineticEnergies, collisionCellCount);

	unsigned int* d_particleCounts;
	dmm.allocateMemory(&d_particleCounts, collisionCellCount);
	dmm.copyElementsFromHostToDevice(
		particleCounts, d_particleCounts, collisionCellCount);

	OPENMPCD_CUDA_LAUNCH_WORKUNITS_SIZES_BEGIN(collisionCellCount, 2, 2)
		DeviceCode::generateCollisionCellMBSFactors <<<gridSize, blockSize>>> (
			workUnitOffset,
			collisionCellCount,
			d_kineticEnergies,
			d_particleCounts,
			output,
			bulkThermostatTargetkT,
			rngs);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	DeviceCode::destroyGPURNGs <<<1, 1>>> (
		collisionCellCount,
		rngs);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			expected, output, collisionCellCount));
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::collisionCellStochasticRotationStep2`",
	"[CUDA]")
{
	static const unsigned int collisionCellCount = 10;

	static const unsigned int particleCounts[] =
	{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9
	};

	static const FP mass = 1;

	REQUIRE(
		sizeof(particleCounts) / sizeof(particleCounts[0])
		==
		collisionCellCount);


	std::size_t particleCount = 0;
	for(std::size_t i = 0; i < collisionCellCount; ++i)
		particleCount += particleCounts[i];



	DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	RNG rng;
	boost::random::uniform_01<MPCParticleVelocityType> velDist;
	boost::random::uniform_01<FP> scalingDist;

	boost::scoped_array<unsigned int> collisionCellIndices(
		new unsigned int[particleCount]);

	boost::scoped_array<MPCParticleVelocityType> velocities(
		new MPCParticleVelocityType[3 * particleCount]);

	boost::scoped_array<MPCParticleVelocityType> ccMomenta(
		new MPCParticleVelocityType[3 * collisionCellCount]);

	boost::scoped_array<FP> ccMasses(new FP[collisionCellCount]);
	boost::scoped_array<FP> ccScalings(new FP[collisionCellCount]);

	std::size_t particle = 0;
	for(std::size_t cc = 0; cc < collisionCellCount; ++cc)
	{
		ccMomenta[cc * 3 + 0] = (velDist(rng) - 0.5) * 10;
		ccMomenta[cc * 3 + 1] = (velDist(rng) - 0.5) * 10;
		ccMomenta[cc * 3 + 2] = (velDist(rng) - 0.5) * 10;

		ccMasses[cc] = 0;

		for(std::size_t p = 0; p < particleCounts[cc]; ++p)
		{
			collisionCellIndices[particle] = cc;

			velocities[particle * 3 + 0] = (velDist(rng) - 0.5) * 10;
			velocities[particle * 3 + 1] = (velDist(rng) - 0.5) * 10;
			velocities[particle * 3 + 2] = (velDist(rng) - 0.5) * 10;

			ccMasses[cc] += mass;

			++particle;
		}

		ccScalings[cc] = scalingDist(rng) * 2;
	}



	MPCParticleVelocityType* d_velocities;
	dmm.allocateMemory(&d_velocities, 3 * particleCount);
	dmm.copyElementsFromHostToDevice(
		velocities.get(), d_velocities, 3 * particleCount);

	unsigned int* d_collisionCellIndices;
	dmm.allocateMemory(&d_collisionCellIndices, particleCount);
	dmm.copyElementsFromHostToDevice(
		collisionCellIndices.get(), d_collisionCellIndices, particleCount);

	MPCParticleVelocityType* d_collisionCellMomenta;
	dmm.allocateMemory(&d_collisionCellMomenta, 3 * collisionCellCount);
	dmm.copyElementsFromHostToDevice(
		ccMomenta.get(), d_collisionCellMomenta, 3 * collisionCellCount);

	FP* d_collisionCellMasses;
	dmm.allocateMemory(&d_collisionCellMasses, collisionCellCount);
	dmm.copyElementsFromHostToDevice(
		ccMasses.get(), d_collisionCellMasses, collisionCellCount);

	FP* d_collisionCellVelocityScalings;
	dmm.allocateMemory(&d_collisionCellVelocityScalings, collisionCellCount);
	dmm.copyElementsFromHostToDevice(
		ccScalings.get(), d_collisionCellVelocityScalings, collisionCellCount);


	OPENMPCD_CUDA_LAUNCH_WORKUNITS_SIZES_BEGIN(particleCount, 2, 2)
		DeviceCode::collisionCellStochasticRotationStep2 <<<gridSize, blockSize>>> (
			workUnitOffset,
			particleCount,
			d_velocities,
			d_collisionCellIndices,
			d_collisionCellMomenta,
			d_collisionCellMasses,
			d_collisionCellVelocityScalings);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	boost::scoped_array<MPCParticleVelocityType> newVelocities(
		new MPCParticleVelocityType[3 * particleCount]);
	dmm.copyElementsFromDeviceToHost(
		d_velocities,
		newVelocities.get(),
		3 * particleCount);



	for(std::size_t p = 0; p < particleCount; ++p)
	{
		RemotelyStoredVector<MPCParticleVelocityType> v(velocities.get(), p);

		const unsigned int cc = collisionCellIndices[p];

		v *= ccScalings[cc];

		RemotelyStoredVector<MPCParticleVelocityType> ccMomentum(
			ccMomenta.get(), cc);

		v += ccMomentum / ccMasses[cc];
	}


	for(std::size_t coord = 0; coord < 3 * particleCount; ++coord)
		REQUIRE(velocities[coord] == Approx(newVelocities[coord]));
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::undoLeesEdwardsVelocityCorrections`",
	"[CUDA]")
{
	static const unsigned int boxSize[3] = {1, 2, 3};
	static const std::size_t particleCount = 123;
	static const FP shearRate = 0.123;
	static const MPCParticlePositionType gridShift[3] = {0.1, 0.2, 0.3};
	static const FP mpcTime = 12.34;

	setSimulationBoxSizeSymbols(boxSize[0], boxSize[1], boxSize[2]);
	setLeesEdwardsSymbols(shearRate, boxSize[1]);

	DeviceMemoryManager dmm;
	dmm.setAutofree(true);

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

	MPCParticlePositionType* d_gridShift;
	dmm.allocateMemory(&d_gridShift, 3);
	dmm.copyElementsFromHostToDevice(gridShift, d_gridShift, 3);

	MPCParticlePositionType* d_positions;
	dmm.allocateMemory(&d_positions, 3 * particleCount);
	dmm.copyElementsFromHostToDevice(positions.get(), d_positions,
			3 * particleCount);

	MPCParticleVelocityType* d_velocities;
	dmm.allocateMemory(&d_velocities, 3 * particleCount);
	dmm.copyElementsFromHostToDevice(
		velocities.get(), d_velocities, 3 * particleCount);

	MPCParticleVelocityType* d_velocityCorrections;
	dmm.allocateMemory(&d_velocityCorrections, particleCount);

	unsigned int* d_collisionCellIndices;
	dmm.allocateMemory(&d_collisionCellIndices, particleCount);


	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(particleCount)
		DeviceCode::sortParticlesIntoCollisionCellsLeesEdwards <<<gridSize, blockSize>>> (
			workUnitOffset,
			particleCount,
			d_gridShift,
			mpcTime,
			d_positions,
			d_velocities,
			d_velocityCorrections,
			d_collisionCellIndices);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	REQUIRE(
		!dmm.elementMemoryEqualOnHostAndDevice(
			velocities.get(), d_velocities, 3 * particleCount));

	OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(particleCount)
		DeviceCode::undoLeesEdwardsVelocityCorrections <<<gridSize, blockSize>>> (
			workUnitOffset,
			particleCount,
			d_velocities,
			d_velocityCorrections);
	OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	boost::scoped_array<MPCParticleVelocityType> newVelocities(
		new MPCParticleVelocityType[3 * particleCount]);
	dmm.copyElementsFromDeviceToHost(
		d_velocities,
		newVelocities.get(),
		3 * particleCount);

	for(std::size_t coord = 0; coord < 3 * particleCount; ++coord)
	{
		REQUIRE(velocities[coord] == Approx(newVelocities[coord]));
	}
}


static void __global__
test_constructGPURNGs_destroyGPURNGs(
	GPURNG* rngs, float* const output,
	const std::size_t rngCount, const std::size_t valuesPerRNG)
{
	CUDA::Random::Distributions::Uniform0e1i<float> dist;

	for(std::size_t rngIdx = 0; rngIdx < rngCount; ++rngIdx)
	{
		GPURNG* const rng = rngs + rngIdx;

		for(std::size_t i = 0; i < valuesPerRNG; ++i)
			output[rngIdx * valuesPerRNG + i] = dist(*rng);
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::constructGPURNGs`, "
	"`OpenMPCD::CUDA::DeviceCode::destroyGPURNGs`",
	"[CUDA]")
{
	static const std::size_t rngCount = 3;
	static const std::size_t valuesPerRNG = 5;

	static const unsigned long long seed = 31415;

	static const float expected[] =
	{
		0.906760275363922119140625f,
		0.819705307483673095703125f,
		0.354439318180084228515625f,
		0.6189174652099609375f,
		0.7520935535430908203125f,
		0.68344795703887939453125f,
		0.19265888631343841552734375f,
		0.4641353189945220947265625f,
		0.94639813899993896484375f,
		0.8619234561920166015625f,
		0.4797227084636688232421875f,
		0.87264597415924072265625f,
		0.145604908466339111328125f,
		0.0365302003920078277587890625f,
		0.1215267479419708251953125f
	};

	DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	GPURNG* rngs;
	dmm.allocateMemory(&rngs, rngCount);

	DeviceCode::constructGPURNGs <<<1, 1>>> (
		rngCount,
		rngs,
		seed);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	float* output;
	dmm.allocateMemory(&output, rngCount * valuesPerRNG);

	test_constructGPURNGs_destroyGPURNGs <<<1, 1>>> (
		rngs, output,
		rngCount, valuesPerRNG);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	DeviceCode::destroyGPURNGs <<<1, 1>>> (
		rngCount,
		rngs);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			expected, output, rngCount * valuesPerRNG));
}
