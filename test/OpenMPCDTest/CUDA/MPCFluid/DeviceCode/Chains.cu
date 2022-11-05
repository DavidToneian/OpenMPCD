/**
 * @file
 * Tests `OpenMPCD/CUDA/MPCFluid/DeviceCode/Chains.hpp`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Base.hpp>

#include <OpenMPCD/CUDA/DeviceBuffer.hpp>
#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Chains.hpp>
#include <OpenMPCD/RemotelyStoredVector.hpp>

#include <boost/scoped_array.hpp>
#include <boost/scoped_ptr.hpp>


template<typename T>
static void getTestcase(
	const unsigned int chainCount,
	const unsigned int chainLength,
	boost::scoped_ptr<OpenMPCD::CUDA::DeviceBuffer<T> >* const deviceVelocities,
	boost::scoped_array<T>* const expectedResult,
	boost::scoped_ptr<OpenMPCD::CUDA::DeviceBuffer<T> >* const deviceBuffer)
{
	using OpenMPCD::CUDA::DeviceBuffer;

	REQUIRE(chainCount != 0);
	REQUIRE(chainLength != 0);
	REQUIRE(deviceVelocities != 0);
	REQUIRE(expectedResult != 0);

	const unsigned int particleCount = chainCount * chainLength;

	OpenMPCD::CUDA::MPCFluid::DeviceCode::
		setMPCParticleCountSymbol(particleCount);


	T hostVelocities[3 * particleCount];
	for(unsigned int p = 0; p < particleCount; ++p)
	{
		hostVelocities[3 * p + 0] = 0.2 * p;
		hostVelocities[3 * p + 1] = - 0.1 * p;
		hostVelocities[3 * p + 2] = 3.21 + 1.0 / p;
	}

	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	deviceVelocities->reset(new DeviceBuffer<T>(3 * particleCount));
	expectedResult->reset(new T[3 * chainCount]);
	deviceBuffer->reset(new DeviceBuffer<T>(3 * chainCount));

	dmm.copyElementsFromHostToDevice(
		hostVelocities,
		deviceVelocities->get()->getPointer(),
		3 * particleCount);
	dmm.zeroMemory(deviceBuffer->get()->getPointer(), 3 * chainCount);


	T* const expected = expectedResult->get();

	for(unsigned int e = 0; e < chainCount; ++e)
	{
		expected[3 * e + 0] = 0;
		expected[3 * e + 1] = 0;
		expected[3 * e + 2] = 0;

		for(unsigned int p = 0; p < chainLength; ++p)
		{
			expected[3 * e + 0] +=
				hostVelocities[3 * (e * chainLength + p) + 0];
			expected[3 * e + 1] +=
				hostVelocities[3 * (e * chainLength + p) + 1];
			expected[3 * e + 2] +=
				hostVelocities[3 * (e * chainLength + p) + 2];
		}

		expected[3 * e + 0] /= chainLength;
		expected[3 * e + 1] /= chainLength;
		expected[3 * e + 2] /= chainLength;
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::DeviceCode::"
		"getCenterOfMassVelocities_chain`",
	"[CUDA]")
{
	typedef OpenMPCD::MPCParticleVelocityType T;

	static const unsigned int chainCount = 123;


	using OpenMPCD::CUDA::DeviceBuffer;

	OpenMPCD::CUDA::DeviceMemoryManager dmm;

	for(unsigned int chainLength = 1; chainLength <= 6; ++chainLength)
	{
		boost::scoped_ptr<DeviceBuffer<T> > deviceVelocities;
		boost::scoped_array<T> expectedResult;
		boost::scoped_ptr<DeviceBuffer<T> > deviceBuffer;

		getTestcase<T>(
			chainCount,
			chainLength,
			&deviceVelocities,
			&expectedResult,
			&deviceBuffer);

		OpenMPCD::CUDA::MPCFluid::DeviceCode::
		getCenterOfMassVelocities_chain(
			chainCount * chainLength,
			chainLength,
			deviceVelocities->getPointer(),
			deviceBuffer->getPointer());

		REQUIRE(
			dmm.elementMemoryEqualOnHostAndDevice(
				expectedResult.get(),
				deviceBuffer->getPointer(),
				3 * chainCount));
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::DeviceCode::"
		"getCenterOfMassVelocities_chain_kernel`",
	"[CUDA]")
{
	typedef OpenMPCD::MPCParticleVelocityType T;

	static const unsigned int chainCount = 123;


	using OpenMPCD::CUDA::DeviceBuffer;

	OpenMPCD::CUDA::DeviceMemoryManager dmm;

	for(unsigned int chainLength = 1; chainLength <= 6; ++chainLength)
	{
		boost::scoped_ptr<DeviceBuffer<T> > deviceVelocities;
		boost::scoped_array<T> expectedResult;
		boost::scoped_ptr<DeviceBuffer<T> > deviceBuffer;

		getTestcase<T>(
			chainCount,
			chainLength,
			&deviceVelocities,
			&expectedResult,
			&deviceBuffer);


		OPENMPCD_CUDA_LAUNCH_WORKUNITS_BEGIN(2 * chainCount)
			OpenMPCD::CUDA::MPCFluid::DeviceCode::
			getCenterOfMassVelocities_chain_kernel<<<gridSize, blockSize>>>(
				workUnitOffset,
				chainLength,
				deviceVelocities->getPointer(),
				deviceBuffer->getPointer());
		OPENMPCD_CUDA_LAUNCH_WORKUNITS_END
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		REQUIRE(
			dmm.elementMemoryEqualOnHostAndDevice(
				expectedResult.get(),
				deviceBuffer->getPointer(),
				3 * chainCount));
	}
}


static __global__ void
test_getCenterOfMassVelocity_chain(
	const unsigned int chainCount,
	const unsigned int chainLength,
	const OpenMPCD::MPCParticleVelocityType* const velocities,
	OpenMPCD::MPCParticleVelocityType* const comVelocities)
{
	for(unsigned int i = 0; i < chainCount; ++i)
	{
		OpenMPCD::RemotelyStoredVector<OpenMPCD::MPCParticleVelocityType>
			result(comVelocities, i);

		result =
			OpenMPCD::CUDA::MPCFluid::DeviceCode::getCenterOfMassVelocity_chain(
				i,
				chainLength,
				velocities);
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::MPCFluid::DeviceCode::"
		"getCenterOfMassVelocity_chain`",
	"[CUDA]")
{
	typedef OpenMPCD::MPCParticleVelocityType T;

	static const unsigned int chainCount = 123;


	using OpenMPCD::CUDA::DeviceBuffer;

	OpenMPCD::CUDA::DeviceMemoryManager dmm;

	for(unsigned int chainLength = 1; chainLength <= 6; ++chainLength)
	{
		boost::scoped_ptr<DeviceBuffer<T> > deviceVelocities;
		boost::scoped_array<T> expectedResult;
		boost::scoped_ptr<DeviceBuffer<T> > deviceBuffer;

		getTestcase<T>(
			chainCount,
			chainLength,
			&deviceVelocities,
			&expectedResult,
			&deviceBuffer);


		test_getCenterOfMassVelocity_chain<<<1, 1>>>(
			chainCount,
			chainLength,
			deviceVelocities->getPointer(),
			deviceBuffer->getPointer());
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		REQUIRE(
			dmm.elementMemoryEqualOnHostAndDevice(
				expectedResult.get(),
				deviceBuffer->getPointer(),
				3 * chainCount));
	}
}
