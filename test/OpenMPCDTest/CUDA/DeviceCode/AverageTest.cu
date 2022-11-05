/**
 * @file
 * Tests functionality in `OpenMPCD/CUDA/DeviceCode/Average.hpp`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/DeviceCode/Average.hpp>
#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>

#include <boost/scoped_array.hpp>

SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::arithmeticMean_kernel`",
	"[CUDA]")
{
	OpenMPCD::CUDA::DeviceMemoryManager dmm;

	typedef double Element;
	static const std::size_t gridSize = 2;
	static const std::size_t blockSize = 512;
	static const std::size_t elementCount = gridSize * blockSize;

	boost::scoped_array<Element> host(new Element[elementCount]);
	Element sum = 0;
	for(std::size_t i = 0; i < elementCount; ++i)
	{
		host[i] = i;
		sum += host[i];
	}
	const Element mean = sum / elementCount;

	Element* const device = dmm.allocateMemory<Element>(elementCount);
	Element* const devOut = dmm.allocateMemory<Element>(1);
	dmm.copyElementsFromHostToDevice(host.get(), device, elementCount);

	OpenMPCD::CUDA::DeviceCode::arithmeticMean_kernel
		<<<gridSize, blockSize>>>(device, elementCount, devOut);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;


	Element deviceMean;
	dmm.copyElementsFromDeviceToHost(devOut, &deviceMean, 1);

	dmm.freeMemory(device);
	dmm.freeMemory(devOut);

	REQUIRE(mean == deviceMean);
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceCode::arithmeticMean`",
	"[CUDA]")
{
	using OpenMPCD::CUDA::DeviceCode::arithmeticMean;

	GIVEN("a range of values on the Device")
	{
		OpenMPCD::CUDA::DeviceMemoryManager dmm;

		typedef double Element;
		static const std::size_t elementCount = 1024;

		boost::scoped_array<Element> host(new Element[elementCount]);
		Element sum = 0;
		for(std::size_t i = 0; i < elementCount; ++i)
		{
			host[i] = i;
			sum += host[i];
		}
		const Element mean = sum / elementCount;

		Element* const device = dmm.allocateMemory<Element>(elementCount);
		Element* const devOut = dmm.allocateMemory<Element>(1);
		dmm.copyElementsFromHostToDevice(host.get(), device, elementCount);

		#ifdef OPENMPCD_DEBUG
			Element* null = 0;

			WHEN("`arithmeticMean` is called with `values == nullptr`")
			{
				THEN("`OpenMPCD::NULLPointerException` is thrown")
				{
					REQUIRE_THROWS_AS(
						arithmeticMean(null, elementCount, devOut),
						OpenMPCD::NULLPointerException);
				}
			}

			WHEN("`arithmeticMean` is called with `numberOfValues == 0`")
			{
				THEN("`OpenMPCD::InvalidArgumentException` is thrown")
				{
					REQUIRE_THROWS_AS(
						arithmeticMean(device, 0, devOut),
						OpenMPCD::InvalidArgumentException);
				}
			}

			WHEN("`arithmeticMean` is called with `output == nullptr`")
			{
				THEN("`OpenMPCD::NULLPointerException` is thrown")
				{
					REQUIRE_THROWS_AS(
						arithmeticMean(device, elementCount, null),
						OpenMPCD::NULLPointerException);
				}
			}

			WHEN("`arithmeticMean` is called with `values` a Host pointer")
			{
				THEN("`OpenMPCD::InvalidArgumentException` is thrown")
				{
					Element e;
					REQUIRE_THROWS_AS(
						arithmeticMean(&e, 1, devOut),
						OpenMPCD::InvalidArgumentException);
				}
			}

			WHEN("`arithmeticMean` is called with `output` a Host pointer")
			{
				THEN("`OpenMPCD::InvalidArgumentException` is thrown")
				{
					Element e;
					REQUIRE_THROWS_AS(
						arithmeticMean(device, 1, &e),
						OpenMPCD::InvalidArgumentException);
				}
			}
		#endif

		arithmeticMean(device, elementCount, devOut);
		REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(&mean, devOut, 1));

		dmm.freeMemory(device);
		dmm.freeMemory(devOut);
	}
}
