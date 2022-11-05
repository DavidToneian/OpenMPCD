/**
 * @file
 * Tests `OpenMPCD::CUDA::DeviceMemoryManager`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/runtime.hpp>

#include <OpenMPCD/Exceptions.hpp>

#include <boost/scoped_array.hpp>

SCENARIO(
	"`OpenMPCD::CUDA::DeviceMemoryManager::~DeviceMemoryManager`",
	"[CUDA]")
{
	using OpenMPCD::CUDA::DeviceMemoryManager;

	GIVEN("a DeviceMemoryManager instance")
	{
		DeviceMemoryManager* devMemMgr = new DeviceMemoryManager;

		WHEN("it is never used")
		{
			THEN("its destruction should not cause trouble")
			{
				delete devMemMgr;
			}
		}

		WHEN("memory is allocated and freed")
		{
			int* const ptr = devMemMgr->allocateMemory<int>(1);
			devMemMgr->freeMemory(ptr);

			THEN("its destruction should not cause trouble")
			{
				delete devMemMgr;
			}
		}
	}
}



SCENARIO(
	"`OpenMPCD::CUDA::DeviceMemoryManager::allocateMemory`, one-parameter version",
	"[CUDA]")
{
	GIVEN("a DeviceMemoryManager instance")
	{
		OpenMPCD::CUDA::DeviceMemoryManager devMemMgr;

		WHEN("`instanceCount` is non-zero")
		{
			THEN("everything should work")
			{
				double* const ptr = devMemMgr.allocateMemory<double>(123);

				REQUIRE(ptr != NULL);

				devMemMgr.freeMemory(ptr);
			}
		}

		WHEN("`instanceCount` is zero")
		{
			THEN("`nullptr` is returned")
			{
				REQUIRE(devMemMgr.allocateMemory<double>(0) == NULL);
			}
		}
	}
}



SCENARIO(
	"`OpenMPCD::CUDA::DeviceMemoryManager::allocateMemory`, two-parameter version",
	"[CUDA]")
{
	GIVEN("a DeviceMemoryManager instance")
	{
		OpenMPCD::CUDA::DeviceMemoryManager devMemMgr;

		WHEN("`instanceCount` is non-zero")
		{
			THEN("everything should work")
			{
				double* ptr = NULL;
				devMemMgr.allocateMemory(&ptr, 123);

				REQUIRE(ptr != NULL);

				devMemMgr.freeMemory(ptr);
			}
		}

		WHEN("`instanceCount` is zero")
		{
			THEN("the pointer is set to `nullptr`")
			{
				double tmp;
				double* ptr = &tmp;

				REQUIRE_NOTHROW(devMemMgr.allocateMemory(&ptr, 0));
				REQUIRE(ptr == NULL);
			}
		}

		#ifdef OPENMPCD_DEBUG
			WHEN("`pointerToPointer` is `nullptr`")
			{
				THEN("an exception is thrown")
				{
					REQUIRE_THROWS_AS(devMemMgr.allocateMemory<int>(NULL, 5), OpenMPCD::NULLPointerException);
				}
			}
		#endif
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceMemoryManager::allocateMemoryUnregistered`, "
	"one-parameter version",
	"[CUDA]")
{
	using OpenMPCD::CUDA::DeviceMemoryManager;

	WHEN("`instanceCount` is non-zero")
	{
		THEN("everything should work")
		{
			double* const ptr =
				DeviceMemoryManager::allocateMemoryUnregistered<double>(123);
			DeviceMemoryManager::freeMemoryUnregistered(ptr);
		}
	}

	WHEN("`instanceCount` is zero")
	{
		THEN("`nullptr` is returned")
		{
			REQUIRE(
				DeviceMemoryManager::allocateMemoryUnregistered<double>(0)
				==
				NULL);
		}
	}
}



SCENARIO(
	"`OpenMPCD::CUDA::DeviceMemoryManager::allocateMemoryUnregistered`, "
	"two-parameter version",
	"[CUDA]")
{
	using OpenMPCD::CUDA::DeviceMemoryManager;

	WHEN("`instanceCount` is non-zero")
	{
		THEN("everything should work")
		{
			double** ptr;
			DeviceMemoryManager::allocateMemoryUnregistered(&ptr, 123);
			DeviceMemoryManager::freeMemoryUnregistered(ptr);
		}
	}

	WHEN("`instanceCount` is zero")
	{
		THEN("the pointer is set to `nullptr`")
		{
			double tmp;
			double* ptr = &tmp;

			REQUIRE_NOTHROW(
				DeviceMemoryManager::allocateMemoryUnregistered(&ptr, 0));
			REQUIRE(ptr == NULL);
		}
	}

	#ifdef OPENMPCD_DEBUG
		WHEN("`pointerToPointer` is `nullptr`")
		{
			THEN("an exception is thrown")
			{
				REQUIRE_THROWS_AS(
					DeviceMemoryManager::allocateMemoryUnregistered<int>(
						NULL, 5),
					OpenMPCD::NULLPointerException);
			}
		}
	#endif
}



SCENARIO(
	"`OpenMPCD::CUDA::DeviceMemoryManager::freeMemory`",
	"[CUDA]")
{
	GIVEN("a DeviceMemoryManager instance")
	{
		OpenMPCD::CUDA::DeviceMemoryManager devMemMgr;

		WHEN("`pointer` is `nullptr`")
		{
			THEN("everything should work")
			{
				devMemMgr.freeMemory(NULL);
			}
		}

		WHEN("`pointer` is has been allocated through that instance")
		{
			int* const ptr = devMemMgr.allocateMemory<int>(10);

			THEN("everything should work")
			{
				devMemMgr.freeMemory(ptr);
			}
		}

		WHEN("`pointer` is has been allocated through another instance")
		{
			OpenMPCD::CUDA::DeviceMemoryManager devMemMgr2;
			int* const ptr = devMemMgr2.allocateMemory<int>(10);

			THEN("only that instance should be able to free it")
			{
				REQUIRE_THROWS_AS(devMemMgr.freeMemory(ptr), OpenMPCD::MemoryManagementException);
				devMemMgr2.freeMemory(ptr);
			}
		}

		WHEN("`pointer` is has been allocated through direct CUDA calls")
		{
			int* ptr;
			cudaMalloc(&ptr, 1);

			THEN("an exception should be thrown")
			{
				REQUIRE_THROWS_AS(devMemMgr.freeMemory(ptr), OpenMPCD::MemoryManagementException);
				cudaFree(ptr);
			}
		}

		WHEN("`pointer` is has been allocated on the host")
		{
			int* const ptr = new int;

			THEN("an exception should be thrown")
			{
				REQUIRE_THROWS_AS(devMemMgr.freeMemory(ptr), OpenMPCD::MemoryManagementException);
				delete ptr;
			}
		}
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceMemoryManager::freeMemoryUnregistered`",
	"[CUDA]")
{
	using OpenMPCD::CUDA::DeviceMemoryManager;

	WHEN("`pointer` is `nullptr`")
	{
		THEN("everything should work")
		{
			DeviceMemoryManager::freeMemoryUnregistered(NULL);
		}
	}

	WHEN("`pointer` is has been allocated through DeviceMemoryManager")
	{
		int* const ptr =
			DeviceMemoryManager::allocateMemoryUnregistered<int>(10);

		THEN("everything should work")
		{
			DeviceMemoryManager::freeMemoryUnregistered(ptr);
		}
	}

	WHEN("`pointer` is has been allocated through direct CUDA calls")
	{
		int* ptr;
		cudaMalloc(&ptr, 1);

		THEN("everything should work")
		{
			DeviceMemoryManager::freeMemoryUnregistered(ptr);
		}
	}

	WHEN("`pointer` is has been allocated on the host")
	{
		int* const ptr = new int;

		THEN("an exception should be thrown")
		{
			REQUIRE_THROWS_AS(
				DeviceMemoryManager::freeMemoryUnregistered(ptr),
				OpenMPCD::MemoryManagementException);
			delete ptr;
		}
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceMemoryManager::isDeviceMemoryPointer`",
	"[CUDA]")
{
	GIVEN("a DeviceMemoryManager instance")
	{
		using OpenMPCD::CUDA::DeviceMemoryManager;
		DeviceMemoryManager devMemMgr;

		WHEN("`isDeviceMemoryPointer` is given `nullptr`")
		{
			THEN("`false` is returned")
			{
				REQUIRE_FALSE(devMemMgr.isDeviceMemoryPointer(0));
			}
		}

		WHEN("`pointer` has been allocated through that instance")
		{
			int* const ptr = devMemMgr.allocateMemory<int>(10);

			THEN("`isDeviceMemoryPointer` is `true` for base pointer")
			{
				REQUIRE(DeviceMemoryManager::isDeviceMemoryPointer(ptr));
			}

			AND_THEN("`isDeviceMemoryPointer` is `true` for advanced pointers")
			{
				REQUIRE(DeviceMemoryManager::isDeviceMemoryPointer(ptr + 1));
				REQUIRE(DeviceMemoryManager::isDeviceMemoryPointer(ptr + 2));
				REQUIRE(DeviceMemoryManager::isDeviceMemoryPointer(ptr + 9));

				char* const tmp = reinterpret_cast<char*>(ptr);
				REQUIRE(DeviceMemoryManager::isDeviceMemoryPointer(tmp + 0));
				REQUIRE(DeviceMemoryManager::isDeviceMemoryPointer(tmp + 1));
			}

			devMemMgr.freeMemory(ptr);
		}

		WHEN("`pointer` has been allocated through direct CUDA calls")
		{
			int* ptr;
			cudaMalloc(&ptr, 3*sizeof(int));

			THEN("`isDeviceMemoryPointer` is `true` for base pointer")
			{
				REQUIRE(DeviceMemoryManager::isDeviceMemoryPointer(ptr));
			}

			AND_THEN("`isDeviceMemoryPointer` is `true` for advanced pointers")
			{
				REQUIRE(DeviceMemoryManager::isDeviceMemoryPointer(ptr + 1));
				REQUIRE(DeviceMemoryManager::isDeviceMemoryPointer(ptr + 2));

				char* const tmp = reinterpret_cast<char*>(ptr);
				REQUIRE(DeviceMemoryManager::isDeviceMemoryPointer(tmp + 0));
				REQUIRE(DeviceMemoryManager::isDeviceMemoryPointer(tmp + 1));
			}

			cudaFree(ptr);
		}

		WHEN("`pointer` has been allocated on the host via `new`")
		{
			int* const ptr = new int;

			THEN("`isDeviceMemoryPointer` is `false` for pointer")
			{
				REQUIRE_FALSE(DeviceMemoryManager::isDeviceMemoryPointer(ptr));
			}

			delete ptr;
		}

		WHEN("`pointer` has been allocated on the host via `new[]`")
		{
			int* const ptr = new int[3];

			THEN("`isDeviceMemoryPointer` is `false` for base pointer")
			{
				REQUIRE_FALSE(DeviceMemoryManager::isDeviceMemoryPointer(ptr));
			}

			AND_THEN("`isDeviceMemoryPointer` is `false` for advanced pointers")
			{
				REQUIRE_FALSE(
					DeviceMemoryManager::isDeviceMemoryPointer(ptr + 1));
				REQUIRE_FALSE(
					DeviceMemoryManager::isDeviceMemoryPointer(ptr + 2));

				char* const tmp = reinterpret_cast<char*>(ptr);
				REQUIRE_FALSE(
					DeviceMemoryManager::isDeviceMemoryPointer(tmp + 0));
				REQUIRE_FALSE(
					DeviceMemoryManager::isDeviceMemoryPointer(tmp + 1));
			}

			delete[] ptr;
		}

		WHEN("`pointer` has been allocated on the host via `malloc`")
		{
			int* const ptr = static_cast<int*>(malloc(3 * sizeof(int)));

			THEN("`isDeviceMemoryPointer` is `false` for base pointer")
			{
				REQUIRE_FALSE(DeviceMemoryManager::isDeviceMemoryPointer(ptr));
			}

			AND_THEN("`isDeviceMemoryPointer` is `false` for advanced pointers")
			{
				REQUIRE_FALSE(
					DeviceMemoryManager::isDeviceMemoryPointer(ptr + 1));
				REQUIRE_FALSE(
					DeviceMemoryManager::isDeviceMemoryPointer(ptr + 2));

				char* const tmp = reinterpret_cast<char*>(ptr);
				REQUIRE_FALSE(
					DeviceMemoryManager::isDeviceMemoryPointer(tmp + 0));
				REQUIRE_FALSE(
					DeviceMemoryManager::isDeviceMemoryPointer(tmp + 1));
			}

			free(ptr);
		}
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceMemoryManager::isHostMemoryPointer`",
	"[CUDA]")
{
	GIVEN("a DeviceMemoryManager instance")
	{
		using OpenMPCD::CUDA::DeviceMemoryManager;
		DeviceMemoryManager devMemMgr;

		WHEN("`isHostMemoryPointer` is given `nullptr`")
		{
			THEN("`false` is returned")
			{
				REQUIRE_FALSE(devMemMgr.isHostMemoryPointer(0));
			}
		}

		WHEN("`pointer` has been allocated through that instance")
		{
			int* const ptr = devMemMgr.allocateMemory<int>(10);

			THEN("`isHostMemoryPointer` is `false` for base pointer")
			{
				REQUIRE_FALSE(DeviceMemoryManager::isHostMemoryPointer(ptr));
			}

			AND_THEN("`isHostMemoryPointer` is `false` for advanced pointers")
			{
				REQUIRE_FALSE(
					DeviceMemoryManager::isHostMemoryPointer(ptr + 1));
				REQUIRE_FALSE(
					DeviceMemoryManager::isHostMemoryPointer(ptr + 2));
				REQUIRE_FALSE(
					DeviceMemoryManager::isHostMemoryPointer(ptr + 9));

				char* const tmp = reinterpret_cast<char*>(ptr);
				REQUIRE_FALSE(
					DeviceMemoryManager::isHostMemoryPointer(tmp + 0));
				REQUIRE_FALSE(
					DeviceMemoryManager::isHostMemoryPointer(tmp + 1));
			}

			devMemMgr.freeMemory(ptr);
		}

		WHEN("`pointer` has been allocated through direct CUDA calls")
		{
			int* ptr;
			cudaMalloc(&ptr, 3*sizeof(int));

			THEN("`isHostMemoryPointer` is `false` for base pointer")
			{
				REQUIRE_FALSE(DeviceMemoryManager::isHostMemoryPointer(ptr));
			}

			AND_THEN("`isHostMemoryPointer` is `false` for advanced pointers")
			{
				REQUIRE_FALSE(
					DeviceMemoryManager::isHostMemoryPointer(ptr + 1));
				REQUIRE_FALSE(
					DeviceMemoryManager::isHostMemoryPointer(ptr + 2));

				char* const tmp = reinterpret_cast<char*>(ptr);
				REQUIRE_FALSE(
					DeviceMemoryManager::isHostMemoryPointer(tmp + 0));
				REQUIRE_FALSE(
					DeviceMemoryManager::isHostMemoryPointer(tmp + 1));
			}

			cudaFree(ptr);
		}

		WHEN("`pointer` has been allocated on the host via `new`")
		{
			int* const ptr = new int;

			THEN("`isHostMemoryPointer` is `true` for pointer")
			{
				REQUIRE(DeviceMemoryManager::isHostMemoryPointer(ptr));
			}

			delete ptr;
		}

		WHEN("`pointer` has been allocated on the host via `new[]`")
		{
			int* const ptr = new int[3];

			THEN("`isHostMemoryPointer` is `true` for base pointer")
			{
				REQUIRE(DeviceMemoryManager::isHostMemoryPointer(ptr));
			}

			AND_THEN("`isHostMemoryPointer` is `true` for advanced pointers")
			{
				REQUIRE(
					DeviceMemoryManager::isHostMemoryPointer(ptr + 1));
				REQUIRE(
					DeviceMemoryManager::isHostMemoryPointer(ptr + 2));

				char* const tmp = reinterpret_cast<char*>(ptr);
				REQUIRE(
					DeviceMemoryManager::isHostMemoryPointer(tmp + 0));
				REQUIRE(
					DeviceMemoryManager::isHostMemoryPointer(tmp + 1));
			}

			delete[] ptr;
		}

		WHEN("`pointer` has been allocated on the host via `malloc`")
		{
			int* const ptr = static_cast<int*>(malloc(3 * sizeof(int)));

			THEN("`isHostMemoryPointer` is `true` for base pointer")
			{
				REQUIRE(DeviceMemoryManager::isHostMemoryPointer(ptr));
			}

			AND_THEN("`isHostMemoryPointer` is `true` for advanced pointers")
			{
				REQUIRE(
					DeviceMemoryManager::isHostMemoryPointer(ptr + 1));
				REQUIRE(
					DeviceMemoryManager::isHostMemoryPointer(ptr + 2));

				char* const tmp = reinterpret_cast<char*>(ptr);
				REQUIRE(
					DeviceMemoryManager::isHostMemoryPointer(tmp + 0));
				REQUIRE(
					DeviceMemoryManager::isHostMemoryPointer(tmp + 1));
			}

			free(ptr);
		}
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceMemoryManager::copyElementsFromHostToDevice`",
	"[CUDA]")
{
	GIVEN("buffers on the Host and the Device")
	{
		using OpenMPCD::CUDA::DeviceMemoryManager;
		DeviceMemoryManager devMemMgr;

		typedef double Element;
		static const std::size_t elementCount = 100;
		const std::size_t byteCount = sizeof(Element) * elementCount;

		boost::scoped_array<Element> host(new Element[elementCount]);
		boost::scoped_array<Element> copyFromDevice(new Element[elementCount]);
		Element* const device = devMemMgr.allocateMemory<Element>(elementCount);

		for(std::size_t i = 0; i < elementCount; ++i)
			host[i] = i;

		memset(copyFromDevice.get(), 0, byteCount);
		cudaMemset(device, 0, byteCount);
		OPENMPCD_CUDA_THROW_ON_ERROR;

		#ifdef OPENMPCD_DEBUG
			THEN("throws if `src == nullptr`")
			{
				Element* const null = 0;
				REQUIRE_THROWS_AS(
					DeviceMemoryManager::copyElementsFromHostToDevice
						(null, device, 1),
					OpenMPCD::NULLPointerException);
			}

			THEN("throws if `dest == nullptr`")
			{
				Element* const null = 0;
				REQUIRE_THROWS_AS(
					DeviceMemoryManager::copyElementsFromHostToDevice
						(host.get(), null, 1),
					OpenMPCD::NULLPointerException);
			}

			THEN("throws if `src` is not a Host pointer")
			{
				REQUIRE_THROWS_AS(
					DeviceMemoryManager::copyElementsFromHostToDevice
						(device, device, 1),
					OpenMPCD::InvalidArgumentException);
			}

			THEN("throws if `dest` is not a Device pointer")
			{
				REQUIRE_THROWS_AS(
					DeviceMemoryManager::copyElementsFromHostToDevice
						(host.get(), host.get(), 1),
					OpenMPCD::InvalidArgumentException);
			}
		#endif


		WHEN("the Host buffer is copied to the Device")
		{
			DeviceMemoryManager::copyElementsFromHostToDevice(
				host.get(), device, elementCount);

			THEN("the Device buffer equals the Host buffer")
			{
				cudaMemcpy(
					copyFromDevice.get(), device,
					byteCount, cudaMemcpyDeviceToHost);
				OPENMPCD_CUDA_THROW_ON_ERROR;

				REQUIRE(
					memcmp(host.get(), copyFromDevice.get(), byteCount) == 0);
			}
		}

		THEN("a call with `count == 0` is allowed")
		{
			REQUIRE_NOTHROW(
				DeviceMemoryManager::copyElementsFromHostToDevice
					(copyFromDevice.get(), device, 0));
		}

		devMemMgr.freeMemory(device);
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceMemoryManager::copyElementsFromDeviceToHost`",
	"[CUDA]")
{
	GIVEN("buffers on the Host and the Device")
	{
		using OpenMPCD::CUDA::DeviceMemoryManager;
		DeviceMemoryManager devMemMgr;

		typedef double Element;
		static const std::size_t elementCount = 100;
		const std::size_t byteCount = sizeof(Element) * elementCount;

		boost::scoped_array<Element> toDevice(new Element[elementCount]);
		boost::scoped_array<Element> copyFromDevice(new Element[elementCount]);
		Element* const device = devMemMgr.allocateMemory<Element>(elementCount);

		for(std::size_t i = 0; i < elementCount; ++i)
			toDevice[i] = i;

		memset(copyFromDevice.get(), 0, byteCount);
		cudaMemcpy(
			device, toDevice.get(), byteCount, cudaMemcpyHostToDevice);
		OPENMPCD_CUDA_THROW_ON_ERROR;


		#ifdef OPENMPCD_DEBUG
			THEN("throws if `src == nullptr`")
			{
				Element* const null = 0;
				REQUIRE_THROWS_AS(
					DeviceMemoryManager::copyElementsFromDeviceToHost
						(null, toDevice.get(), 1),
					OpenMPCD::NULLPointerException);
			}

			THEN("throws if `dest == nullptr`")
			{
				Element* const null = 0;
				REQUIRE_THROWS_AS(
					DeviceMemoryManager::copyElementsFromDeviceToHost
						(device, null, 1),
					OpenMPCD::NULLPointerException);
			}

			THEN("throws if `src` is not a Device pointer")
			{
				REQUIRE_THROWS_AS(
					DeviceMemoryManager::copyElementsFromDeviceToHost
						(toDevice.get(), toDevice.get(), 1),
					OpenMPCD::InvalidArgumentException);
			}

			THEN("throws if `dest` is not a Host pointer")
			{
				REQUIRE_THROWS_AS(
					DeviceMemoryManager::copyElementsFromDeviceToHost
						(device, device, 1),
					OpenMPCD::InvalidArgumentException);
			}
		#endif


		WHEN("the Device buffer is copied to the Host")
		{
			DeviceMemoryManager::copyElementsFromDeviceToHost(
				device, copyFromDevice.get(), elementCount);

			THEN("the Device buffer equals the Host buffer")
			{
				REQUIRE(
					memcmp(
						toDevice.get(), copyFromDevice.get(),
						byteCount)
					== 0);
			}
		}

		THEN("a call with `count == 0` is allowed")
		{
			REQUIRE_NOTHROW(
				DeviceMemoryManager::copyElementsFromDeviceToHost
					(device, copyFromDevice.get(), 0));
		}

		devMemMgr.freeMemory(device);
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceMemoryManager::copyElementsFromDeviceToDevice`",
	"[CUDA]")
{
	GIVEN("buffers on the Device")
	{
		using OpenMPCD::CUDA::DeviceMemoryManager;
		DeviceMemoryManager devMemMgr;

		typedef double Element;
		static const std::size_t elementCount = 100;
		const std::size_t byteCount = sizeof(Element) * elementCount;

		boost::scoped_array<Element> toDevice(new Element[elementCount]);
		boost::scoped_array<Element> copyFromDevice(new Element[elementCount]);
		Element* const d_src = devMemMgr.allocateMemory<Element>(elementCount);
		Element* const d_dest = devMemMgr.allocateMemory<Element>(elementCount);

		for(std::size_t i = 0; i < elementCount; ++i)
			toDevice[i] = i;

		memset(copyFromDevice.get(), 0, byteCount);
		cudaMemset(d_src, 0, byteCount);
		cudaMemset(d_dest, 0, byteCount);

		cudaMemcpy(
			d_src, toDevice.get(), byteCount, cudaMemcpyHostToDevice);
		OPENMPCD_CUDA_THROW_ON_ERROR;


		#ifdef OPENMPCD_DEBUG
			THEN("throws if `src == nullptr`")
			{
				Element* const null = 0;
				REQUIRE_THROWS_AS(
					DeviceMemoryManager::copyElementsFromDeviceToDevice
						(null, toDevice.get(), 1),
					OpenMPCD::NULLPointerException);
			}

			THEN("throws if `dest == nullptr`")
			{
				Element* const null = 0;
				REQUIRE_THROWS_AS(
					DeviceMemoryManager::copyElementsFromDeviceToDevice
						(d_src, null, 1),
					OpenMPCD::NULLPointerException);
			}

			THEN("throws if `src` is not a Device pointer")
			{
				REQUIRE_THROWS_AS(
					DeviceMemoryManager::copyElementsFromDeviceToDevice
						(toDevice.get(), d_dest, 1),
					OpenMPCD::InvalidArgumentException);
			}

			THEN("throws if `dest` is not a Device pointer")
			{
				REQUIRE_THROWS_AS(
					DeviceMemoryManager::copyElementsFromDeviceToDevice
						(d_src, toDevice.get(), 1),
					OpenMPCD::InvalidArgumentException);
			}
		#endif


		WHEN("the Device destination buffer is copied to the Host")
		{
			DeviceMemoryManager::copyElementsFromDeviceToDevice(
				d_src, d_dest, elementCount);

			DeviceMemoryManager::copyElementsFromDeviceToHost(
				d_dest, copyFromDevice.get(), elementCount);

			THEN("the Device buffer equals the Host buffer")
			{
				REQUIRE(
					memcmp(
						toDevice.get(), copyFromDevice.get(),
						byteCount)
					== 0);
			}

			AND_WHEN("a call with `count == 0` is made")
			{
				cudaMemset(d_src, 255, byteCount);
				DeviceMemoryManager::copyElementsFromDeviceToDevice(
					d_src, d_dest, 0);

				THEN("nothing has changed")
				{
					DeviceMemoryManager::copyElementsFromDeviceToHost(
						d_dest, copyFromDevice.get(), elementCount);
					REQUIRE(
						memcmp(
							toDevice.get(), copyFromDevice.get(),
							byteCount)
						== 0);
				}
			}
		}

		devMemMgr.freeMemory(d_src);
		devMemMgr.freeMemory(d_dest);
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceMemoryManager::zeroMemory`",
	"[CUDA]")
{
	GIVEN("a buffer on the Device")
	{
		using OpenMPCD::CUDA::DeviceMemoryManager;
		DeviceMemoryManager devMemMgr;

		typedef unsigned int Element;
		static const std::size_t elementCount = 100;
		static const std::size_t untouchedElementCount = 10;
		const std::size_t byteCount = sizeof(Element) * elementCount;
		REQUIRE(elementCount > untouchedElementCount);

		boost::scoped_array<Element> host(new Element[elementCount]);
		Element* const device = devMemMgr.allocateMemory<Element>(elementCount);

		cudaMemset(device, 1, byteCount);
		OPENMPCD_CUDA_THROW_ON_ERROR;


		#ifdef OPENMPCD_DEBUG
			THEN("throws if `start == nullptr`")
			{
				Element* const null = 0;
				REQUIRE_THROWS_AS(
					DeviceMemoryManager::zeroMemory(null, 1),
					OpenMPCD::NULLPointerException);
			}

			THEN("throws if `start` is not a Device pointer")
			{
				Element buf;
				REQUIRE_THROWS_AS(
					DeviceMemoryManager::zeroMemory(&buf, 1),
					OpenMPCD::InvalidArgumentException);
			}
		#endif


		WHEN("the Device buffer is zeroed")
		{
			DeviceMemoryManager::zeroMemory(
				device, elementCount - untouchedElementCount);

			THEN("the Device buffer equals zero except the last elements")
			{
				cudaMemcpy(
					host.get(), device, byteCount, cudaMemcpyDeviceToHost);
				OPENMPCD_CUDA_THROW_ON_ERROR;

				boost::scoped_array<Element>
					expected(new Element[elementCount]);

				memset(
					expected.get(), 0,
					sizeof(Element) * (elementCount - untouchedElementCount));
				memset(
					expected.get() + (elementCount - untouchedElementCount), 1,
					sizeof(Element) * untouchedElementCount);

				REQUIRE(memcmp(host.get(), expected.get(), byteCount) == 0);
			}
		}

		devMemMgr.freeMemory(device);
	}
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceMemoryManager::elementMemoryEqualOnHostAndDevice`",
	"[CUDA]")
{
	GIVEN("buffers on the Host and the Device")
	{
		using OpenMPCD::CUDA::DeviceMemoryManager;
		DeviceMemoryManager devMemMgr;

		typedef unsigned char Element;
		static const std::size_t elementCount = 3;
		const std::size_t byteCount = sizeof(Element) * elementCount;

		boost::scoped_array<Element> host(new Element[elementCount]);
		Element* const device = devMemMgr.allocateMemory<Element>(elementCount);

		for(std::size_t i = 0; i < elementCount; ++i)
			host[i] = i;

		WHEN("they are not equal")
		{
			THEN("`elementMemoryEqualOnHostAndDevice` returns `false`")
			{
				REQUIRE_FALSE(
					DeviceMemoryManager::elementMemoryEqualOnHostAndDevice(
						host.get(), device, elementCount - 1));
				REQUIRE_FALSE(
					DeviceMemoryManager::elementMemoryEqualOnHostAndDevice(
						host.get(), device, elementCount));
			}
		}

		WHEN("they are equal except for the last element")
		{
			cudaMemcpy(
				device, host.get(), elementCount - 1, cudaMemcpyHostToDevice);
			OPENMPCD_CUDA_THROW_ON_ERROR;

			THEN("`elementMemoryEqualOnHostAndDevice` returns appropriately")
			{
				REQUIRE(
					DeviceMemoryManager::elementMemoryEqualOnHostAndDevice(
						host.get(), device, elementCount - 1));
				REQUIRE_FALSE(
					DeviceMemoryManager::elementMemoryEqualOnHostAndDevice(
						host.get(), device, elementCount));
			}
		}

		WHEN("they are equal except for all elements")
		{
			cudaMemcpy(
				device, host.get(), elementCount, cudaMemcpyHostToDevice);
			OPENMPCD_CUDA_THROW_ON_ERROR;

			THEN("`elementMemoryEqualOnHostAndDevice` returns `true`")
			{
				REQUIRE(
					DeviceMemoryManager::elementMemoryEqualOnHostAndDevice(
						host.get(), device, elementCount - 1));
				REQUIRE(
					DeviceMemoryManager::elementMemoryEqualOnHostAndDevice(
						host.get(), device, elementCount));
			}
		}

		devMemMgr.freeMemory(device);
	}
}

