/**
 * @file
 * Tests `OpenMPCD::CUDA::DeviceBuffer`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/DeviceBuffer.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/Exceptions.hpp>

#include <boost/scoped_array.hpp>

SCENARIO(
	"`OpenMPCD::CUDA::DeviceBuffer::DeviceBuffer`",
	"[CUDA]")
{
	using OpenMPCD::CUDA::DeviceBuffer;

	#ifdef OPENMPCD_DEBUG
		WHEN("`elementCount` is `0`")
		{
			THEN("`InvalidArgumentException` is thrown")
			{
				REQUIRE_THROWS_AS(
					DeviceBuffer<int>(0), OpenMPCD::InvalidArgumentException);
				REQUIRE_THROWS_AS(
					DeviceBuffer<float>(0), OpenMPCD::InvalidArgumentException);
			}
		}
	#endif

	DeviceBuffer<int>(1);
	DeviceBuffer<float>(2);
	DeviceBuffer<std::string>(100);
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceBuffer::getPointer`",
	"[CUDA]")
{
	using OpenMPCD::CUDA::DeviceBuffer;
	using OpenMPCD::CUDA::DeviceMemoryManager;

	static const unsigned int elementCount = 123;

	DeviceBuffer<int> db(elementCount);
	const DeviceBuffer<int>& cdb = db;

	int hostBuffer[elementCount];

	REQUIRE(DeviceMemoryManager::isDeviceMemoryPointer(db.getPointer()));
	REQUIRE(DeviceMemoryManager::isDeviceMemoryPointer(cdb.getPointer()));

	REQUIRE(db.getPointer() == cdb.getPointer());


	REQUIRE_NOTHROW(
		DeviceMemoryManager::copyElementsFromDeviceToHost(
			db.getPointer(), hostBuffer, elementCount));
	REQUIRE_NOTHROW(
		DeviceMemoryManager::copyElementsFromDeviceToHost(
			cdb.getPointer(), hostBuffer, elementCount));

	REQUIRE_NOTHROW(
		DeviceMemoryManager::copyElementsFromHostToDevice(
			hostBuffer, db.getPointer(), elementCount));
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceBuffer::getElementCount`",
	"[CUDA]")
{
	using OpenMPCD::CUDA::DeviceBuffer;
	using OpenMPCD::CUDA::DeviceMemoryManager;

	static const unsigned int elementCount = 234;

	DeviceBuffer<int> db(elementCount);
	const DeviceBuffer<int>& cdb = db;

	REQUIRE(db.getElementCount() == elementCount);
	REQUIRE(cdb.getElementCount() == elementCount);
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceBuffer::copyFromDevice`",
	"[CUDA]")
{
	static const unsigned int elementCount = 234;

	typedef int T;

	using OpenMPCD::CUDA::DeviceBuffer;

	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	DeviceBuffer<T> db(elementCount);
	dmm.zeroMemory(db.getPointer(), db.getElementCount());

	T values[elementCount];
	for(std::size_t i = 0; i < elementCount; ++i)
		values[i] = i;

	T* const d_src = dmm.allocateMemory<T>(elementCount);
	dmm.copyElementsFromHostToDevice(values, d_src, elementCount);

	db.copyFromDevice(d_src);

	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			values, db.getPointer(), elementCount));
}


SCENARIO(
	"`OpenMPCD::CUDA::DeviceBuffer::zeroMemory`",
	"[CUDA]")
{
	static const unsigned int elementCount = 234;

	typedef int T;

	using OpenMPCD::CUDA::DeviceBuffer;

	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	DeviceBuffer<T> db(elementCount);

	T values[elementCount];
	for(std::size_t i = 0; i < elementCount; ++i)
		values[i] = i;

	dmm.copyElementsFromHostToDevice(values, db.getPointer(), elementCount);

	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			values, db.getPointer(), elementCount));



	for(std::size_t i = 0; i < elementCount; ++i)
		values[i] = 0;

	REQUIRE_FALSE(
		dmm.elementMemoryEqualOnHostAndDevice(
			values, db.getPointer(), elementCount));


	db.zeroMemory();

	REQUIRE(
		dmm.elementMemoryEqualOnHostAndDevice(
			values, db.getPointer(), elementCount));
}
