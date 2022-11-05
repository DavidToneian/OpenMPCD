/**
 * @file
 * Tests `OpenMPCD::CUDA::Bitset`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/CUDA/Bitset.hpp>

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/Macros.hpp>


__global__ void tets_getBitCount(bool* const success)
{
	*success = true;

	for(std::size_t i = 0; i < 40; ++i)
	{
		OpenMPCD::CUDA::Bitset bitset(i);
		if(bitset.getBitCount() != i)
			*success = false;
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::Bitset::getBitCount`",
	"")
{
	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);
	bool* success;
	dmm.allocateMemory(&success, 1);
	dmm.zeroMemory(success, 1);

	tets_getBitCount<<<1, 1>>>(success);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	const bool true_ = true;
	REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(&true_, success, 1));
}



__global__ void
test_get_set_initialize(
	OpenMPCD::CUDA::Bitset* const bitset, const std::size_t bitCount)
{
	new(bitset) OpenMPCD::CUDA::Bitset(bitCount);

	for(std::size_t bit = 0; bit < bitset->getBitCount(); ++bit)
		bitset->set(bit, bit % 3 == 0);
}
__global__ void
test_get_set_toggle(OpenMPCD::CUDA::Bitset* const bitset)
{
	const unsigned int bit = blockIdx.x * blockDim.x + threadIdx.x;

	if(bitset->getBitCount() == 0)
		return;

	if(bit >= bitset->getBitCount())
		return;

	for(unsigned int loop = 0; loop < 10; ++loop)
		bitset->set(bit, !bitset->get(bit));
}
__global__ void
test_get_set_toggle_check(
	OpenMPCD::CUDA::Bitset* const bitset, bool* const success)
{
	*success = true;

	for(std::size_t bit = 0; bit < bitset->getBitCount(); ++bit)
	{
		const bool expected = bit % 3 == 0;
		if(bitset->get(bit) != expected)
			*success = false;
	}
}
__global__ void
test_get_set_finalize(OpenMPCD::CUDA::Bitset* const bitset)
{
	bitset->~Bitset();
}
SCENARIO(
	"`OpenMPCD::CUDA::Bitset::get`, "
		"`OpenMPCD::CUDA::Bitset::set`: concurrency",
	"")
{
	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	bool* success;
	dmm.allocateMemory(&success, 1);
	const bool true_ = true;

	OpenMPCD::CUDA::Bitset* bitset;
	dmm.allocateMemory(&bitset, 1);

	for(std::size_t i = 0; i < 500; i += 5)
	{
		dmm.zeroMemory(success, 1);

		test_get_set_initialize<<<1, 1>>>(bitset, i);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		test_get_set_toggle<<<1, 512>>>(bitset);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		test_get_set_toggle_check<<<1, 1>>>(bitset, success);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		test_get_set_finalize<<<1, 1>>>(bitset);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(&true_, success, 1));
	}
}


__global__ void test_get_set_bit_independence(
	OpenMPCD::CUDA::Bitset* const bitset, bool* const success)
{
	*success = true;

	for(std::size_t bit = 0; bit < bitset->getBitCount(); ++bit)
	{
		bitset->set(bit, false);

		if(bitset->get(bit))
			*success = false;
	}

	for(std::size_t bit = 0; bit < bitset->getBitCount(); ++bit)
	{
		if(bitset->get(bit))
			*success = false;
	}


	for(std::size_t bit = 0; bit < bitset->getBitCount(); ++bit)
	{
		bitset->set(bit, true);
		if(!bitset->get(bit))
			*success = false;

		bitset->set(bit, false);
		if(bitset->get(bit))
			*success = false;
	}



	for(std::size_t bit = 0; bit < bitset->getBitCount(); ++bit)
	{
		bitset->set(bit, true);

		if(!bitset->get(bit))
			*success = false;
	}

	for(std::size_t bit = 0; bit < bitset->getBitCount(); ++bit)
	{
		if(!bitset->get(bit))
			*success = false;
	}


	for(std::size_t bit = 0; bit < bitset->getBitCount(); ++bit)
	{
		bitset->set(bit, false);
		if(bitset->get(bit))
			*success = false;

		bitset->set(bit, true);
		if(!bitset->get(bit))
			*success = false;
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::Bitset::get`, "
		"`OpenMPCD::CUDA::Bitset::set`: independence of bits",
	"")
{
	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	bool* success;
	dmm.allocateMemory(&success, 1);
	const bool true_ = true;

	OpenMPCD::CUDA::Bitset* bitset;
	dmm.allocateMemory(&bitset, 1);

	for(std::size_t i = 0; i < 500; i += 5)
	{
		dmm.zeroMemory(success, 1);

		test_get_set_initialize<<<1, 1>>>(bitset, i);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		test_get_set_bit_independence<<<1, 1>>>(bitset, success);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		test_get_set_finalize<<<1, 1>>>(bitset);
		cudaDeviceSynchronize();
		OPENMPCD_CUDA_THROW_ON_ERROR;

		REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(&true_, success, 1));
	}
}




__global__ void test_setAll(bool* const success)
{
	*success = true;

	for(std::size_t i = 0; i < 40; ++i)
	{
		OpenMPCD::CUDA::Bitset bitset(i);

		bitset.setAll(false);
		for(std::size_t bit = 0; bit < bitset.getBitCount(); ++bit)
		{
			if(bitset.get(bit))
				*success = false;
		}

		bitset.setAll(true);
		for(std::size_t bit = 0; bit < bitset.getBitCount(); ++bit)
		{
			if(!bitset.get(bit))
				*success = false;
		}
	}
}
SCENARIO(
	"`OpenMPCD::CUDA::Bitset::setAll`",
	"")
{
	OpenMPCD::CUDA::DeviceMemoryManager dmm;
	dmm.setAutofree(true);

	bool* success;
	dmm.allocateMemory(&success, 1);
	const bool true_ = true;

	dmm.zeroMemory(success, 1);
	test_setAll<<<1, 1>>>(success);
	cudaDeviceSynchronize();
	OPENMPCD_CUDA_THROW_ON_ERROR;

	REQUIRE(dmm.elementMemoryEqualOnHostAndDevice(&true_, success, 1));
}
