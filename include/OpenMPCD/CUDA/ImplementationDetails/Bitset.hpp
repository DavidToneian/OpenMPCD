/**
 * @file
 * Implements functionality of the `OpenMPCD::CUDA::Bitset` class.
 */

#ifndef OPENMPCD_CUDA_IMPLEMENTATIONDETAILS_BITSET_HPP
#define OPENMPCD_CUDA_IMPLEMENTATIONDETAILS_BITSET_HPP

#include <OpenMPCD/CUDA/Bitset.hpp>

#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>


namespace OpenMPCD
{
namespace CUDA
{

OPENMPCD_CUDA_DEVICE
Bitset::Bitset(const std::size_t bitCount_)
	: storage(
		new unsigned int[(bitCount + bitsPerElement - 1) / bitsPerElement]),
	  bitCount(bitCount_)
{
}

OPENMPCD_CUDA_DEVICE
Bitset::~Bitset()
{
	delete[] storage;
}

OPENMPCD_CUDA_DEVICE
std::size_t Bitset::getBitCount() const
{
	return bitCount;
}

OPENMPCD_CUDA_DEVICE
void Bitset::set(const std::size_t bit, const bool value)
{
	OPENMPCD_DEBUG_ASSERT(bit < getBitCount());

	const std::size_t element = bit / bitsPerElement;
	const std::size_t bitInElement = bit % bitsPerElement;

	if(value)
	{
		atomicOr(&storage[element], 1 << bitInElement);
	}
	else
	{
		atomicAnd(&storage[element], ~(1U << bitInElement));
	}
}

OPENMPCD_CUDA_DEVICE
bool Bitset::get(const std::size_t bit) const
{
	OPENMPCD_DEBUG_ASSERT(bit < getBitCount());

	const std::size_t element = bit / bitsPerElement;
	const std::size_t bitInElement = bit % bitsPerElement;

	return (storage[element] >> bitInElement) & 1;
}

OPENMPCD_CUDA_DEVICE
void Bitset::setAll(const bool value)
{
	const std::size_t elementCount =
		(bitCount + bitsPerElement - 1) / bitsPerElement;

	for(std::size_t i = 0; i < elementCount; ++i)
		storage[i] = value ? -1 : 0;
}

} //namespace CUDA
} //namespace OpenMPCD


#endif //OPENMPCD_CUDA_IMPLEMENTATIONDETAILS_BITSET_HPP
