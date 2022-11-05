/**
 * @file
 * Defines the `OpenMPCD::CUDA::Bitset` class.
 */

#ifndef OPENMPCD_CUDA_BITSET_HPP
#define OPENMPCD_CUDA_BITSET_HPP

#include <OpenMPCD/CUDA/Macros.hpp>

#include <climits>

namespace OpenMPCD
{
namespace CUDA
{

/**
 * Represents a constant-size collection of boolean values.
 */
class Bitset
{
public:
	/**
	 * The constructor.
	 *
	 * The initial value of the bits is unspecified.
	 *
	 * @param[in] bitCount_
	 *            The number of bits to store.
	 */
	OPENMPCD_CUDA_DEVICE
	Bitset(const std::size_t bitCount_);

private:
	/**
	 * The copy constructor.
	 */
	Bitset(const Bitset&);

public:
	/**
	 * The destructor.
	 */
	OPENMPCD_CUDA_DEVICE
	~Bitset();

public:
	/**
	 * Returns the number of accessible bits.
	 */
	OPENMPCD_CUDA_DEVICE
	std::size_t getBitCount() const;

	/**
	 * Sets the given bit to the given value.
	 *
	 * This function is thread-safe.
	 *
	 * If `OPENMPCD_DEBUG` is defined, asserts that `bit` has a valid value.
	 *
	 * @param[in] bit
	 *            The bit to set, indexed in the range `[0, getBitCount())`.
	 * @param[in] value
	 *            The value to set the bit to.
	 */
	OPENMPCD_CUDA_DEVICE
	void set(const std::size_t bit, const bool value);

	/**
	 * Returns the given bit.
	 *
	 * This function is thread-safe.
	 *
	 * If `OPENMPCD_DEBUG` is defined, asserts that `bit` has a valid value.
	 *
	 * @param[in] bit
	 *            The bit to set, indexed in the range `[0, getBitCount())`.
	 */
	OPENMPCD_CUDA_DEVICE
	bool get(const std::size_t bit) const;

	/**
	 * Sets all bits to the given value.
	 *
	 * @param[in] value
	 *            The value to set the bits to.
	 */
	OPENMPCD_CUDA_DEVICE
	void setAll(const bool value);

private:
	/**
	 * The assignment operator.
	 */
	const Bitset& operator=(const Bitset&);

private:
	unsigned int* const storage; ///< The memory where the bits are stored.
	const std::size_t bitCount;  ///< The number of bits stored.

private:
	static const std::size_t bitsPerElement = sizeof(unsigned int) * CHAR_BIT;
		///< The number of bits per stored element.
}; //class Bitset
} //namespace CUDA
} //namespace OpenMPCD


#include <OpenMPCD/CUDA/ImplementationDetails/Bitset.hpp>

#endif //OPENMPCD_CUDA_BITSET_HPP
