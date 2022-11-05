/**
 * @file
 * Defines the `OpenMPCD::CUDA::Random::Generators::Philox4x32_10` class.
 */

#ifndef OPENMPCD_CUDA_RANDOM_GENERATORS_PHILOX4X32_10_HPP
#define OPENMPCD_CUDA_RANDOM_GENERATORS_PHILOX4X32_10_HPP

#include <OpenMPCD/CUDA/Macros.hpp>

#include <curand.h>
#include <curand_philox4x32_x.h>
#include <curand_kernel.h>

namespace OpenMPCD
{
namespace CUDA
{
namespace Random
{
namespace Generators
{

/**
 * Philox4x32-10 @cite Salmon2011 counter-bases PRNG.
 *
 * This class is compatible with the Philox4x32-10 PRNG as implemented in the
 * cuRAND library.
 */
class Philox4x32_10
{
public:
	/**
	 * The constructor.
	 *
	 * @param[in] seed        The seed.
	 * @param[in] subsequence The subsequence number.
	 */
	OPENMPCD_CUDA_DEVICE
	Philox4x32_10(
		const unsigned long long seed,
		const unsigned long long subsequence)
	{
		curand_init(seed, subsequence, 0, getState());
	}

public:
	/**
	 * Returns a pointer to the internal cuRAND state.
	 */
	OPENMPCD_CUDA_DEVICE
	curandStatePhilox4_32_10_t* getState()
	{
		return &state;
	}

private:
	curandStatePhilox4_32_10_t state; ///< The internal cuRAND state.
}; //class Philox4x32_10

} //namespace Generators
} //namespace Random
} //namespace CUDA
} //namespace OpenMPCD


#endif //OPENMPCD_CUDA_RANDOM_GENERATORS_PHILOX4X32_10_HPP
