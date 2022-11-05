/**
 * @file
 * Defines the `OpenMPCD::CUDA::Random::Generators::minstd_rand` and
 * `OpenMPCD::CUDA::Random::Generators::minstd_rand0` classes.
 */

#ifndef OPENMPCD_CUDA_RANDOM_GENERATORS_MINSTD_HPP
#define OPENMPCD_CUDA_RANDOM_GENERATORS_MINSTD_HPP

#include <OpenMPCD/CUDA/Random/Generators/LinearCongruent.hpp>

#include <boost/cstdint.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace Random
{
namespace Generators
{

typedef
	LinearCongruent<boost::uint_fast32_t, 48271, 0, 2147483647>
	minstd_rand;
	///< Generator corresponding to `std::minstd_rand`.
typedef
	LinearCongruent<boost::uint_fast32_t, 16807, 0, 2147483647>
	minstd_rand0;
	///< Generator corresponding to `std::minstd_rand0`.

} //namespace Generators
} //namespace Random
} //namespace CUDA
} //namespace OpenMPCD


#endif //OPENMPCD_CUDA_RANDOM_GENERATORS_MINSTD_HPP
