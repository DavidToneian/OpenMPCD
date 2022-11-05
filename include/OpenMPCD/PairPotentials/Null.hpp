/**
 * @file
 * Defines the `OpenMPCD::PairPotentials::Null` class.
 */

#ifndef OPENMPCD_PAIRPOTENTIALS_NULL_HPP
#define OPENMPCD_PAIRPOTENTIALS_NULL_HPP

#include <OpenMPCD/PairPotentials/Base.hpp>

namespace OpenMPCD
{
namespace PairPotentials
{

/**
 * Dummy interaction that represents a potential that is `0` everywhere, and
 * hence does not give rise to a force.
 */
template<typename T = FP>
class Null : public Base<T>
{
public:
	/**
	 * The constructor.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	Null()
	{
	}

	OPENMPCD_CUDA_HOST_AND_DEVICE
	Vector3D<T> force(const Vector3D<T>&) const
	{
		return Vector3D<T>(0, 0, 0);
	}

	OPENMPCD_CUDA_HOST_AND_DEVICE
	T potential(const Vector3D<T>&) const
	{
		return 0;
	}
}; //class Null

} //namespace PairPotentials
} //namespace OpenMPCD
#endif //OPENMPCD_PAIRPOTENTIALS_NULL_HPP
