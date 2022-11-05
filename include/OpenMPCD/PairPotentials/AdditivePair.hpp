#ifndef OPENMPCD_PAIRPOTENTIALS_ADDITIVEPAIR_HPP
#define OPENMPCD_PAIRPOTENTIALS_ADDITIVEPAIR_HPP

#include <OpenMPCD/PairPotentials/Base.hpp>

#include <OpenMPCD/CUDA/DeviceCode/Utilities.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>

namespace OpenMPCD
{
namespace PairPotentials
{

/**
 * Represents a potential that consists of the sum of two other potentials.
 */
template<typename T = FP>
class AdditivePair : public Base<T>
{
public:
	/**
	 * The constructor.
	 *
	 * @throw OpenMPCD::NULLPointerException
	 *        If `OPENMPCD_DEBUG` is defined, throws if `potential1 == nullptr`
	 *        or `potential2 == nullptr`.
	 *
	 * @param[in] potential1 A pointer to the first potential.
	 *                       This pointer must remain valid as long as this
	 *                       instance exists.
	 * @param[in] potential2 A pointer to the second potential.
	 *                       This pointer must remain valid as long as this
	 *                       instance exists.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	AdditivePair(
		const Base<T>* const potential1, const Base<T>* const potential2)
		: potential1(potential1), potential2(potential2)
	{
		#ifdef OPENMPCD_DEBUG
			if(potential1 == NULL || potential2 == NULL)
			{
				OPENMPCD_THROW(
					NULLPointerException, "`potential1`, `potential2`");
			}
		#endif
	}

	/**
	 * Returns the force vector of the interaction for a given position vector.
	 *
	 * @param[in] R The relative position vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	Vector3D<T> force(const Vector3D<T>& R) const
	{
		return potential1->force(R) + potential2->force(R);
	}

	/**
	 * Returns the potential of the interaction for a given position vector.
	 *
	 * @param[in] R The relative position vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T potential(const Vector3D<T>& R) const
	{
		return potential1->potential(R) + potential2->potential(R);
	}

private:
	const Base<T>* const potential1; ///< The first potential.
	const Base<T>* const potential2; ///< The second potential.
}; //class AdditivePair

} //namespace PairPotentials
} //namespace OpenMPCD
#endif //OPENMPCD_PAIRPOTENTIALS_ADDITIVEPAIR_HPP
