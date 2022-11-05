#ifndef OPENMPCD_PAIRPOTENTIALS_LENNARDJONES_HPP
#define OPENMPCD_PAIRPOTENTIALS_LENNARDJONES_HPP

#include <OpenMPCD/PairPotentials/Base.hpp>

namespace OpenMPCD
{
namespace PairPotentials
{

/** Lennard-Jones Interaction
 * \f[
 *   4 \varepsilon \cdot
 *   \left(
 *     \left( \frac{\sigma}{r - r_{\textrm{offset}}} \right)^{12}
 *     -
 *     \left( \frac{\sigma}{r - r_{\textrm{offset}}} \right)^6
 *   \right)
 * \f]
 * If \f$ r \f$ exceeds a parameter \f$ r_{\textrm{cut}} \f$ (given to the
 * constructor of the class), the resulting potential and force will be zero.
 *
 * @tparam T The numeric base type.
 */
template<typename T = FP>
class LennardJones : public Base<T>
{
public:
	/**
	 * The constructor
	 *
	 * @param[in]   r_offset     Offset for distance
	 * @param[in]   r_cut        Cutoff param; If distance > r_cut, potential is zero
	 * @param[in]   sigma        At distance sigma, inter-particle potential is zero
	 * @param[in]   epsilon      Depth of the potential well
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	LennardJones(const T r_offset, const T r_cut, const T sigma, const T epsilon)
		: r_offset(r_offset), r_cut(r_cut), sigma(sigma), epsilon(epsilon)
	{
	}

	/**
	 * Returns the force vector of the interaction for a given position vector.
	 *
	 * This function returns the directional derivative
	 * \f[ - \nabla_R V \left( \vec{R} \right) \f]
	 * where \f$ \vec{R} \f$ is the `R` parameter, \f$ V \f$ is the potential
	 * as given by the `potential` function, and \f$ \nabla_R V \f$ is the
	 * gradient of \f$ V \f$ with respect to \f$ \vec{R} \f$.
	 *
	 * @param[in]   R    The relative position vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	Vector3D<T> force(const Vector3D<T>& R) const
	{
		const T Rv = R.getMagnitude();

		if (Rv > r_cut ) {
			return Vector3D<T>(0,0,0);
		}

		const T oneOverRMinusROffset = 1.0 / (Rv - r_offset);
		const T ratio = sigma * oneOverRMinusROffset;
		const T ratio6 = pow(ratio, 6);
		const T ratio12 = ratio6 * ratio6;
		return 24 * epsilon * (2 * ratio12 - ratio6) * oneOverRMinusROffset*R.getNormalized();
	}

	/**
	 * Returns the potential of the interaction for a given position vector.
	 * @param[in]   R    The relative position vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T potential(const Vector3D<T>& R) const
	{
		const T Rv = R.getMagnitude();

		if (Rv > r_cut)
		{
			return 0.0;
		}
		else
		{
			return 4*epsilon*(pow(sigma/(Rv-r_offset ), 12) - pow(sigma/(Rv-r_offset),6));
		}
	}

private:
    const T r_offset;
    const T r_cut;
    const T sigma;
    const T epsilon;
}; //class LennardJones

} //namespace PairPotentials
} //namespace OpenMPCD
#endif //OPENMPCD_PAIRPOTENTIALS_LENNARDJONES_HPP
