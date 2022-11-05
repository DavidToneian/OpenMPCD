#ifndef OPENMPCD_PAIRPOTENTIALS_FENE_HPP
#define OPENMPCD_PAIRPOTENTIALS_FENE_HPP

#include <OpenMPCD/PairPotentials/Base.hpp>

namespace OpenMPCD
{
namespace PairPotentials
{

/**
 * FENE Interaction
 * \f$ -0.5 K R^2 \log (1 - (\frac{r - l_0}{R})^2)\f$
 *
 * @tparam T The numeric base type.
 */
template<typename T = FP>
class FENE : public Base<T>
{
public:
	/**
	 * The constructor.
	 * @param[in] K 	Strength of FENE Interaction
	 * @param[in] l_0   Mean Bond length
	 * @param[in] R     Maximum FENE elongation
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	FENE(const T K, const T l_0, const T R)
		: K(K), l0(l_0), R(R)
	{
	}

	/**
	 * Returns the force vector of the interaction for a given position vector.
	 *
	 * This function returns the directional derivative
	 * \f[ - \nabla_R V \left( \vec{R} \right) \f]
	 * where \f$ \vec{R} \f$ is the `Rvec` parameter, \f$ V \f$ is the potential
	 * as given by the `potential` function, and \f$ \nabla_R V \f$ is the
	 * gradient of \f$ V \f$ with respect to \f$ \vec{R} \f$.
	 *
	 * @param[in]   Rvec    The relative position vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	Vector3D<T> force(const Vector3D<T>& Rvec) const
	{
		const T abs_r = Rvec.getMagnitude();
		return
			- Rvec.getNormalized() *
			(
					(K*(abs_r - l0))
					/
					( 1 - ( (abs_r - l0) * ( abs_r - l0 ) ) / ( R * R ) )
			);
	}

	/**
	 * Returns the potential of the interaction for a given position vector.
	 * @param[in]   Rvec    The relative position vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T potential(const Vector3D<T>& Rvec) const
	{
		T tmp = ((Rvec.getMagnitude() - l0) / R);
		tmp *= tmp;
		return -0.5*K*R*R*log( 1 - tmp );
	}

	/**
	 * Returns the \f$ K \f$ parameter.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T getK() const
	{
		return K;
	}

	/**
	 * Returns the \f$ R \f$ parameter.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T getR() const
	{
		return R;
	}

	/**
	 * Returns the \f$ l_0 \f$ parameter.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T get_l_0() const
	{
		return l0;
	}

private:
	const T K;
	const T l0;
	const T R;
}; //class FENE

} //namespace PairPotentials
} //namespace OpenMPCD
#endif //OPENMPCD_PAIRPOTENTIALS_FENE_HPP
