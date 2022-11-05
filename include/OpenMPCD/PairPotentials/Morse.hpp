#ifndef OPENMPCD_PAIRPOTENTIALS_MORSE_HPP
#define OPENMPCD_PAIRPOTENTIALS_MORSE_HPP

#include <OpenMPCD/PairPotentials/Base.hpp>

namespace OpenMPCD
{
namespace PairPotentials
{

/**
 * Morse Interaction
 * \f$ \epsilon ( \exp (-2 \ alpha (r - \sigma) ) - 2 \exp (-\alpha(r-\sigma))) - \textrm{shift} \f$
 *
 * @tparam T The numeric base type.
 */
template<typename T = FP>
class Morse : public Base<T>
{
public:
	/**
	 * The constructor.
	 * @param[in]   eps      depth of the potential well
	 * @param[in]   alpha    'width' of the potential well, smaller a, larger well
	 * @param[in]   sigma    equilibrium bond distance
	 * @param[in]   shift    The \f$ \textrm{shift} \f$ parameter.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	Morse(const T eps, const T alpha, const T sigma, const T shift)
		: epsilon(eps), alpha(alpha), sigma(sigma), shift(shift)
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
	 * @param[in]   R   The relative position vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	Vector3D<T> force(const Vector3D<T>& R) const
	{
		const T r = R.getMagnitude();
		return (-1)*epsilon * ((-2)*alpha*exp( -2*alpha*(r - sigma) ) + 2*alpha*exp( (-1)*alpha*(r-sigma) ))*R.getNormalized();
	}

	/**
	 * Returns the potential of the interaction for a given position vector.
	 * @param[in]   R   The relative position vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T potential(const Vector3D<T>& R) const
	{
		const T r = R.getMagnitude();
		return epsilon*
					(exp( -2*alpha*(r - sigma) )
	                 - 2 * exp( (-1)*alpha*(r-sigma)))
	                 - shift;
	}

private:
	const T epsilon;
	const T alpha;
	const T sigma;
	const T shift;
}; //class Morse

} //namespace PairPotentials
} //namespace OpenMPCD
#endif //OPENMPCD_PAIRPOTENTIALS_MORSE_HPP
