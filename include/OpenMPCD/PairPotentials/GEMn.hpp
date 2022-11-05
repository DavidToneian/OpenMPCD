#ifndef OPENMPCD_PAIRPOTENTIALS_GEMN_HPP
#define OPENMPCD_PAIRPOTENTIALS_GEMN_HPP

#include <OpenMPCD/PairPotentials/Base.hpp>

namespace OpenMPCD
{
namespace PairPotentials
{

/**
 * GEM-n Interaction
 * \f$ \epsilon \exp ( - (\frac{r}{\sigma})^n ) \f$
 * An extension of the Gaussian Core Model,
 * see Stillinger, F. H.    Phase Transition in the gaussian core system    J.Chem.Phys. 65, 3968-3947 (1976)
 * and Nikoubashman, A.     Non-Equilibrium Computer Experiments of Soft Matter Systems TU Wien, PhD Thesis (2012)
 *
 * @tparam T The numeric base type.
 */
template<typename T = FP>
class GEMn : public Base<T>
{
public:
	/**
	 * The constructor.
	 * @param[in]   eps   Depth of the potential well.
	 * @param[in]   sigma Width of the gaussian core.
	 * @param[in]   n     Exponent of \f$ r \f$.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	GEMn(const T eps, const T sigma, const int n)
		: epsilon(eps), sigma(sigma), n(n)
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
		const T abs_r = R.getMagnitude();
		const T rOverSigma = pow( (abs_r/sigma), double(n));
		return (( n * epsilon * rOverSigma * exp((-1) * rOverSigma)) / (abs_r)) * R.getNormalized();
	}

	/**
	 * Returns the potential of the interaction for a given position vector.
	 * @param[in]   R   The relative position vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T potential(const Vector3D<T>& R) const
	{
		return epsilon*exp((-1) * pow( (R.getMagnitude()/sigma), double(n)) );
	}

private:
	const T epsilon;
	const T sigma;
	const int n;
}; //class GEMn

} //namespace PairPotentials
} //namespace OpenMPCD
#endif //OPENMPCD_PAIRPOTENTIALS_GEMN_HPP
