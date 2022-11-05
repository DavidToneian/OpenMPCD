#ifndef OPENMPCD_PAIRPOTENTIALS_WEEKSCHANDLERANDERSEN_DISTANCEOFFSET_HPP
#define OPENMPCD_PAIRPOTENTIALS_WEEKSCHANDLERANDERSEN_DISTANCEOFFSET_HPP

#include <OpenMPCD/PairPotentials/Base.hpp>

#include <OpenMPCD/CUDA/DeviceCode/Utilities.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>

namespace OpenMPCD
{
namespace PairPotentials
{

/**
 * A generalization of the Weeks-Chandler-Andersen (WCA) potential.
 *
 * The Weeks-Chandler-Andersen potential has been introduced by Weeks, Chandler,
 * and Andersen, in J. Chem. Phys. 54, 5237 (1971). DOI: 10.1063/1.1674820
 *
 * This generalization introduces an offset \f$ D \f$ of the particle distance
 * \f$ r \f$. With \f$ \epsilon \f$ and \f$ sigma \f$ being parameters, the
 * interaction potential is given by
 * \f[
 *     4 * \epsilon *
 *     \left(
 *         \left( \frac{ \sigma }{ r - D } \right)^{12}
 *         -
 *         \left( \frac{ \sigma }{ r - D } \right)^{6}
 *         +
 *         \frac{ 1 }{ 4 }
 *     \right)
 *     *
 *     \theta \left( 2^{1/6} \sigma - r + D \right)
 * \f]
 * with \f$ \theta \left( x \right) \f$ being the Heaviside step function,
 * which is \f$ 1 \f$ if \f$ x > 0 \f$, and \f$ 0 \f$ otherwise.
 */
template<typename T = FP>
class WeeksChandlerAndersen_DistanceOffset : public Base<T>
{
public:
	/**
	 * The constructor.
	 *
	 * @throw OpenMPCD::AssertionException
	 *        If `OPENMPCD_DEBUG` is defined, throws if either
	 *        `epsilon < 0`, `sigma < 0`, or `d < 0`.
	 *
	 * @param[in] epsilon The \f$ \epsilon \f$ parameter.
	 * @param[in] sigma   The \f$ \sigma \f$ parameter.
	 * @param[in] d       The \f$ D \f$ parameter.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	WeeksChandlerAndersen_DistanceOffset(
		const T epsilon, const T sigma, const T d)
		: epsilon(epsilon), sigma(sigma), d(d)
	{
		OPENMPCD_DEBUG_ASSERT(!(epsilon < 0));
		OPENMPCD_DEBUG_ASSERT(!(sigma < 0));
		OPENMPCD_DEBUG_ASSERT(!(d < 0));
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
	 * @throw OpenMPCD::AssertionException
	 *        If `OPENMPCD_DEBUG` is defined, throws if
	 *        `R.getMagnitudeSquared() - d * d <= 0`.
	 *
	 * @param[in] R The relative position vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	Vector3D<T> force(const Vector3D<T>& R) const
	{
		const T r2 = R.getMagnitudeSquared();

		OPENMPCD_DEBUG_ASSERT(r2 - d * d > 0);

		using OpenMPCD::CUDA::DeviceCode::pow;

		const T sigma2 = sigma * sigma;
		const T cutoff = pow(2, 1.0/6) * sigma;

		const T r = sqrt(r2);
		if(r - d >= cutoff)
			return Vector3D<T>(0, 0, 0);

		const T denominator2 = (r - d) * (r - d);
		const T frac2 = sigma2 / denominator2;
		const T frac6 = frac2 * frac2 * frac2;
		const T frac12 = frac6 * frac6;

		return R * (24 * epsilon / (r * (r - d)) * ( 2 * frac12 - frac6 ));
	}

	/**
	 * Returns the potential of the interaction for a given position vector.
	 *
	 * @throw OpenMPCD::AssertionException
	 *        If `OPENMPCD_DEBUG` is defined, throws if
	 *        `R.getMagnitudeSquared() - d * d <= 0`.
	 *
	 * @param[in] R The relative position vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T potential(const Vector3D<T>& R) const
	{
		const T r2 = R.getMagnitudeSquared();
		const T d2 = d * d;

		OPENMPCD_DEBUG_ASSERT(r2 - d2 > 0);

		using OpenMPCD::CUDA::DeviceCode::pow;

		const T sigma2 = sigma * sigma;
		const T cutoff = pow(2, 1.0/6) * sigma;

		const T r = sqrt(r2);
		if(r - d >= cutoff)
			return 0;
		const T denominator2 = (r - d) * (r - d);
		const T frac2 = sigma2 / denominator2;
		const T frac6 = frac2 * frac2 * frac2;
		const T frac12 = frac6 * frac6;

		return 4 * epsilon * (frac12 - frac6 + 1.0/4);
	}

	/**
	 * Returns the \f$ \epsilon \f$ parameter.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T getEpsilon() const
	{
		return epsilon;
	}

	/**
	 * Returns the \f$ \sigma \f$ parameter.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T getSigma() const
	{
		return sigma;
	}

	/**
	 * Returns the \f$ D \f$ parameter.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T getD() const
	{
		return d;
	}

private:
	const T epsilon; ///< The \f$ \epsilon \f$ parameter.
	const T sigma;   ///< The \f$ \sigma \f$ parameter.
	const T d;       ///< The \f$ D \f$ parameter.
}; //class WeeksChandlerAndersen_DistanceOffset

} //namespace PairPotentials
} //namespace OpenMPCD
#endif //OPENMPCD_PAIRPOTENTIALS_WEEKSCHANDLERANDERSEN_DISTANCEOFFSET_HPP
