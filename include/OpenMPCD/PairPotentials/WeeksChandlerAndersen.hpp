#ifndef OPENMPCD_PAIRPOTENTIALS_WEEKSCHANDLERANDERSEN_HPP
#define OPENMPCD_PAIRPOTENTIALS_WEEKSCHANDLERANDERSEN_HPP

#include <OpenMPCD/PairPotentials/Base.hpp>

#include <OpenMPCD/CUDA/DeviceCode/Utilities.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>

namespace OpenMPCD
{
namespace PairPotentials
{

/**
 * Weeks-Chandler-Andersen (WCA) potential.
 *
 * This potential has been introduced by Weeks, Chandler, and Andersen, in
 * J. Chem. Phys. 54, 5237 (1971). DOI: 10.1063/1.1674820
 *
 * It is given, as a function of the distance \f$ r \f$ of particles, and with
 * \f$ \epsilon \f$ and \f$ sigma \f$ being parameters, by
 * \f[
 *     4 * \epsilon *
 *     \left(
 *         \left( \frac{ \sigma }{ r } \right)^{12}
 *         -
 *         \left( \frac{ \sigma }{ r } \right)^{6}
 *         +
 *         \frac{ 1 }{ 4 }
 *     \right)
 *     *
 *     \theta \left( 2^{1/6} \sigma - r \right)
 * \f]
 * with \f$ \theta \left( x \right) \f$ being the Heaviside step function,
 * which is \f$ 1 \f$ if \f$ x > 0 \f$, and \f$ 0 \f$ otherwise.
 */
template<typename T = FP>
class WeeksChandlerAndersen : public Base<T>
{
public:
	/**
	 * The constructor.
	 *
	 * @param[in] epsilon The \f$ \epsilon \f$ parameter.
	 * @param[in] sigma   The \f$ \sigma \f$ parameter.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	WeeksChandlerAndersen(const T epsilon, const T sigma)
		: epsilon(epsilon), sigma(sigma)
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
	 * @throw OpenMPCD::AssertionException
	 *        If `OPENMPCD_DEBUG` is defined, throws if
	 *        `OpenMPCD::Scalar::isZero(R.getMagnitudeSquared())`.
	 *
	 * @param[in] R The relative position vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	Vector3D<T> force(const Vector3D<T>& R) const
	{
		const T r2 = R.getMagnitudeSquared();

		OPENMPCD_DEBUG_ASSERT(!Scalar::isZero(r2));

		using OpenMPCD::CUDA::DeviceCode::pow;

		const T sigma2 = sigma * sigma;
		const T cutoff = pow(2, 1.0/3) * sigma2;

		if( r2 >= cutoff )
			return Vector3D<T>(0, 0, 0);

		const T frac2 = sigma2 / r2;
		const T frac6 = frac2 * frac2 * frac2;
		const T frac12 = frac6 * frac6;

		return R * (24 * epsilon / r2 * ( 2 * frac12 - frac6 ));
	}

	/**
	 * Returns the potential of the interaction for a given position vector.
	 *
	 * @throw OpenMPCD::AssertionException
	 *        If `OPENMPCD_DEBUG` is defined, throws if
	 *        `OpenMPCD::Scalar::isZero(R.getMagnitudeSquared())`.
	 *
	 * @param[in] R The relative position vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T potential(const Vector3D<T>& R) const
	{
		const T r2 = R.getMagnitudeSquared();

		OPENMPCD_DEBUG_ASSERT(!Scalar::isZero(r2));

		using OpenMPCD::CUDA::DeviceCode::pow;

		const T sigma2 = sigma * sigma;
		const T cutoff = pow(2, 1.0/3) * sigma2;

		if( r2 >= cutoff )
			return 0;

		const T frac2 = sigma2 / r2;
		const T frac6 = frac2 * frac2 * frac2;
		const T frac12 = frac6 * frac6;

		return 4 * epsilon * (frac12 - frac6 + 1.0/4);
	}

private:
	const T epsilon; ///< The \f$ \epsilon \f$ parameter.
	const T sigma;   ///< The \f$ \sigma \f$ parameter.
}; //class WeeksChandlerAndersen

} //namespace PairPotentials
} //namespace OpenMPCD
#endif //OPENMPCD_PAIRPOTENTIALS_WEEKSCHANDLERANDERSEN_HPP
