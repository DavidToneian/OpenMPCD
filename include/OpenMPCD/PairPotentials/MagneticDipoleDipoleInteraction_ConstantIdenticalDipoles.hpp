/**
 * @file
 * Defines the
 * `OpenMPCD::PairPotentials::MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles`
 * class.
 */

#ifndef OPENMPCD_PAIRPOTENTIALS_MAGNETICDIPOLEDIPOLEINTERACTION_CONSTANTIDENTICALDIPOLES_HPP
#define OPENMPCD_PAIRPOTENTIALS_MAGNETICDIPOLEDIPOLEINTERACTION_CONSTANTIDENTICALDIPOLES_HPP

#include <OpenMPCD/PairPotentials/Base.hpp>

#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>

#include <boost/core/is_same.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/remove_cv.hpp>

namespace OpenMPCD
{
namespace PairPotentials
{

/**
 * Interactions between two constant and identical magnetic dipoles.
 *
 * The general magnetic dipole-dipole interaction potential is given by
 * \f[
 * 		- \frac{ \mu_0 }{ 4 \pi r^3 }
 * 		\left(
 * 			3
 * 			\left(\vec{m_1} \cdot \hat{r} \right)
 * 			\left(\vec{m_2} \cdot \hat{r} \right)
 * 			-
 * 			\vec{m_1} \cdot \vec{m_2}
 * 		\right)
 * \f]
 * where \f$ \mu_0 \f$ is the vacuum permeability, \f$ \hat{r} \f$ and \f$ r \f$
 * are, respectively, the unit vector and length of the vector \f$ \vec{r} \f$
 * that points from one dipole's position to the other's, \f$ \vec{m_1} \f$ and
 * \f$ \vec{m_2} \f$ are the magnetic dipole moments, and \f$ \cdot \f$ denotes
 * the inner product.
 *
 * In the special case treated in this class, the magnetic dipole moments are
 * assumed to be constant throughout time in size and orientation. Therefore,
 * with \f$ m \f$ being the magnitude of the individual dipole moments and with
 * \f$ \hat{m} \f$ being the unit vector of the individual dipole moments, the
 * interaction potential is given by
 * \f[
 * 		- \frac{ \mu_0 m^2 }{ 4 \pi r^3 }
 * 		\left( 3 \left(\hat{m} \cdot \hat{r} \right)^2 - 1 \right)
 * \f]
 *
 * @tparam T The numeric data type.
 */
template<typename T = FP>
class MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles
	: public Base<T>
{
public:
	/**
	 * The constructor.
	 *
	 * @param[in] prefactor_
	 *            The term \f$ \frac{\mu_0 m^2}{4 \pi} \f$.
	 * @param[in] orientation_
	 *            The orientation unit vector \f$ \hat{m} \f$ of the dipole
	 *            moments.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles(
		const T prefactor_, const Vector3D<T>& orientation_)
		: prefactor(prefactor_), orientation(orientation_)
	{
		//Type `float` is not currently tested in the unit tests, since it
		//seems to result in considerable inaccuracies.
		BOOST_STATIC_ASSERT(
			!boost::is_same<
				typename boost::remove_cv<T>::type,
				float>
			::value);
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

		const T r_m2 = 1 / r2;
		const T r_m3 = r_m2 / sqrt(r2);
		const T r_m5 = r_m2 * r_m3;

		const T dot = orientation.dot(R);

		Vector3D<T> F(2 * dot * orientation);

		F -= (5 * r_m2 * dot * dot - 1) * R;

		return (3 * prefactor * r_m5) * F;
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

		const T r_m2 = 1 / r2;
		const T r_m3 = r_m2 / sqrt(r2);
		const T dot = R.dot(orientation);

		return - prefactor * r_m3 * (3 * dot * dot * r_m2 - 1);
	}

	/**
	 * Returns the term \f$ \frac{\mu_0 m^2}{4 \pi} \f$.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T getPrefactor() const
	{
		return prefactor;
	}

	/**
	 * Returns the dipole orientation \f$ \hat{m} \f$.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	const Vector3D<T>& getDipoleOrientation() const
	{
		return orientation;
	}

private:
	const T prefactor; ///< The term \f$ \frac{\mu_0 m^2}{4 \pi} \f$.
	const Vector3D<T> orientation;
		///< The orientation unit vector \f$ \hat{m} \f$ of the dipole moments.
}; //class MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles

} //namespace PairPotentials
} //namespace OpenMPCD
#endif //OPENMPCD_PAIRPOTENTIALS_MAGNETICDIPOLEDIPOLEINTERACTION_CONSTANTIDENTICALDIPOLES_HPP
