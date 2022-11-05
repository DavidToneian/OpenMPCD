/**
 * @file
 * Defines the
 * `OpenMPCD::PairPotentials::MagneticDipoleDipoleInteraction_ConstantEqualDipolesAlongZ`
 * class.
 */

#ifndef OPENMPCD_PAIRPOTENTIALS_MAGNETICDIPOLEDIPOLEINTERACTION_CONSTANTEQUALDIPOLESALONGZ_HPP
#define OPENMPCD_PAIRPOTENTIALS_MAGNETICDIPOLEDIPOLEINTERACTION_CONSTANTEQUALDIPOLESALONGZ_HPP

#include <OpenMPCD/PairPotentials/Base.hpp>

#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>

namespace OpenMPCD
{
namespace PairPotentials
{

/**
 * Interactions between two constant and equal magnetic dipoles oriented along
 * the Z axis.
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
 * assumed to be constant throughout time in size and orientation, with the
 * latter being along the \f$ z \f$ axis. Therefore, with \f$ m \f$ being the
 * magnitude of the individual dipole moments and with \f$ R_z \f$ being the
 * \f$ z \f$ component of \f$ \hat{r} \f$, the interaction potential is given by
 * \f[
 * 		- \frac{ \mu_0 m^2 }{ 4 \pi r^3 }
 * 		\left( 3 R_z^2 - 1 \right)
 * \f]
 *
 * @tparam T The numeric data type.
 */
template<typename T = FP>
class MagneticDipoleDipoleInteraction_ConstantEqualDipolesAlongZ
	: public Base<T>
{
public:
	/**
	 * The constructor.
	 *
	 * @param[in] prefactor The term \f$ \frac{\mu_0 m^2}{4 \pi} \f$.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	MagneticDipoleDipoleInteraction_ConstantEqualDipolesAlongZ(
		const T prefactor)
		: prefactor(prefactor)
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

		const T r_m2 = 1 / r2;
		const T r_m3 = r_m2 / sqrt(r2);
		const T r_m5 = r_m2 * r_m3;
		const T r_m7 = r_m5 * r_m2;

		const T r_x = R.getX();
		const T r_y = R.getY();
		const T r_z = R.getZ();
		const T r_z_squared = r_z * r_z;
		const T F_x = - 5 * r_x * r_z_squared * r_m7 + r_x * r_m5;
		const T F_y = - 5 * r_y * r_z_squared * r_m7 + r_y * r_m5;
		const T F_z = - 5 * r_z * r_z_squared * r_m7 + 3 * r_z * r_m5;

		const Vector3D<T> F(F_x, F_y, F_z);
		return (3 * prefactor) * F;
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
		const T r_z_squared = R.getZ() * R.getZ();

		return - prefactor * r_m3 * (3 * r_z_squared * r_m2 - 1);
	}

	/**
	 * Returns the term \f$ \frac{\mu_0 m^2}{4 \pi} \f$.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	T getPrefactor() const
	{
		return prefactor;
	}

private:
	const T prefactor; ///< The term \f$ \frac{\mu_0 m^2}{4 \pi} \f$.
}; //class MagneticDipoleDipoleInteraction_ConstantEqualDipolesAlongZ

} //namespace PairPotentials
} //namespace OpenMPCD
#endif //OPENMPCD_PAIRPOTENTIALS_MAGNETICDIPOLEDIPOLEINTERACTION_CONSTANTEQUALDIPOLESALONGZ_HPP
