#ifndef OPENMPCD_PAIRPOTENTIALS_BASE_HPP
#define OPENMPCD_PAIRPOTENTIALS_BASE_HPP

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/Types.hpp>
#include <OpenMPCD/Vector3D.hpp>

namespace OpenMPCD
{

/**
 * Namespace hosting various pair potentials, i.e. potentials depending on the
 * distance between pairs of particles.
 */
namespace PairPotentials
{

/**
 * Abstract base class for pair potentials.
 *
 * @tparam T The numeric base type.
 */
template<typename T = FP>
class Base
{
public:
	/**
	 * The destructor.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	virtual ~Base()
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
	virtual Vector3D<T> force(const Vector3D<T>& Rvec) const = 0;

	/**
	 * Returns the potential of the interaction for a given position vector.
	 * @param[in]   Rvec    The relative position vector.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	virtual T potential(const Vector3D<T>& Rvec) const = 0;

	/**
	 * Returns the force exerted on the particle at `r1` due to the particle at
	 * `r2`.
	 *
	 * @param[in] r1 The position of the first particle.
	 * @param[in] r2 The position of the second particle.
	 */
	OPENMPCD_CUDA_HOST_AND_DEVICE
	const Vector3D<T>
	forceOnR1DueToR2(const Vector3D<T>& r1, const Vector3D<T>& r2) const
	{
		return force(r1 - r2);
	}

}; //class Base

} //namespace PairPotentials
} //namespace OpenMPCD
#endif //OPENMPCD_PAIRPOTENTIALS_BASE_HPP
