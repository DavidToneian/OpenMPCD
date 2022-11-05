/**
 * @file
 * Defines the `OpenMPCD::CUDA::BoundaryCondition::LeesEdwards` class.
 */

#ifndef OPENMPCD_CUDA_BOUNDARYCONDITION_LEESEDWARDS_HPP
#define OPENMPCD_CUDA_BOUNDARYCONDITION_LEESEDWARDS_HPP

#include <OpenMPCD/CUDA/BoundaryCondition/Base.hpp>

#include <OpenMPCD/Configuration.hpp>
#include <OpenMPCD/Types.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace BoundaryCondition
{


/**
 * Lees-Edwards boundary conditions.
 *
 * @see LeesEdwardsBoundaryConditions
 *
 * The configuration group given in the constructor is expected to have exactly
 * one child setting, called `shearRate` and of floating-point type, that
 * specifies the Lees-Edwards shear rate \f$ \dot{\gamma} \f$.
 */
class LeesEdwards : public Base
{
public:
	/**
	 * The constructor.
	 *
	 * @throw OpenMPCD::InvalidConfigurationException
	 *        Throws if the configuration is invalid.
	 *
	 * @param[in] config
	 *            The configuration group that holds the Lees-Edwards boundary
	 *            condition configuration.
	 * @param[in] simulationBoxSizeX
	 *            The size of the primary simulation volume along the `x`
	 *            direction.
	 * @param[in] simulationBoxSizeY
	 *            The size of the primary simulation volume along the `y`
	 *            direction.
	 * @param[in] simulationBoxSizeZ
	 *            The size of the primary simulation volume along the `z`
	 *            direction.
	 */
	LeesEdwards(
		const Configuration::Setting& config,
		const unsigned int simulationBoxSizeX,
		const unsigned int simulationBoxSizeY,
		const unsigned int simulationBoxSizeZ);

	/**
	 * The destructor.
	 */
	virtual ~LeesEdwards()
	{
	}

public:
	virtual FP getTotalAvailableVolume() const;

	/**
	 * Returns the Lees-Edwards shear rate \f$ \dot{\gamma} \f$.
	 */
	FP getShearRate() const
	{
		return shearRate;
	}

private:
	const unsigned int primarySimulationVolume;
		///< The total volume of the primary simulation volume.

	FP shearRate; ///< The shear rate \f$ \dot{\gamma} \f$.
}; //class LeesEdwards

} //namespace BoundaryCondition
} //namespace CUDA
} //namespace OpenMPCD

#endif //OPENMPCD_CUDA_BOUNDARYCONDITION_LEESEDWARDS_HPP
