/**
 * @file
 * Defines the `OpenMPCD::CUDA::BoundaryCondition::Factory` class.
 */

#ifndef OPENMPCD_CUDA_BOUNDARYCONDITION_FACTORY_HPP
#define OPENMPCD_CUDA_BOUNDARYCONDITION_FACTORY_HPP

#include <OpenMPCD/CUDA/BoundaryCondition/Base.hpp>
#include <OpenMPCD/Configuration.hpp>

#include <vector>

namespace OpenMPCD
{
namespace CUDA
{
namespace BoundaryCondition
{

/**
 * Class used to construct `OpenMPCD::CUDA::BoundaryCondition::Base` instances.
 *
 * Boundary conditions are configured in the OpenMPCD configuration file in the
 * `boundaryConditions` setting, which has to contain exactly one configuration
 * group, the name of which decides which type of boundary condition to use.
 */
class Factory
{
private:
	Factory(); ///< The constructor.

public:
	/**
	 * Returns a newly constructed boundary condition instance.
	 *
	 * The caller is responsible for deleting the pointer.
	 *
	 * @throw OpenMPCD::InvalidConfigurationException
	 *        Throws if either no, or multiple, boundary conditions have been
	 *        defined in the given configuration group, or none of the
	 *        configured boundary conditions are recognized.
	 *
	 * @param[in] config
	 *            The configuration group that holds the boundary condition
	 *            configuration.
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
	static BoundaryCondition::Base* getInstance(
		const Configuration::Setting& config,
		const unsigned int simulationBoxSizeX,
		const unsigned int simulationBoxSizeY,
		const unsigned int simulationBoxSizeZ);
}; //class Factory

} //namespace BoundaryCondition
} //namespace CUDA
} //namespace OpenMPCD

#endif //OPENMPCD_CUDA_BOUNDARYCONDITION_FACTORY_HPP
