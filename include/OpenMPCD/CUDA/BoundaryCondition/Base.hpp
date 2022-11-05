/**
 * @file
 * Defines the `OpenMPCD::CUDA::BoundaryCondition::Base` class.
 */

#ifndef OPENMPCD_CUDA_BOUNDARYCONDITION_BASE_HPP
#define OPENMPCD_CUDA_BOUNDARYCONDITION_BASE_HPP

#include <OpenMPCD/Types.hpp>

namespace OpenMPCD
{
namespace CUDA
{

/**
 * Namespace for boundary conditions.
 */
namespace BoundaryCondition
{


/**
 * Base class for boundary condition.
 */
class Base
{
protected:
	/**
	 * The constructor.
	 */
	Base()
	{
	}

public:
	/**
	 * The destructor.
	 */
	virtual ~Base()
	{
	}

public:
	/**
	 * Returns the total available volume.
	 *
	 * The total available volume is here understood to be the volume of the
	 * primary simulation volume, reduced by the volume that is excluded by the
	 * boundary conditions.
	 *
	 * The volume returned is in units of the MPC collision cell volume, where
	 * the MPC collision cell is assumed to be cubic and of side length
	 * \f$ 1 \f$.
	 */
	virtual FP getTotalAvailableVolume() const = 0;
}; //class Base

} //namespace BoundaryCondition
} //namespace CUDA
} //namespace OpenMPCD

#endif //OPENMPCD_CUDA_BOUNDARYCONDITION_BASE_HPP
