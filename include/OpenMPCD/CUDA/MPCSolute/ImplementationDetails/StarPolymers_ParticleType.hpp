/**
 * @file
 * Defines
 * `OpenMPCD::CUDA::MPCSolute::ImplementationDetails::StarPolymers::ParticleType`.
 */

#ifndef OPENMPCD_CUDA_MPCSOLUTE_IMPLEMENTATIONDETAILS_STARPOLYMERS_PARTICLETYPE_HPP
#define OPENMPCD_CUDA_MPCSOLUTE_IMPLEMENTATIONDETAILS_STARPOLYMERS_PARTICLETYPE_HPP

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCSolute
{
namespace ImplementationDetails
{
namespace StarPolymers
{

/**
 * Holds the enumeration of particle types for star polymers.
 */
class ParticleType
{
public:
	/**
	 * Enumerates particle types.
	 */
	enum Enum
	{
		Core,    ///< Type for the core particle.
		Arm,     ///< Type for the particles that make up an arm.
		Magnetic ///< Type for the magnetic particles at the ends of arms.
	}; //enum Enum
private:
	ParticleType(); ///< The constructor.
}; //class ParticleType

} //namespace StarPolymers
} //namespace ImplementationDetails
} //namespace MPCSolute
} //namespace CUDA
} //namespace OpenMPCD

#endif //OPENMPCD_CUDA_MPCSOLUTE_IMPLEMENTATIONDETAILS_STARPOLYMERS_PARTICLETYPE_HPP
