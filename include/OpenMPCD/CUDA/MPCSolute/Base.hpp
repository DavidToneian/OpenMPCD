/**
 * @file
 * Defines the OpenMPCD::CUDA::MPCSolute::Base class.
 */

#ifndef OPENMPCD_CUDA_MPCSOLUTE_BASE_HPP
#define OPENMPCD_CUDA_MPCSOLUTE_BASE_HPP

#include <OpenMPCD/CUDA/DeviceMemoryManager.hpp>
#include <OpenMPCD/CUDA/MPCSolute/Instrumentation/Base.hpp>
#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>
#include <OpenMPCD/RemotelyStoredVector.hpp>
#include <OpenMPCD/Vector3D.hpp>

namespace OpenMPCD
{
namespace CUDA
{
	class Simulation;

/**
 * Namespace for MPC Solute classes.
 */
namespace MPCSolute
{

/**
 * Base class for MPC solutes.
 *
 * @tparam PositionCoordinate The type to store position coordinates.
 * @tparam VelocityCoordinate The type to store velocity coordinates.
 */
template<typename PositionCoordinate, typename VelocityCoordinate>
class Base
{
protected:
	/**
	 * The constructor.
	 */
	Base();

private:
	Base(const Base&); ///< The copy constructor.

public:
	/**
	 * The destructor.
	 */
	virtual ~Base();

public:
	/**
	 * Performs, on the Device, an MD timestep of size `getMDTimeStepSize()`.
	 */
	virtual void performMDTimestep() = 0;

	/**
	 * Returns the number of MPC solute particles.
	 */
	virtual std::size_t getParticleCount() const = 0;

	/**
	 * Returns the number of logical entities in the solute.
	 */
	virtual std::size_t getNumberOfLogicalEntities() const = 0;

	/**
	 * Copies the MPC solute particles from the CUDA Device to the Host.
	 */
	void fetchFromDevice() const;

	/**
	 * Returns a MPC solute particle's position vector.
	 *
	 * @warning
	 * This function only returns the position that was current the last time
	 * `fetchFromDevice` was called.
	 *
	 * @throw OpenMPCD::OutOfBoundsException
	 *        If `OPENMPCD_DEBUG` is defined, throws if
	 *        `particleID >= getParticleCount()`.
	 *
	 * @param[in] particleID The particle ID.
	 */
	const RemotelyStoredVector<const PositionCoordinate>
		getPosition(const std::size_t particleID) const;

	/**
	 * Returns a MPC solute particle's velocity vector.
	 *
	 * @warning
	 * This function only returns the velocity that was current the last time
	 * `fetchFromDevice` was called.
	 *
	 * @throw OpenMPCD::OutOfBoundsException
	 *        If `OPENMPCD_DEBUG` is defined, throws if
	 *        `particleID >= getParticleCount()`.
	 *
	 * @param[in] particleID The particle ID.
	 */
	const RemotelyStoredVector<const VelocityCoordinate>
		getVelocity(const std::size_t particleID) const;

	/**
	 * Returns a const pointer to the MPC solute positions on the Device.
	 */
	const PositionCoordinate* getDevicePositions() const
	{
		return d_positions;
	}

	/**
	 * Returns a pointer to the MPC solute positions on the Device.
	 */
	PositionCoordinate* getDevicePositions()
	{
		return d_positions;
	}

	/**
	 * Returns a pointer to the MPC solute positions on the Host.
	 */
	PositionCoordinate* getHostPositions()
	{
		return h_positions.get();
	}

	/**
	 * Returns a const pointer to the MPC solute velocities on the Device.
	 */
	const VelocityCoordinate* getDeviceVelocities() const
	{
		return d_velocities;
	}

	/**
	 * Returns a pointer to the MPC solute velocities on the Device.
	 */
	VelocityCoordinate* getDeviceVelocities()
	{
		return d_velocities;
	}

	/**
	 * Returns a pointer to the MPC solute velocities on the Host.
	 */
	VelocityCoordinate* getHostVelocities()
	{
		return h_velocities.get();
	}

	/**
	 * Returns the number of time units advanced per MD step.
	 */
	FP getMDTimeStepSize() const
	{
		return mdTimeStepSize;
	}

	/**
	 * Returns whether the solute has instrumentation configured.
	 */
	bool hasInstrumentation() const
	{
		return instrumentation != NULL;
	}

	/**
	 * Returns the solute instrumentation.
	 *
	 * @throw OpenMPCD::NULLPointerException
	 *        If `OPENMPCD_DEBUG` is defined, throws if `!hasInstrumentation()`.
	 */
	Instrumentation::Base& getInstrumentation() const
	{
		OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
			hasInstrumentation(), NULLPointerException);

		return *const_cast<Instrumentation::Base*>(instrumentation);
	}

	/**
	 * Returns the mass of a particle, which is assumed to be equal for all
	 * particles in this instance.
	 */
	virtual FP getParticleMass() const = 0;

protected:
	/**
	 * Copies the MPC solute particles from the Host to the CUDA Device.
	 */
	void pushToDevice();

private:
	const Base& operator=(const Base&); ///< The assignment operator.

protected:
	DeviceMemoryManager deviceMemoryManager; ///< The Device memory manager.

	FP mdTimeStepSize; ///< The number of time units to advance per MD step.

	Instrumentation::Base* instrumentation; ///< The solute instrumentation.

	mutable boost::scoped_array<PositionCoordinate> h_positions;
		///< Host buffer for particle positions.
	mutable boost::scoped_array<VelocityCoordinate> h_velocities;
		///< Host buffer for particle velocities.

	PositionCoordinate* d_positions;  ///< Particle positions on the Device.
	VelocityCoordinate* d_velocities; ///< Particle velocities on the Device.
}; //class Base

} //namespace MPCSolute
} //namespace CUDA
} //namespace OpenMPCD

#endif
