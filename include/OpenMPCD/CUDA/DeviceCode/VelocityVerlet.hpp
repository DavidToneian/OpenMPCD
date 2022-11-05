/**
 * @file
 * Defines CUDA device code for velocity-Verlet integration.
 */

#ifndef OPENMPCD_CUDA_DEVICECODE_VELOCITYVERLET_HPP
#define OPENMPCD_CUDA_DEVICECODE_VELOCITYVERLET_HPP

#include <OpenMPCD/RemotelyStoredVector.hpp>
#include <OpenMPCD/Types.hpp>
#include <OpenMPCD/Vector3D.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace DeviceCode
{
	/**
	 * Performs the first step in the velocity-Verlet algorithm.
	 * This function returns the new position \f$x(t+h)\f$ according to
	 * \f$x(t+h) = x(t) + v(t) h + \frac{1}{2} a h^2\f$
	 * where \f$v\f$ is the velocity, \f$a\f$ the acceleration
	 * and \f$h\f$ the timestep.
	 * @param[in] position     The position.
	 * @param[in] velocity     The velocity.
	 * @param[in] acceleration The acceleration.
	 * @param[in] timestep     The timestep.
	 */
	__device__ MPCParticlePositionType velocityVerletStep1(
		const MPCParticlePositionType position,
		const MPCParticleVelocityType velocity,
		const FP acceleration, const FP timestep);

	/**
	 * Performs the second step in the velocity-Verlet algorithm.
	 * This function returns the new velocity \f$y(t+h)\f$ according to
	 * \f$v(t+h) = v(t) + \frac{h}{2}\left(a(t) + a(t+h)\right)\f$
	 * where \f$x\f$ is the position, \f$a\f$ the acceleration
	 * and \f$h\f$ the timestep.
	 * @param[in] velocity        The velocity.
	 * @param[in] oldAcceleration The acceleration at the time \f$t\f$.
	 * @param[in] newAcceleration The acceleration at the time \f$t+h\f$.
	 * @param[in] timestep        The timestep.
	 */
	__device__ MPCParticleVelocityType velocityVerletStep2(
		const MPCParticleVelocityType velocity,
		const FP oldAcceleration,
		const FP newAcceleration,
		const FP timestep);

	/**
	 * Performs the first step in the velocity-Verlet algorithm.
	 * This function updates the position \f$x\f$ according to
	 * \f$x(t+h) = x(t) + v(t) h + \frac{1}{2} a h^2\f$
	 * where \f$v\f$ is the velocity, \f$a\f$ the acceleration
	 * and \f$h\f$ the timestep.
	 * @param[in,out] position     The position.
	 * @param[in]     velocity     The velocity.
	 * @param[in]     acceleration The acceleration.
	 * @param[in]     timestep     The timestep.
	 */
	__device__ void velocityVerletStep1(
		RemotelyStoredVector<MPCParticlePositionType>* const position,
		const RemotelyStoredVector<MPCParticleVelocityType> velocity,
		const Vector3D<FP> acceleration, const FP timestep);

	/**
	 * Performs the second step in the velocity-Verlet algorithm.
	 * This function updates the velocity \f$v\f$ according to
	 * \f$v(t+h) = v(t) + \frac{h}{2}\left(a(t) + a(t+h)\right)\f$
	 * where \f$x\f$ is the position, \f$a\f$ the acceleration
	 * and \f$h\f$ the timestep.
	 * @param[in,out] velocity        The velocity.
	 * @param[in]     oldAcceleration The acceleration at the time \f$t\f$.
	 * @param[in]     newAcceleration The acceleration at the time \f$t+h\f$.
	 * @param[in]     timestep        The timestep.
	 */
	__device__ void velocityVerletStep2(
		RemotelyStoredVector<MPCParticleVelocityType>* const velocity,
		const Vector3D<FP> oldAcceleration,
		const Vector3D<FP> newAcceleration,
		const FP timestep);
} //namespace DeviceCode
} //namespace CUDA
} //namespace OpenMPCD

#endif
