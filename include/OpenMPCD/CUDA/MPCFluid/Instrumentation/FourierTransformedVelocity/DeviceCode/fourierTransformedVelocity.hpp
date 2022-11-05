/**
 * @file
 * Defines CUDA Device code for OpenMPCD::CUDA::MPCFluid::Instrumentation
 */

#ifndef OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_DEVICECODE_FOURIERTRANSFORMEDVELOCITY_HPP
#define OPENMPCD_CUDA_MPCFLUID_INSTRUMENTATION_DEVICECODE_FOURIERTRANSFORMEDVELOCITY_HPP

#include <OpenMPCD/Types.hpp>
#include <OpenMPCD/Vector3D.hpp>

#include <complex>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{
namespace DeviceCode
{
	/**
	 * @page velocityInFourierSpace Velocity in Fourier Space
	 * "Velocity in fourier space" refers to the discrete fourier transformation of the spatial coordinates
	 * of the velocities of a MPC fluid's constituents. The corresponding formula is given by
	 * \f[ \vec{v}\left(\vec{k}, t\right) =
	 *     \frac{1}{N} \sum_{i=1}^N \vec{v}_i\left(t\right) \cdot
	 *       \exp\left( \mathrm{i} \vec{k} \cdot \vec{r}_i\left(t\right) \right)
	 * \f]
	 * See, for example, "Hydrodnymic correlations in multiparticle collision dynamics" by Chien-Cheng Huang,
	 * Gerhard Gompper, and Roland G. Winkler, published in Physical Review E 86, 056711 (2012), equation (38).
	 */

	/**
	 * Calculates the velocity in Fourier space, assuming a simple MPC fluid.
	 *
	 * See @ref velocityInFourierSpace
	 *
	 * @param[in]  particleCount The number of MPC particles in the fluid.
	 * @param[in]  positions     The array of MPC fluid particle positions.
	 * @param[in]  velocities    The array of MPC fluid particle velocities.
	 * @param[in]  k             The \f$\vec{k}\f$ vector.
	 * @param[out] buffer1       A Device buffer that holds at least
	 *                           <c>3*numberOfConstituents*sizeof(MPCParticleVelocityType)</c>
	 *                           bytes.
	 * @param[out] buffer2       A Device buffer that holds at least
	 *                           <c>3*numberOfConstituents*sizeof(MPCParticleVelocityType)</c>
	 *                           bytes.
	 */
	const Vector3D<std::complex<MPCParticleVelocityType> > calculateVelocityInFourierSpace_simpleMPCFluid(
		const unsigned int particleCount,
		const MPCParticlePositionType* const positions,
		const MPCParticleVelocityType* const velocities,
		const Vector3D<MPCParticlePositionType> k,
		MPCParticleVelocityType* const buffer1,
		MPCParticleVelocityType* const buffer2);

	/**
	 * Device function for calculation of summands of the real and imaginary part of the velocity in Fourier space,
	 * assuming a simple MPC fluid.
	 *
	 * @remark
	 * This function is not intended to be called directly by the user. Instead, call
	 * calculateVelocityInFourierSpace_simpleMPCFluid().
	 *
	 * See @ref velocityInFourierSpace
	 *
	 * @param[in]  workUnitOffset  The number of particles to skip.
	 * @param[in]  positions       The device array of MPC fluid particle positions.
	 * @param[in]  velocities      The device array of MPC fluid particle velocities.
	 * @param[in]  k               The \f$\vec{k}\f$ vector.
	 * @param[out] realBuffer      A Device buffer that holds at least
	 *                             <c>3*numberOfConstituents*sizeof(MPCParticleVelocityType)</c>
	 *                             bytes, used to store the real parts of the summands.
	 * @param[out] imaginaryBuffer A Device buffer that holds at least
	 *                             <c>3*numberOfConstituents*sizeof(MPCParticleVelocityType)</c>
	 *                             bytes, used to store the imaginary parts of the summands.
	 */
	__global__ void calculateVelocityInFourierSpace_simpleMPCFluid_single(
		const unsigned int workUnitOffset,
		const MPCParticlePositionType* const positions,
		const MPCParticleVelocityType* const velocities,
		const Vector3D<MPCParticlePositionType> k,
		MPCParticleVelocityType* const realBuffer,
		MPCParticleVelocityType* const imaginaryBuffer);







	/**
	 * Calculates the velocity in Fourier space, assuming an MPC fluid consisting of pairs of MPC particles.
	 *
	 * See @ref velocityInFourierSpace
	 *
	 * @param[in]  doubletCount  The number of MPC particle doublets in the fluid.
	 * @param[in]  positions     The array of MPC fluid particle positions.
	 * @param[in]  velocities    The array of MPC fluid particle velocities.
	 * @param[in]  k             The \f$\vec{k}\f$ vector.
	 * @param[out] buffer1       A Device buffer that holds at least
	 *                           <c>3*doubletCount*sizeof(MPCParticleVelocityType)</c>
	 *                           bytes.
	 * @param[out] buffer2       A Device buffer that holds at least
	 *                           <c>3*doubletCount*sizeof(MPCParticleVelocityType)</c>
	 *                           bytes.
	 */
	const Vector3D<std::complex<MPCParticleVelocityType> > calculateVelocityInFourierSpace_doubletMPCFluid(
		const unsigned int doubletCount,
		const MPCParticlePositionType* const positions,
		const MPCParticleVelocityType* const velocities,
		const Vector3D<MPCParticlePositionType> k,
		MPCParticleVelocityType* const buffer1,
		MPCParticleVelocityType* const buffer2);

	/**
	 * Device function for calculation of summands of the real and imaginary part of the velocity in Fourier space,
	 * assuming an MPC fluid consisting of pairs of MPC particles.
	 *
	 * @remark
	 * This function is not intended to be called directly by the user. Instead, call
	 * calculateVelocityInFourierSpace_doubletMPCFluid().
	 *
	 * See @ref velocityInFourierSpace
	 *
	 * @param[in]  workUnitOffset  The number of MPC particle doublets to skip.
	 * @param[in]  positions       The device array of MPC fluid particle positions.
	 * @param[in]  velocities      The device array of MPC fluid particle velocities.
	 * @param[in]  k               The \f$\vec{k}\f$ vector.
	 * @param[out] realBuffer      A Device buffer that holds at least
	 *                             <c>3*doubletCount*sizeof(MPCParticleVelocityType)</c>
	 *                             bytes, used to store the real parts of the summands.
	 * @param[out] imaginaryBuffer A Device buffer that holds at least
	 *                             <c>3*doubletCount*sizeof(MPCParticleVelocityType)</c>
	 *                             bytes, used to store the imaginary parts of the summands.
	 */
	__global__ void calculateVelocityInFourierSpace_doubletMPCFluid_single(
		const unsigned int workUnitOffset,
		const MPCParticlePositionType* const positions,
		const MPCParticleVelocityType* const velocities,
		const Vector3D<MPCParticlePositionType> k,
		MPCParticleVelocityType* const realBuffer,
		MPCParticleVelocityType* const imaginaryBuffer);







	/**
	 * Calculates the velocity in Fourier space, assuming an MPC fluid consisting of triplets of MPC particles.
	 *
	 * See @ref velocityInFourierSpace
	 *
	 * @param[in]  tripletCount  The number of MPC particle triplets in the fluid.
	 * @param[in]  positions     The array of MPC fluid particle positions.
	 * @param[in]  velocities    The array of MPC fluid particle velocities.
	 * @param[in]  k             The \f$\vec{k}\f$ vector.
	 * @param[out] buffer1       A Device buffer that holds at least
	 *                           <c>3*tripletCount*sizeof(MPCParticleVelocityType)</c>
	 *                           bytes.
	 * @param[out] buffer2       A Device buffer that holds at least
	 *                           <c>3*tripletCount*sizeof(MPCParticleVelocityType)</c>
	 *                           bytes.
	 */
	const Vector3D<std::complex<MPCParticleVelocityType> > calculateVelocityInFourierSpace_tripletMPCFluid(
		const unsigned int tripletCount,
		const MPCParticlePositionType* const positions,
		const MPCParticleVelocityType* const velocities,
		const Vector3D<MPCParticlePositionType> k,
		MPCParticleVelocityType* const buffer1,
		MPCParticleVelocityType* const buffer2);

	/**
	 * Device function for calculation of summands of the real and imaginary part of the velocity in Fourier space,
	 * assuming an MPC fluid consisting of triplets of MPC particles.
	 *
	 * @remark
	 * This function is not intended to be called directly by the user. Instead, call
	 * calculateVelocityInFourierSpace_tripletMPCFluid().
	 *
	 * See @ref velocityInFourierSpace
	 *
	 * @param[in]  workUnitOffset  The number of MPC particle triplets to skip.
	 * @param[in]  positions       The device array of MPC fluid particle positions.
	 * @param[in]  velocities      The device array of MPC fluid particle velocities.
	 * @param[in]  k               The \f$\vec{k}\f$ vector.
	 * @param[out] realBuffer      A Device buffer that holds at least
	 *                             <c>3*tripletCount*sizeof(MPCParticleVelocityType)</c>
	 *                             bytes, used to store the real parts of the summands.
	 * @param[out] imaginaryBuffer A Device buffer that holds at least
	 *                             <c>3*tripletCount*sizeof(MPCParticleVelocityType)</c>
	 *                             bytes, used to store the imaginary parts of the summands.
	 */
	__global__ void calculateVelocityInFourierSpace_tripletMPCFluid_single(
		const unsigned int workUnitOffset,
		const MPCParticlePositionType* const positions,
		const MPCParticleVelocityType* const velocities,
		const Vector3D<MPCParticlePositionType> k,
		MPCParticleVelocityType* const realBuffer,
		MPCParticleVelocityType* const imaginaryBuffer);





	/**
	 * Calculates the velocity in Fourier space, assuming an MPC fluid consisting of chains of MPC particles.
	 *
	 * See @ref velocityInFourierSpace
	 *
	 * @param[in]  chainCount    The number of MPC particle chains in the fluid.
	 * @param[in]  chainLength   The number of MPC fluid particles in a chain.
	 * @param[in]  positions     The array of MPC fluid particle positions.
	 * @param[in]  velocities    The array of MPC fluid particle velocities.
	 * @param[in]  k             The \f$\vec{k}\f$ vector.
	 * @param[out] buffer1       A Device buffer that holds at least
	 *                           <c>3*chainCount*sizeof(MPCParticleVelocityType)</c>
	 *                           bytes.
	 * @param[out] buffer2       A Device buffer that holds at least
	 *                           <c>3*chainCount*sizeof(MPCParticleVelocityType)</c>
	 *                           bytes.
	 */
	const Vector3D<std::complex<MPCParticleVelocityType> > calculateVelocityInFourierSpace_chainMPCFluid(
		const unsigned int chainCount,
		const unsigned int chainLength,
		const MPCParticlePositionType* const positions,
		const MPCParticleVelocityType* const velocities,
		const Vector3D<MPCParticlePositionType> k,
		MPCParticleVelocityType* const buffer1,
		MPCParticleVelocityType* const buffer2);

	/**
	 * Device function for calculation of summands of the real and imaginary part of the velocity in Fourier space,
	 * assuming an MPC fluid consisting of chains of MPC particles.
	 *
	 * @remark
	 * This function is not intended to be called directly by the user. Instead, call
	 * calculateVelocityInFourierSpace_chainMPCFluid().
	 *
	 * See @ref velocityInFourierSpace
	 *
	 * @param[in]  workUnitOffset  The number of MPC particle chains to skip.
	 * @param[in]  chainLength     The number of MPC fluid particles in a chain.
	 * @param[in]  positions       The device array of MPC fluid particle positions.
	 * @param[in]  velocities      The device array of MPC fluid particle velocities.
	 * @param[in]  k               The \f$\vec{k}\f$ vector.
	 * @param[out] realBuffer      A Device buffer that holds at least
	 *                             <c>3*tripletCount*sizeof(MPCParticleVelocityType)</c>
	 *                             bytes, used to store the real parts of the summands.
	 * @param[out] imaginaryBuffer A Device buffer that holds at least
	 *                             <c>3*tripletCount*sizeof(MPCParticleVelocityType)</c>
	 *                             bytes, used to store the imaginary parts of the summands.
	 */
	__global__ void calculateVelocityInFourierSpace_chainMPCFluid_single(
		const unsigned int workUnitOffset,
		const unsigned int chainLength,
		const MPCParticlePositionType* const positions,
		const MPCParticleVelocityType* const velocities,
		const Vector3D<MPCParticlePositionType> k,
		MPCParticleVelocityType* const realBuffer,
		MPCParticleVelocityType* const imaginaryBuffer);






	/**
	 * Reduces the calculated summands of the velocity in Fourier space.
	 *
	 * @remark
	 * This function is not intended to be called by the user.
	 *
	 * See @ref velocityInFourierSpace
	 *
	 * @param[in] summandCount    The number of summands.
	 * @param[in] realBuffer      Device buffer holding the real parts of the summands.
	 * @param[in] imaginaryBuffer Device buffer holding the imaginary parts of the summands.
	 *
	 * @return Returns the velocity in Fourier space.
	 */
	const Vector3D<std::complex<MPCParticleVelocityType> > reduceVelocityInFourierSpaceBuffers(
		const unsigned int summandCount,
		MPCParticleVelocityType* const realBuffer,
		MPCParticleVelocityType* const imaginaryBuffer);
} //namespace DeviceCode
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

#endif
