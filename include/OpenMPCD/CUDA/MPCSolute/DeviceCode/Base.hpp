/**
 * @file
 * Defines CUDA Device code for OpenMPCD::CUDA::MPCSolute::Base
 */

#ifndef OPENMPCD_CUDA_MPCSOLUTE_DEVICECODE_BASE_HPP
#define OPENMPCD_CUDA_MPCSOLUTE_DEVICECODE_BASE_HPP

#include <OpenMPCD/Types.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCSolute
{
/**
 * Contains CUDA Device code.
 */
namespace DeviceCode
{
	/**
	 * Sets the symbol mpcParticleCount for the solute.
	 * @param[in] count The number of MPC solute particles.
	 */
	void setMPCParticleCountSymbol(const unsigned int count);
} //namespace DeviceCode
} //namespace MPCSolute
} //namespace CUDA
} //namespace OpenMPCD

#endif
