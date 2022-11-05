/**
 * @file
 * Declares some central CUDA constant symbols.
 */

#ifndef OPENMPCD_CUDA_DEVICECODE_SYMBOLS_HPP
#define OPENMPCD_CUDA_DEVICECODE_SYMBOLS_HPP

#include <OpenMPCD/Types.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace DeviceCode
{

extern __constant__ unsigned int mpcSimulationBoxSizeX;
	///< The size of the primary simulation box along the `x` direction.
extern __constant__ unsigned int mpcSimulationBoxSizeY;
	///< The size of the primary simulation box along the `y` direction.
extern __constant__ unsigned int mpcSimulationBoxSizeZ;
	///< The size of the primary simulation box along the `z` direction.

extern __constant__ unsigned int collisionCellCount;
	///< The number of collision cells in the system.

extern __constant__ FP streamingTimestep;
	///< The MPC streaming time-step.

extern __constant__ FP srdCollisionAngle;
	///< The collision angle for SRD collisions.


/**
 * Sets the symbols `mpcSimulationBoxSizeX`, `mpcSimulationBoxSizeY`,
 * `mpcSimulationBoxSizeZ`, and `collisionCellCount` in
 * `OpenMPCD::CUDA::DeviceCode`.
 *
 * `collisionCellCount` will be set to the product of the three arguments
 * supplied.
 *
 * @throw OpenMPCD::InvalidArgumentException
 *        If `OPENMPCD_DEBUG` is defined, throws if either argument is `0`.
 *
 * @param[in] simBoxSizeX The value for `mpcSimulationBoxSizeX`.
 * @param[in] simBoxSizeY The value for `mpcSimulationBoxSizeY`.
 * @param[in] simBoxSizeZ The value for `mpcSimulationBoxSizeZ`.
 */
void setSimulationBoxSizeSymbols(
	const unsigned int simBoxSizeX,
	const unsigned int simBoxSizeY,
	const unsigned int simBoxSizeZ);

/**
 * Sets the symbol `OpenMPCD::CUDA::DeviceCode::streamingTimestep`.
 *
 * @throw OpenMPCD::InvalidArgumentException
 *        If `OPENMPCD_DEBUG` is defined, throws if `timestep == 0`.
 *
 * @param[in] timestep The streaming timestep for each MPC fluid streaming step.
 */
void setMPCStreamingTimestep(const FP timestep);

/**
 * Sets the symbol `OpenMPCD::CUDA::DeviceCode::srdCollisionAngle`.
 *
 * @param[in] angle The SRD collision angle.
 */
void setSRDCollisionAngleSymbol(const FP angle);

} //namespace DeviceCode
} //namespace CUDA
} //namespace OpenMPCD

#endif
