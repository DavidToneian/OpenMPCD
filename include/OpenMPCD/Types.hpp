/**
 * @file
 * Defines general types used in OpenMPCD.
 */

#ifndef OPENMPCD_TYPES_HPP
#define OPENMPCD_TYPES_HPP

#include <boost/random/mersenne_twister.hpp>

namespace OpenMPCD
{
	typedef double FP; ///< Default floating point type.

	typedef FP MPCParticlePositionType; ///< The data type for the positions of MPC particles.
	typedef FP MPCParticleVelocityType; ///< The data type for the velocities of MPC particles.

	typedef boost::mt11213b RNG; ///< The random number generator type.
}

#endif
