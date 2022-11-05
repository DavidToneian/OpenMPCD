#include <OpenMPCD/CUDA/MPCFluid/DeviceCode/Symbols.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace MPCFluid
{
namespace DeviceCode
{

__constant__ unsigned int mpcParticleCount;

__constant__ FP omega;
__constant__ FP cos_omegaTimesTimestep;
__constant__ FP sin_omegaTimesTimestep;

} //namespace DeviceCode
} //namespace MPCFluid
} //namespace CUDA
} //namespace OpenMPCD

