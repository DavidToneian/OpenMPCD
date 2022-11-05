#include <OpenMPCD/CUDA/MPCSolute/DeviceCode/Base.hpp>

#include <OpenMPCD/CUDA/Macros.hpp>
#include <OpenMPCD/CUDA/MPCSolute/DeviceCode/Symbols.cu>

void OpenMPCD::CUDA::MPCSolute::DeviceCode::setMPCParticleCountSymbol(const unsigned int count)
{
	cudaMemcpyToSymbol(soluteParticleCount, &count, sizeof(count));
	OPENMPCD_CUDA_THROW_ON_ERROR;
}
