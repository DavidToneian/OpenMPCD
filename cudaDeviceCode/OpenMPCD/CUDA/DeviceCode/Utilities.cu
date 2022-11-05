#include <OpenMPCD/CUDA/DeviceCode/Utilities.hpp>

__device__ void OpenMPCD::CUDA::DeviceCode::atomicAdd(double* const target, const double increment)
{
	unsigned long long int* const address = reinterpret_cast<unsigned long long int*>(target);
    unsigned long long int old = *address;
    unsigned long long int assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address, assumed, __double_as_longlong(increment + __longlong_as_double(assumed)));
    } while (assumed != old);
}

__device__ void OpenMPCD::CUDA::DeviceCode::atomicAdd(float* const target, const float increment)
{
	::atomicAdd(target, increment);
}
