/**
 * @file
 * Implements functionality in the `OpenMPCD::CUDA::DeviceCode::NormalMode`
 * namespace.
 */

#ifndef OPENMPCD_CUDA_DEVICECODE_IMPLEMENTATIONDETAILS_NORMALMODE_HPP
#define OPENMPCD_CUDA_DEVICECODE_IMPLEMENTATIONDETAILS_NORMALMODE_HPP

#include <OpenMPCD/Utility/MathematicalFunctions.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>
#include <OpenMPCD/RemotelyStoredVector.hpp>

#include <boost/static_assert.hpp>
#include <boost/type_traits/is_floating_point.hpp>

namespace OpenMPCD
{
namespace CUDA
{
namespace DeviceCode
{
namespace NormalMode
{

template<typename T>
OPENMPCD_CUDA_DEVICE
const Vector3D<T> computeNormalCoordinate(
	const unsigned int i, const Vector3D<T>* const vectors,
	const std::size_t N, const T shift)
{
	BOOST_STATIC_ASSERT(boost::is_floating_point<T>::value);
		//non-floating-point `T` is probably a mistake

	OPENMPCD_DEBUG_ASSERT(vectors);
	OPENMPCD_DEBUG_ASSERT(N != 0);
	OPENMPCD_DEBUG_ASSERT(i <= N);

	Vector3D<T> ret(0, 0, 0);

	const T argPart = T(i) / T(N);
	for(std::size_t n = 1; n <= N; ++n)
	{
		const T arg = argPart * (n + shift);
		ret += vectors[n - 1] * Utility::MathematicalFunctions::cospi(arg);
	}

	return ret / T(N);
}

template<typename T>
OPENMPCD_CUDA_DEVICE
const Vector3D<T> computeNormalCoordinate(
	const unsigned int i, const T* const vectors, const std::size_t N,
	const T shift)
{
	BOOST_STATIC_ASSERT(boost::is_floating_point<T>::value);
		//non-floating-point `T` is probably a mistake

	OPENMPCD_DEBUG_ASSERT(vectors);
	OPENMPCD_DEBUG_ASSERT(N != 0);
	OPENMPCD_DEBUG_ASSERT(i <= N);

	Vector3D<T> ret(0, 0, 0);

	const T argPart = T(i) / T(N);
	for(std::size_t n = 1; n <= N; ++n)
	{
		const OpenMPCD::RemotelyStoredVector<const T> v(vectors, n - 1);
		ret +=
			v * Utility::MathematicalFunctions::cospi(argPart * (n + shift));
	}

	return ret / T(N);
}

template<typename T>
OPENMPCD_CUDA_DEVICE
void computeNormalCoordinates(
	const T* const vectors, const std::size_t N, T* const result,
	const T shift)
{
	BOOST_STATIC_ASSERT(boost::is_floating_point<T>::value);
		//non-floating-point `T` is probably a mistake

	for(std::size_t i = 0; i <= N; ++i)
	{
		OpenMPCD::RemotelyStoredVector<T> r(result, i);
		r = computeNormalCoordinate(i, vectors, N, shift);
	}
}

template<typename T>
__global__ void computeNormalCoordinates(
	const unsigned int workUnitOffset,
	const unsigned int chainLength,
	const unsigned int chainCount,
	const T* const positions,
	T* const normalModeCoordinates,
	const T shift)
{
	BOOST_STATIC_ASSERT(boost::is_floating_point<T>::value);
		//non-floating-point `T` is probably a mistake

	OPENMPCD_DEBUG_ASSERT(chainLength != 0);
	OPENMPCD_DEBUG_ASSERT(positions);
	OPENMPCD_DEBUG_ASSERT(normalModeCoordinates);

	const unsigned int chainID =
		blockIdx.x * blockDim.x + threadIdx.x + workUnitOffset;

	if(chainID >= chainCount)
		return;

	computeNormalCoordinates(
		positions + 3 * chainID * chainLength,
		chainLength,
		normalModeCoordinates + 3 * chainID * (chainLength + 1),
		shift);
}

} //namespace NormalMode
} //namespace DeviceCode
} //namespace CUDA
} //namespace OpenMPCD

#endif //OPENMPCD_CUDA_DEVICECODE_IMPLEMENTATIONDETAILS_NORMALMODE_HPP
