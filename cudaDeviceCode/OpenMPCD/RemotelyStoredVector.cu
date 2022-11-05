#include <OpenMPCD/RemotelyStoredVector.hpp>

#include <OpenMPCD/CUDA/DeviceCode/Utilities.hpp>
#include <OpenMPCD/Vector3D.hpp>

#include <boost/static_assert.hpp>

template<typename T, unsigned int D> __device__ void
	OpenMPCD::RemotelyStoredVector<T, D>::atomicAdd(
		const RemotelyStoredVector<typename boost::add_const<T>::type, D>& rhs)
{
	for(std::size_t i=0; i<D; ++i)
		OpenMPCD::CUDA::DeviceCode::atomicAdd(&storage[i], rhs.get(i));
}

template<typename T, unsigned int D> __device__ void
	OpenMPCD::RemotelyStoredVector<T, D>::atomicAdd(
		const RemotelyStoredVector<typename boost::remove_const<T>::type, D>& rhs)
{
	for(std::size_t i=0; i<D; ++i)
		OpenMPCD::CUDA::DeviceCode::atomicAdd(&storage[i], rhs.get(i));
}

template<typename T, unsigned int D> __device__ void
	OpenMPCD::RemotelyStoredVector<T, D>::atomicAdd(
		const Vector3D<typename boost::remove_const<T>::type>& rhs)
{
	BOOST_STATIC_ASSERT(D == 3);

	OpenMPCD::CUDA::DeviceCode::atomicAdd(&storage[0], rhs.getX());
	OpenMPCD::CUDA::DeviceCode::atomicAdd(&storage[1], rhs.getY());
	OpenMPCD::CUDA::DeviceCode::atomicAdd(&storage[2], rhs.getZ());
}

template __device__ void
OpenMPCD::RemotelyStoredVector<float, 3U>::atomicAdd(const Vector3D<float>& rhs);
template __device__ void
OpenMPCD::RemotelyStoredVector<double, 3U>::atomicAdd(const Vector3D<double>& rhs);
