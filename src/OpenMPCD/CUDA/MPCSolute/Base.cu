#include <OpenMPCD/CUDA/MPCSolute/Base.hpp>

#include <OpenMPCD/Configuration.hpp>
#include <OpenMPCD/CUDA/runtime.hpp>


namespace OpenMPCD
{
namespace CUDA
{
namespace MPCSolute
{

template<typename PositionCoordinate, typename VelocityCoordinate>
Base<PositionCoordinate, VelocityCoordinate>::Base()
	: mdTimeStepSize(0),
	  instrumentation(NULL),
	  d_positions(NULL), d_velocities(NULL)
{
	deviceMemoryManager.setAutofree(true);
}

template<typename PositionCoordinate, typename VelocityCoordinate>
Base<PositionCoordinate, VelocityCoordinate>::~Base()
{
	delete instrumentation;
}

template<typename PositionCoordinate, typename VelocityCoordinate>
void Base<PositionCoordinate, VelocityCoordinate>::fetchFromDevice() const
{
	deviceMemoryManager.copyElementsFromDeviceToHost(
		d_positions, h_positions.get(), 3 * getParticleCount());
	deviceMemoryManager.copyElementsFromDeviceToHost(
		d_velocities, h_velocities.get(), 3 * getParticleCount());
}

template<typename PositionCoordinate, typename VelocityCoordinate>
const RemotelyStoredVector<const PositionCoordinate>
	Base<PositionCoordinate, VelocityCoordinate>::getPosition(
		const std::size_t particleID) const
{
	#ifdef OPENMPCD_DEBUG
		if(particleID >= getParticleCount())
			OPENMPCD_THROW(OutOfBoundsException, "particleID");
	#endif

	return
		RemotelyStoredVector<const PositionCoordinate>(
			h_positions.get(), particleID);
}

template<typename PositionCoordinate, typename VelocityCoordinate>
const RemotelyStoredVector<const VelocityCoordinate>
	Base<PositionCoordinate, VelocityCoordinate>::getVelocity(
		const std::size_t particleID) const
{
	#ifdef OPENMPCD_DEBUG
		if(particleID >= getParticleCount())
			OPENMPCD_THROW(OutOfBoundsException, "particleID");
	#endif

	return
		RemotelyStoredVector<const VelocityCoordinate>(
			h_velocities.get(), particleID);
}

template<typename PositionCoordinate, typename VelocityCoordinate>
void Base<PositionCoordinate, VelocityCoordinate>::pushToDevice()
{
	deviceMemoryManager.copyElementsFromHostToDevice(
		h_positions.get(), d_positions, 3 * getParticleCount());
	deviceMemoryManager.copyElementsFromHostToDevice(
		h_velocities.get(), d_velocities, 3 * getParticleCount());
}

} //namespace MPCSolute
} //namespace CUDA
} //namespace OpenMPCD

template class OpenMPCD::CUDA::MPCSolute::Base<double, double>;
