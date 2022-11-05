/**
 * @file
 * Implements functionality of `OpenMPCD::MPI`.
 */

#ifndef OPENMPCD_IMPLEMENTATION_MPI_HPP
#define OPENMPCD_IMPLEMENTATION_MPI_HPP

#include <OpenMPCD/MPI.hpp>

#include <OpenMPCD/Exceptions.hpp>

#include <limits>

namespace OpenMPCD
{

template<typename T>
void MPI::broadcast(
	T* const buffer, const std::size_t count, const std::size_t root) const
{
	#ifdef OPENMPCD_DEBUG
		if(count != 0 && buffer == NULL)
			OPENMPCD_THROW(NULLPointerException, "`buffer`");

		if(root >= getNumberOfProcesses())
			OPENMPCD_THROW(OutOfBoundsException, "`root`");
	#endif

	if(count == 0)
		return;

	const int ret =
		MPI_Bcast(buffer, count * sizeof(T), MPI_BYTE, root, MPI_COMM_WORLD);
	if(ret != MPI_SUCCESS)
		OPENMPCD_THROW(Exception, "`MPI_Bcast`");
}

} //namespace OpenMPCD

#endif //OPENMPCD_IMPLEMENTATION_MPI_HPP
