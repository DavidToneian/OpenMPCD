/**
 * @file
 * Defines the `OpenMPCD::MPI` class.
 */

#ifndef OPENMPCD_MPI_HPP
#define OPENMPCD_MPI_HPP

#include <cstddef>

#include <mpi.h>

namespace OpenMPCD
{

/**
 * Wraps MPI functionality.
 *
 * The first instance of this class that is constructed in a program will
 * initialize the MPI library, and will also finalize it once it is destructed.
 * After that point, no new instances of this class may be constructed, and
 * at that point, no other instances of this class may exist.
 *
 * All functions can throw `OpenMPCD::Exception` in case of an MPI error.
 */
class MPI
{
public:
	/**
	 * The constructor.
	 */
	MPI();

	/**
	 * The destructor.
	 */
	~MPI();

public:
	/**
	 * Returns the total number of processes.
	 */
	std::size_t getNumberOfProcesses() const;

	/**
	 * Returns the rank of the current process.
	 */
	std::size_t getRank() const;

	/**
	 * Implements the `MPI_BCAST` functionality.
	 *
	 * The MPI communicator used is `MPI_COMM_WORLD`.
	 *
	 * @throw OpenMPCD::NULLPointerException
	 *        If `OPENMPCD_DEBUG` is defined, throws if
	 *        `count != 0 && buffer == nullptr`.
	 * @throw OpenMPCD::OutOfBoundsException
	 *        If `OPENMPCD_DEBUG` is defined, throws if
	 *        `root >= getNumberOfProcesses()`.
	 *
	 * @tparam T The type of elements to broadcast.
	 *
	 * @param[in,out] buffer
	 *                If `root == getRank()`, this designates the buffer to send
	 *                from; otherwise, this designates the buffer to receive
	 *                into. In either case, the buffer must be able to hold at
	 *                least `count` elements, unless `count == 0`.
	 * @param[in]     count
	 *                The number of elements to broadcast. May be `0`, in which
	 *                case this function does nothing.
	 * @param[in]     root
	 *                The root process rank.
	 */
	template<typename T>
	void broadcast(
		T* const buffer, const std::size_t count,
		const std::size_t root) const;

	/**
	 * Implements the `MPI_BARRIER` functionality.
	 *
	 * The communicator is taken to be `MPI_COMM_WORLD`.
	 */
	void barrier();

	/**
	 * Calls `MPI_TEST` on the given request.
	 *
	 * @throw OpenMPCD::NULLPointerException
	 *        If `OPENMPCD_DEBUG` is defined, throws if `request == nullptr`.
	 * @throw OpenMPCD::InvalidArgumentException
	 *        If `OPENMPCD_DEBUG` is defined, throws if
	 *        `*request == MPI_REQUEST_NULL`.
	 *
	 * @param[in,out] request
	 *                The request to test; if completed, the pointee will be set
	 *                `MPI_REQUEST_NULL`.
	 * @param[out]    sender
	 *                If not `nullptr` and if the request is completed, the
	 *                pointee is set to the rank of the sender of the received
	 *                message, if such information is available.
	 *
	 * @return Returns true if the request has been completed, false otherwise.
	 */
	bool test(
		MPI_Request* const request,
		std::size_t* const sender = NULL) const;

private:
	bool initializedByThisInstance;
		///< Holds whether this instance initialized MPI.
}; //class MPI

} //namespace OpenMPCD

#include <OpenMPCD/ImplementationDetails/MPI.hpp>

#endif //OPENMPCD_MPI_HPP
