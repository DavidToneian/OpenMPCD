#include <OpenMPCD/MPI.hpp>

#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>
#include <OpenMPCD/Utility/CompilerDetection.hpp>

#include <limits>

namespace OpenMPCD
{

MPI::MPI()
	: initializedByThisInstance(false)
{
	int initialized;
	const int ret = MPI_Initialized(&initialized);

	if(ret != MPI_SUCCESS)
		OPENMPCD_THROW(Exception, "`MPI_Initialized`");

	if(!initialized)
	{
		int provided;
		const int ret =
			MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);

		if(ret != MPI_SUCCESS)
			OPENMPCD_THROW(Exception, "`MPI_Init_thread`");

		if(provided < MPI_THREAD_FUNNELED)
		{
			MPI_Finalize();

			OPENMPCD_THROW(
				Exception,
				"`MPI_Init_thread` failed to provide a thread-safe "
				"environment.");
		}

		initializedByThisInstance = true;
	}
}

MPI::~MPI()
{
	#if defined(OPENMPCD_COMPILER_GCC) && __GNUC__ >= 6
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Wterminate"
	#endif

	int finalized;
	const int ret = MPI_Finalized(&finalized);

	if(ret != MPI_SUCCESS)
		OPENMPCD_THROW(Exception, "`MPI_Finalized`");

	if(finalized)
	{
		if(initializedByThisInstance)
			OPENMPCD_THROW(
				Exception,
				"`MPI_Finalized` has been called by the wrong instance.");
	}
	else
	{
		if(initializedByThisInstance)
		{
			const int ret = MPI_Finalize();
			if(ret != MPI_SUCCESS)
				OPENMPCD_THROW(Exception, "`MPI_Finalize`");
		}
	}

	#if defined(OPENMPCD_COMPILER_GCC) && __GNUC__ >= 6
		#pragma GCC diagnostic push
	#endif
}

std::size_t MPI::getNumberOfProcesses() const
{
	int size;

	const int ret = MPI_Comm_size(MPI_COMM_WORLD, &size);
	if(ret != MPI_SUCCESS)
		OPENMPCD_THROW(Exception, "`MPI_Comm_size`");

	#ifdef OPENMPCD_COMPILER_GCC
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Wsign-compare"
	#endif
	OPENMPCD_DEBUG_ASSERT(size > 0);
	OPENMPCD_DEBUG_ASSERT(size <= std::numeric_limits<std::size_t>::max());
	#ifdef OPENMPCD_COMPILER_GCC
		#pragma GCC diagnostic push
	#endif

	return static_cast<std::size_t>(size);
}

std::size_t MPI::getRank() const
{
	int rank;

	const int ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	if(ret != MPI_SUCCESS)
		OPENMPCD_THROW(Exception, "`MPI_Comm_rank`");

	#ifdef OPENMPCD_COMPILER_GCC
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Wsign-compare"
	#endif
	OPENMPCD_DEBUG_ASSERT(rank >= 0);
	OPENMPCD_DEBUG_ASSERT(rank <= std::numeric_limits<std::size_t>::max());
	#ifdef OPENMPCD_COMPILER_GCC
		#pragma GCC diagnostic push
	#endif

	return static_cast<std::size_t>(rank);
}

void MPI::barrier()
{
	if(MPI_Barrier(MPI_COMM_WORLD) != MPI_SUCCESS)
		OPENMPCD_THROW(Exception, "`MPI_Barrier`");
}

bool MPI::test(
	MPI_Request* const request, std::size_t* const sender) const
{
	#ifdef OPENMPCD_DEBUG
		if(request == NULL)
			OPENMPCD_THROW(NULLPointerException, "`request`");

		if(*request == MPI_REQUEST_NULL)
			OPENMPCD_THROW(InvalidArgumentException, "`request`");
	#endif

	int completed;
	MPI_Status status;
	const int ret =	MPI_Test(request, &completed, &status);
	if(ret != MPI_SUCCESS)
		OPENMPCD_THROW(Exception, "`MPI_Test`");

	if(completed && sender != NULL)
	{
		#ifdef OPENMPCD_COMPILER_GCC
			#pragma GCC diagnostic push
			#pragma GCC diagnostic ignored "-Wsign-compare"
		#endif
		OPENMPCD_DEBUG_ASSERT(status.MPI_SOURCE >= 0);
		OPENMPCD_DEBUG_ASSERT(
			status.MPI_SOURCE <= std::numeric_limits<std::size_t>::max());
		#ifdef OPENMPCD_COMPILER_GCC
			#pragma GCC diagnostic push
		#endif

		*sender = static_cast<std::size_t>(status.MPI_SOURCE);
	}

	return completed;
}

} //namespace OpenMPCD
