/**
 * Tests the `OpenMPCD::MPI` class.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/MPI.hpp>

SCENARIO(
	"`OpenMPCD::MPI::MPI`",
	"[MPI]")
{
	WHEN("there is a single instance of `MPI`")
	{
		OpenMPCD::MPI mpi;
	}

	WHEN("there is an instance of `MPI`")
	{
		OpenMPCD::MPI mpi;

		THEN("another one can be constructed")
		{
			OpenMPCD::MPI mpi_;
		}
	}
}

SCENARIO(
	"`OpenMPCD::MPI::getRank`",
	"[MPI]")
{
	OpenMPCD::MPI mpi;

	REQUIRE(mpi.getRank() < mpi.getNumberOfProcesses());
}

SCENARIO(
	"`OpenMPCD::MPI::broadcast`",
	"[MPI]")
{
	OpenMPCD::MPI mpi;

	#ifdef OPENMPCD_DEBUG
		WHEN("Invalid arguments are passed to `broadcast`")
		{
			THEN("an exception is thrown")
			{
				int x;

				REQUIRE_THROWS_AS(
					mpi.broadcast<int>(NULL, 1, 0),
					OpenMPCD::NULLPointerException);

				REQUIRE_THROWS_AS(
					mpi.broadcast(&x, 1, mpi.getNumberOfProcesses()),
					OpenMPCD::OutOfBoundsException);
				REQUIRE_THROWS_AS(
					mpi.broadcast(&x, 1, mpi.getNumberOfProcesses() + 1),
					OpenMPCD::OutOfBoundsException);
			}
		}
	#endif

	WHEN("`broadcast` is called with `count == 0`")
	{
		THEN("nothing happens")
		{
			int x;
			mpi.broadcast(&x, 0, 0);

			REQUIRE_NOTHROW(mpi.broadcast<int>(NULL, 0, 0));
		}
	}

	WHEN("`broadcast` is called with one element")
	{
		THEN("the data is broadcast accordingly")
		{
			for(std::size_t rank = 0; rank < mpi.getNumberOfProcesses(); ++rank)
			{
				std::size_t buffer[3] = {0, 0, 0};
				buffer[0] = mpi.getRank() + 1;

				mpi.broadcast(buffer, 1, rank);

				REQUIRE(buffer[0] == rank + 1);
				REQUIRE(buffer[1] == 0);
				REQUIRE(buffer[2] == 0);
			}
		}
	}

	WHEN("`broadcast` is called with two elements")
	{
		THEN("the data is broadcast accordingly")
		{
			for(std::size_t rank = 0; rank < mpi.getNumberOfProcesses(); ++rank)
			{
				std::size_t buffer[3] = {0, 0, 0};
				buffer[0] = mpi.getRank() + 1;
				buffer[1] = buffer[0] * 2;

				mpi.broadcast(buffer, 2, rank);

				REQUIRE(buffer[0] == rank + 1);
				REQUIRE(buffer[1] == buffer[0] * 2);
				REQUIRE(buffer[2] == 0);
			}
		}
	}
}


SCENARIO(
	"`OpenMPCD::MPI::barrier`",
	"[MPI]")
{
	OpenMPCD::MPI mpi;

	REQUIRE_NOTHROW(mpi.barrier());
}


SCENARIO(
	"`OpenMPCD::MPI::test`",
	"[MPI]")
{
	OpenMPCD::MPI mpi;

	#ifdef OPENMPCD_DEBUG
		WHEN("Invalid arguments are passed to `test`")
		{
			THEN("an exception is thrown")
			{
				REQUIRE_THROWS_AS(
					mpi.test(NULL),
					OpenMPCD::NULLPointerException);

				REQUIRE_THROWS_AS(
					mpi.test(NULL, NULL),
					OpenMPCD::NULLPointerException);

				std::size_t tmp;
				REQUIRE_THROWS_AS(
					mpi.test(NULL, &tmp),
					OpenMPCD::NULLPointerException);

				MPI_Request req = MPI_REQUEST_NULL;
				REQUIRE_THROWS_AS(
					mpi.test(&req),
					OpenMPCD::InvalidArgumentException);
			}
		}
	#endif


	GIVEN("open send and receive requests to oneself")
	{
		bool tmp;
		MPI_Request requestSend;
		MPI_Request requestReceive;

		REQUIRE(
			MPI_Isend(
				&tmp, sizeof(tmp), MPI_BYTE, mpi.getRank(),
				0, MPI_COMM_WORLD, &requestSend) == MPI_SUCCESS);

		REQUIRE(
			MPI_Irecv(
				&tmp, sizeof(tmp), MPI_BYTE, MPI_ANY_SOURCE,
				0, MPI_COMM_WORLD, &requestReceive) == MPI_SUCCESS);

		THEN("one can `test` them both, and get sender info")
		{
			std::size_t sender;

			while(!mpi.test(&requestSend));
			while(!mpi.test(&requestReceive, &sender));

			REQUIRE(sender == mpi.getRank());
		}
	}

	GIVEN("open send and receive requests to someone else")
	{
		bool tmp;
		MPI_Request requestSend;
		MPI_Request requestReceive;

		const int sendTo = (mpi.getRank() + 1) % mpi.getNumberOfProcesses();

		REQUIRE(
			MPI_Isend(
				&tmp, sizeof(tmp), MPI_BYTE, sendTo,
				1, MPI_COMM_WORLD, &requestSend) == MPI_SUCCESS);

		REQUIRE(
			MPI_Irecv(
				&tmp, sizeof(tmp), MPI_BYTE, MPI_ANY_SOURCE,
				1, MPI_COMM_WORLD, &requestReceive) == MPI_SUCCESS);

		THEN("one can `test` them both, and get sender info")
		{
			std::size_t sender;

			while(!mpi.test(&requestSend));
			while(!mpi.test(&requestReceive, &sender));

			if(mpi.getRank() == 0)
			{
				REQUIRE(sender == mpi.getNumberOfProcesses() - 1);
			}
			else
			{
				REQUIRE(sender == mpi.getRank() - 1);
			}
		}
	}

	mpi.barrier();
}
