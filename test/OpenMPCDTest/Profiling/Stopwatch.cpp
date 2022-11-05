/**
 * @file
 * Tests functionality of `OpenMPCD::Profiling::Stopwatch`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/Profiling/Stopwatch.hpp>

#include <boost/chrono.hpp>
#include <boost/thread/thread.hpp>


SCENARIO(
	"`OpenMPCD::Profiling::Stopwatch::getElapsedMicroseconds`",
	"")
{
	using OpenMPCD::Profiling::Stopwatch;


	const boost::chrono::high_resolution_clock::time_point startEarly =
		boost::chrono::high_resolution_clock::now();

	boost::this_thread::sleep_for(boost::chrono::microseconds(1));

	Stopwatch watch;

	boost::this_thread::sleep_for(boost::chrono::microseconds(1));

	const boost::chrono::high_resolution_clock::time_point startLate =
		boost::chrono::high_resolution_clock::now();

	boost::this_thread::sleep_for(boost::chrono::microseconds(1));

	const Stopwatch::MicrosecondDuration t1 = watch.getElapsedMicroseconds();

	boost::this_thread::sleep_for(boost::chrono::microseconds(1));

	const Stopwatch::MicrosecondDuration t2 = watch.getElapsedMicroseconds();

	boost::this_thread::sleep_for(boost::chrono::microseconds(1));

	const boost::chrono::high_resolution_clock::time_point stop =
			boost::chrono::high_resolution_clock::now();

	boost::this_thread::sleep_for(boost::chrono::microseconds(1));

	const Stopwatch::MicrosecondDuration t3 = watch.getElapsedMicroseconds();

	boost::this_thread::sleep_for(boost::chrono::microseconds(1));

	const Stopwatch::MicrosecondDuration t4 = watch.getElapsedMicroseconds();


	const boost::chrono::microseconds durationLong =
		boost::chrono::duration_cast<boost::chrono::microseconds>(
			stop - startEarly);
	const boost::chrono::microseconds durationShort =
		boost::chrono::duration_cast<boost::chrono::microseconds>(
			stop - startLate);

	REQUIRE(t1 > 0);
	REQUIRE(t2 > t1);
	REQUIRE(t3 > t2);
	REQUIRE(t4 > t3);

	REQUIRE(t2 < durationLong.count());
	REQUIRE(t3 > durationShort.count());
}
