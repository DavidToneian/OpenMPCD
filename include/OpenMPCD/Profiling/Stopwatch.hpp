/**
 * @file
 * Defines the `OpenMPCD::Profiling::Stopwatch` class.
 */

#ifndef OPENMPCD_PROFILING_STOPWATCH_HPP
#define OPENMPCD_PROFILING_STOPWATCH_HPP

#include <boost/chrono.hpp>

namespace OpenMPCD
{
namespace Profiling
{

/**
 * Marks a code region.
 * The marked region starts with the construction of an instance of this class,
 * and ends with its destruction.
 */
class Stopwatch
{
public:
	typedef boost::chrono::microseconds::rep MicrosecondDuration;
		/**< An integral type that represents a duration of time in terms of
	         microseconds.*/

public:
	/**
	 * The constructor.
	 *
	 * Upon construction, the stopwatch starts measuring time.
	 */
	Stopwatch()
		: startingTime(boost::chrono::high_resolution_clock::now())
	{
	}

private:
	Stopwatch(const Stopwatch&); ///< The copy constructor.

public:
	/**
	 * The destructor.
	 */
	~Stopwatch()
	{
	}

public:
	/**
	 * Returns the number of microseconds that have elapsed since the stopwatch
	 * started measuring.
	 */
	const MicrosecondDuration getElapsedMicroseconds() const
	{
		const boost::chrono::high_resolution_clock::time_point now =
			boost::chrono::high_resolution_clock::now();
		const boost::chrono::microseconds m =
			boost::chrono::duration_cast<boost::chrono::microseconds>(
				now - startingTime);

		return m.count();
	}

private:
	const Stopwatch& operator=(const Stopwatch&); ///< The assignment operator.

private:
	const boost::chrono::high_resolution_clock::time_point startingTime;
		///< The point in time the stopwatch started measuring.
}; //class Stopwatch
} //namespace Profiling
} //namespace OpenMPCD

#endif //OPENMPCD_PROFILING_STOPWATCH_HPP
