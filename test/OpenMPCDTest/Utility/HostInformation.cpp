/**
 * @file
 * Tests functionality in `OpenMPCD::Utility::HostInformation`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/Utility/HostInformation.hpp>

#include <boost/lexical_cast.hpp>

#include <ctime>

#ifndef _XOPEN_SOURCE
	#define _XOPEN_SOURCE 500
#endif
#include <limits.h>
#include <unistd.h>


SCENARIO(
	"`OpenMPCD::Utility::HostInformation::getHostname`",
	"")
{
	using OpenMPCD::Utility::HostInformation::getHostname;

	static const std::size_t bufferSizeWithoutNULL = HOST_NAME_MAX + 10;
	char buffer[bufferSizeWithoutNULL + 1] = {0};

	const int result = gethostname(buffer, bufferSizeWithoutNULL);

	REQUIRE(result == 0);

	REQUIRE(getHostname() == buffer);
}


SCENARIO(
	"`OpenMPCD::Utility::HostInformation::getCurrentUTCTimeAsString`",
	"")
{
	static const int secondsTolerance = 2;

	do
	{
		const std::string result =
			OpenMPCD::Utility::HostInformation::getCurrentUTCTimeAsString();

		const std::time_t t = std::time(0);
		const std::tm* const tm = std::gmtime(&t);

		REQUIRE(result.length() == std::string("YYYY-MM-DDTHH:MM:SS").length());

		for(std::size_t i = 0; i <= 3; ++i)
		{
			REQUIRE(result[i] >= '0');
			REQUIRE(result[i] <= '9');
		}
		REQUIRE(result[4] == '-');
		for(std::size_t i = 5; i <= 6; ++i)
		{
			REQUIRE(result[i] >= '0');
			REQUIRE(result[i] <= '9');
		}
		REQUIRE(result[7] == '-');
		for(std::size_t i = 8; i <= 9; ++i)
		{
			REQUIRE(result[i] >= '0');
			REQUIRE(result[i] <= '9');
		}
		REQUIRE(result[10] == 'T');
		for(std::size_t i = 11; i <= 12; ++i)
		{
			REQUIRE(result[i] >= '0');
			REQUIRE(result[i] <= '9');
		}
		REQUIRE(result[13] == ':');
		for(std::size_t i = 14; i <= 15; ++i)
		{
			REQUIRE(result[i] >= '0');
			REQUIRE(result[i] <= '9');
		}
		REQUIRE(result[16] == ':');
		for(std::size_t i = 17; i <= 18; ++i)
		{
			REQUIRE(result[i] >= '0');
			REQUIRE(result[i] <= '9');
		}


		const int year = boost::lexical_cast<int>(result.substr(0, 4));
		const int month = boost::lexical_cast<int>(result.substr(5, 2));
		const int day = boost::lexical_cast<int>(result.substr(8, 2));
		const int hour = boost::lexical_cast<int>(result.substr(11, 2));
		const int minute = boost::lexical_cast<int>(result.substr(14, 2));
		const int second = boost::lexical_cast<int>(result.substr(17, 2));

		if(tm->tm_sec > second)
		{
			//A second or more may have passed between the time measurements
			//above, but the minute should not have changed (unless the
			//difference in time was a minute or longer, which should not happen
			//usually). For simplicity, just repeat the test.


			REQUIRE(tm->tm_sec - second <= secondsTolerance);

			continue;
		}
		else if(tm->tm_sec < second)
		{
			//A second or more has passed, and the minute has wrapped around.
			//For simplicity, just repeat the test.

			const int difference = tm->tm_sec + 60 - second;

			REQUIRE(difference <= secondsTolerance);

			continue;
		}


		REQUIRE(year == 1900 + tm->tm_year);
		REQUIRE(month == 1 + tm->tm_mon);
		REQUIRE(day == tm->tm_mday);
		REQUIRE(hour == tm->tm_hour);
		REQUIRE(minute == tm->tm_min);
	} while(0);
}
