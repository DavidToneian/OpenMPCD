/**
 * @file
 * Implements functionality of `OpenMPCD::OnTheFlyStatistics`.
 */

#ifndef OPENMPCD_IMPLEMENTATIONDETAILS_ONTHEFLYSTATISTICS_HPP
#define OPENMPCD_IMPLEMENTATIONDETAILS_ONTHEFLYSTATISTICS_HPP

#include <OpenMPCD/OnTheFlyStatistics.hpp>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/is_arithmetic.hpp>

#include <sstream>
#include <vector>

namespace OpenMPCD
{

template<typename T>
const std::string OnTheFlyStatistics<T>::serializeToString() const
{
	BOOST_STATIC_ASSERT(boost::is_arithmetic<T>::value);

	std::stringstream ss;

	ss.precision(std::numeric_limits<T>::digits10 + 2);

	ss << "1;"; //format version
	ss << sampleSize << ";";
	ss << mean << ";";
	ss << varianceHelper;

	return ss.str();
}

template<typename T>
void OnTheFlyStatistics<T>::unserializeFromString(const std::string& state)
{
	BOOST_STATIC_ASSERT(boost::is_arithmetic<T>::value);

	std::vector<std::string> parts;
	boost::algorithm::split(parts, state, boost::algorithm::is_any_of(";"));

	try
	{
		if(parts[0][0] == '-')
			OPENMPCD_THROW(InvalidArgumentException, "");
		const unsigned int version =
			boost::lexical_cast<unsigned int>(parts[0]);

		if(version != 1)
			OPENMPCD_THROW(InvalidArgumentException, "Unknown version");

		if(parts.size() != 4)
			OPENMPCD_THROW(InvalidArgumentException, "");

		if(parts[1][0] == '-')
			OPENMPCD_THROW(InvalidArgumentException, "");

		const std::size_t _sampleSize =
			boost::lexical_cast<std::size_t>(parts[1]);
		const T _mean = boost::lexical_cast<T>(parts[2]);
		const T _varianceHelper = boost::lexical_cast<T>(parts[3]);

		if(_sampleSize == 0)
		{
			if(_mean != 0 || _varianceHelper != 0)
				OPENMPCD_THROW(InvalidArgumentException, "");
		}
		if(_varianceHelper < 0)
			OPENMPCD_THROW(InvalidArgumentException, "");

		sampleSize = _sampleSize;
		mean = _mean;
		varianceHelper = _varianceHelper;
	}
	catch(const boost::bad_lexical_cast&)
	{
		OPENMPCD_THROW(InvalidArgumentException, "");
	}
}

} //namespace OpenMPCD

#endif //OPENMPCD_IMPLEMENTATIONDETAILS_ONTHEFLYSTATISTICS_HPP
