/**
 * @file
 * Implements functionality of `OpenMPCD::OnTheFlyStatisticsDDDA`.
 */

#ifndef OPENMPCD_IMPLEMENTATIONDETAILS_ONTHEFLYSTATISTICSDDDA_HPP
#define OPENMPCD_IMPLEMENTATIONDETAILS_ONTHEFLYSTATISTICSDDDA_HPP

#include <OpenMPCD/Exceptions.hpp>
#include <OpenMPCD/OPENMPCD_DEBUG_ASSERT.hpp>
#include <OpenMPCD/OnTheFlyStatisticsDDDA.hpp>

#include <cmath>


namespace OpenMPCD
{

template<typename T>
OnTheFlyStatisticsDDDA<T>::OnTheFlyStatisticsDDDA()
{
	blocks.reserve(5);
	waiting.reserve(5);

	blocks.resize(1);
	waiting.resize(1);
}

template<typename T>
void OnTheFlyStatisticsDDDA<T>::addDatum(const T& datum)
{
	addDatum(datum, 0);
}

template<typename T>
std::size_t OnTheFlyStatisticsDDDA<T>::getSampleSize() const
{
	return blocks[0].getSampleSize();
}

template<typename T>
const T OnTheFlyStatisticsDDDA<T>::getSampleMean() const
{
	if(getSampleSize() == 0)
	{
		OPENMPCD_THROW(
			InvalidCallException,
			"Tried to get mean without having supplied data.");
	}

	return blocks[0].getSampleMean();
}

template<typename T>
std::size_t OnTheFlyStatisticsDDDA<T>::getMaximumBlockSize() const
{
	if(getSampleSize() == 0)
	{
		OPENMPCD_THROW(
			InvalidCallException,
			"Tried to get mean without having supplied data.");
	}

	return std::size_t(1) << getMaximumBlockID();
}

template<typename T>
std::size_t OnTheFlyStatisticsDDDA<T>::getMaximumBlockID() const
{
	OPENMPCD_DEBUG_ASSERT(blocks.size() > 0);

	return blocks.size() - 1;
}

template<typename T>
bool OnTheFlyStatisticsDDDA<T>::hasBlockVariance(
	const std::size_t blockID) const
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		blockID <= getMaximumBlockID(),
		InvalidArgumentException);

	return blocks[blockID].getSampleSize() >= 2;
}

template<typename T>
const T OnTheFlyStatisticsDDDA<T>::getBlockVariance(
	const std::size_t blockID) const
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		blockID <= getMaximumBlockID(),
		InvalidArgumentException);

	if(!hasBlockVariance(blockID))
		OPENMPCD_THROW(InvalidCallException, "!hasBlockVariance(blockID)");

	return blocks[blockID].getSampleVariance();
}

template<typename T>
const T OnTheFlyStatisticsDDDA<T>::getBlockStandardDeviation(
	const std::size_t blockID) const
{
	return sqrt(getBlockVariance(blockID));
}

template<typename T>
const T OnTheFlyStatisticsDDDA<T>::getSampleStandardDeviation() const
{
	return getBlockStandardDeviation(0);
}

template<typename T>
const T OnTheFlyStatisticsDDDA<T>::getBlockStandardErrorOfTheMean(
	const std::size_t blockID) const
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		blockID <= getMaximumBlockID(),
		InvalidArgumentException);

	if(!hasBlockVariance(blockID))
		OPENMPCD_THROW(InvalidCallException, "!hasBlockVariance(blockID)");

	return blocks[blockID].getStandardErrorOfTheMean();
}

template<typename T>
const T OnTheFlyStatisticsDDDA<T>::
	getEstimatedStandardDeviationOfBlockStandardErrorOfTheMean(
		const std::size_t blockID) const
{
	OPENMPCD_DEBUG_ASSERT_EXCEPTIONTYPE(
		blockID <= getMaximumBlockID(),
		InvalidArgumentException);

	if(!hasBlockVariance(blockID))
		OPENMPCD_THROW(InvalidCallException, "!hasBlockVariance(blockID)");

	const T se = getBlockStandardErrorOfTheMean(blockID);
	const std::size_t reducedSampleSize = blocks[blockID].getSampleSize();

	return se / sqrt(2 * reducedSampleSize);
}

template<typename T>
std::size_t
OnTheFlyStatisticsDDDA<T>::getOptimalBlockIDForStandardErrorOfTheMean() const
{
	if(getSampleSize() < 2)
		OPENMPCD_THROW(InvalidCallException, "Not enough data added.");

	const T rawStandardError = blocks[0].getStandardErrorOfTheMean();

	if(rawStandardError == 0)
		return 0;

	std::size_t optimalBlockID = getMaximumBlockID();

	for(std::size_t blockID = getMaximumBlockID(); ; --blockID)
	{
		if(!hasBlockVariance(blockID))
		{
			OPENMPCD_DEBUG_ASSERT(blockID == getMaximumBlockID());

			--optimalBlockID;
			continue;
		}

		const std::size_t blockSize = std::size_t(1) << blockID;
		const T blockedStandardError =
			blocks[blockID].getStandardErrorOfTheMean();
		const T quotient = blockedStandardError / rawStandardError;
		const T quotientPower2 = quotient * quotient;
		const T quotientPower4 = quotientPower2 * quotientPower2;

		const T threshold = 2 * getSampleSize() * quotientPower4;
		if(blockSize * blockSize * blockSize > threshold)
			optimalBlockID = blockID;

		if(blockID == 0)
			break;
	}

	return optimalBlockID;
}

template<typename T>
bool
OnTheFlyStatisticsDDDA<T>::optimalStandardErrorOfTheMeanEstimateIsReliable()
const
{
	const std::size_t blockID = getOptimalBlockIDForStandardErrorOfTheMean();
	const std::size_t blockSize = std::size_t(1) << blockID;

	return 50 * blockSize < getSampleSize();
}

template<typename T>
const T OnTheFlyStatisticsDDDA<T>::getOptimalStandardErrorOfTheMean() const
{
	const std::size_t blockID = getOptimalBlockIDForStandardErrorOfTheMean();

	return getBlockStandardErrorOfTheMean(blockID);
}

template<typename T>
void
OnTheFlyStatisticsDDDA<T>::addDatum(const T& datum, const std::size_t blockID)
{
	OPENMPCD_DEBUG_ASSERT(blockID < blocks.size());
	OPENMPCD_DEBUG_ASSERT(blocks.size() == waiting.size());

	blocks[blockID].addDatum(datum);

	if(waiting[blockID])
	{
		const T mean = (datum + *waiting[blockID]) / 2.0;

		waiting[blockID] = boost::none;
		if(blockID + 1 == blocks.size())
		{
			blocks.push_back(OnTheFlyStatistics<T>());
			waiting.push_back(boost::none);
		}

		addDatum(mean, blockID + 1);
	}
	else
	{
		waiting[blockID] = datum;
	}
}

template<typename T>
const std::string OnTheFlyStatisticsDDDA<T>::serializeToString() const
{
	BOOST_STATIC_ASSERT(boost::is_arithmetic<T>::value);

	std::stringstream ss;

	ss.precision(std::numeric_limits<T>::digits10 + 2);

	ss << "1|"; //format version
	ss << blocks.size();
	for(std::size_t i = 0; i < blocks.size(); ++i)
		ss << "|" << blocks[i].serializeToString();

	OPENMPCD_DEBUG_ASSERT(blocks.size() == waiting.size());
	for(std::size_t i = 0; i < blocks.size(); ++i)
	{
		ss << "|";
		if(waiting[i])
			ss << *waiting[i];
	}

	return ss.str();
}

template<typename T>
void OnTheFlyStatisticsDDDA<T>::unserializeFromString(const std::string& state)
{
	BOOST_STATIC_ASSERT(boost::is_arithmetic<T>::value);

	std::vector<std::string> parts;
	boost::algorithm::split(parts, state, boost::algorithm::is_any_of("|"));

	try
	{
		if(parts[0][0] == '-')
			OPENMPCD_THROW(InvalidArgumentException, "");
		const unsigned int version =
			boost::lexical_cast<unsigned int>(parts[0]);

		if(version != 1)
			OPENMPCD_THROW(InvalidArgumentException, "Unknown version");

		if(parts[1][0] == '-')
			OPENMPCD_THROW(InvalidArgumentException, "");
		const std::size_t blockCount =
			boost::lexical_cast<unsigned int>(parts[1]);

		if(parts.size() != 2 + 2 * blockCount)
			OPENMPCD_THROW(InvalidArgumentException, "");

		std::vector<OnTheFlyStatistics<T> > _blocks;
		for(std::size_t i = 0; i < blockCount; ++i)
		{
			OnTheFlyStatistics<T> block;
			block.unserializeFromString(parts[2 + i]);
			_blocks.push_back(block);
		}

		std::vector<boost::optional<T> > _waiting;
		for(std::size_t i = 0; i < blockCount; ++i)
		{
			boost::optional<T> w;
			const std::string& part = parts[2 + blockCount + i];
			if(part.size() > 0)
				w = boost::lexical_cast<T>(part);
			_waiting.push_back(w);
		}

		blocks = _blocks;
		waiting = _waiting;

		if(blockCount == 0)
		{
			blocks.resize(1);
			waiting.resize(1);
		}
	}
	catch(const boost::bad_lexical_cast&)
	{
		OPENMPCD_THROW(InvalidArgumentException, "");
	}
}

} //namespace OpenMPCD

#endif //OPENMPCD_IMPLEMENTATIONDETAILS_ONTHEFLYSTATISTICSDDDA_HPP
