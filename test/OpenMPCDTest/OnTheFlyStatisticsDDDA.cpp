/**
 * @file
 * Tests functionality of `OpenMPCD::OnTheFlyStatisticsDDDA`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/OnTheFlyStatisticsDDDA.hpp>

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>


#define SEQ_DATATYPE (float)(double)

template<typename T>
void test_emptyInstance()
{
	OpenMPCD::OnTheFlyStatisticsDDDA<T> ddda;

	REQUIRE(ddda.getSampleSize() == 0);

	REQUIRE_THROWS_AS(
		ddda.getSampleMean(), OpenMPCD::InvalidCallException);
	REQUIRE_THROWS_AS(
		ddda.getMaximumBlockSize(), OpenMPCD::InvalidCallException);

	REQUIRE(ddda.getMaximumBlockID() == 0);
}
SCENARIO(
	"`OpenMPCD::OnTheFlyStatisticsDDDA` empty instance",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_emptyInstance<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}


template<typename T>
void test_oneDatum()
{
	OpenMPCD::OnTheFlyStatisticsDDDA<T> ddda;

	ddda.addDatum(5);

	REQUIRE(ddda.getSampleSize() == 1);
	REQUIRE(ddda.getSampleMean() == 5);
	REQUIRE(ddda.getMaximumBlockSize() == 1);
	REQUIRE(ddda.getMaximumBlockID() == 0);
}
SCENARIO(
	"`OpenMPCD::OnTheFlyStatisticsDDDA` with one datum",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_oneDatum<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}


template<typename T>
void test_twoData()
{
	OpenMPCD::OnTheFlyStatisticsDDDA<T> ddda;

	ddda.addDatum(5);
	ddda.addDatum(2);

	REQUIRE(ddda.getSampleSize() == 2);
	REQUIRE(ddda.getSampleMean() == 7.0 / 2);
	REQUIRE(ddda.getMaximumBlockSize() == 2);
	REQUIRE(ddda.getMaximumBlockID() == 1);
}
SCENARIO(
	"`OpenMPCD::OnTheFlyStatisticsDDDA` with two data",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_twoData<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}


template<typename T>
void test_threeData()
{
	OpenMPCD::OnTheFlyStatisticsDDDA<T> ddda;

	ddda.addDatum(5);
	ddda.addDatum(2);
	ddda.addDatum(-1.2);

	REQUIRE(ddda.getSampleSize() == 3);
	REQUIRE(ddda.getSampleMean() == Approx(5.8 / 3));
	REQUIRE(ddda.getMaximumBlockSize() == 2);
	REQUIRE(ddda.getMaximumBlockID() == 1);
}
SCENARIO(
	"`OpenMPCD::OnTheFlyStatisticsDDDA` with three data",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_threeData<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}



template<typename T>
void test_fourData()
{
	OpenMPCD::OnTheFlyStatisticsDDDA<T> ddda;

	ddda.addDatum(5);
	ddda.addDatum(2);
	ddda.addDatum(-1.2);
	ddda.addDatum(40);

	REQUIRE(ddda.getSampleSize() == 4);
	REQUIRE(ddda.getSampleMean() == Approx(45.8 / 4));
	REQUIRE(ddda.getMaximumBlockSize() == 4);
	REQUIRE(ddda.getMaximumBlockID() == 2);
}
SCENARIO(
	"`OpenMPCD::OnTheFlyStatisticsDDDA` with four data",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_fourData<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}



template<typename T>
void test_dynamicData()
{
	static const std::size_t dataSizes[] =
		{
			1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		};

	boost::mt11213b rng;
	boost::random::normal_distribution<T> dist(0, 1);
	for(std::size_t ds = 0; ds < sizeof(dataSizes)/sizeof(dataSizes[0]); ++ds)
	{
		const std::size_t dataSize = dataSizes[ds];

		OpenMPCD::OnTheFlyStatisticsDDDA<T> ddda;
		std::vector<T> data;
		data.reserve(dataSize);

		T sum = 0;
		for(std::size_t i = 0; i < dataSize; ++i)
		{
			const T datum = dist(rng);
			data.push_back(datum);

			ddda.addDatum(datum);

			sum += datum;
		}

		std::size_t blockSizeCount = 1;
		std::size_t maximumBlockSize = 1;
		for(;;)
		{
			if(maximumBlockSize * 2 > dataSize)
				break;
			blockSizeCount += 1;
			maximumBlockSize *= 2;
		}

		REQUIRE(ddda.getSampleSize() == dataSize);
		REQUIRE(ddda.getSampleMean() == Approx(sum / dataSize));
		REQUIRE(ddda.getMaximumBlockSize() == maximumBlockSize);
		REQUIRE(ddda.getMaximumBlockID() == blockSizeCount - 1);
	}
}
SCENARIO(
	"`OpenMPCD::OnTheFlyStatisticsDDDA` with dynamically generated data",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_dynamicData<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}



template<typename T>
void test_hasBlockVariance_getBlockVariance()
{
	static const std::size_t dataSizes[] =
		{
			1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		};

	boost::mt11213b rng;
	boost::random::normal_distribution<T> dist(0, 1);
	for(std::size_t ds = 0; ds < sizeof(dataSizes)/sizeof(dataSizes[0]); ++ds)
	{
		const std::size_t dataSize = dataSizes[ds];

		OpenMPCD::OnTheFlyStatisticsDDDA<T> ddda;
		std::vector<T> data;
		data.reserve(dataSize);

		for(std::size_t i = 0; i < dataSize; ++i)
		{
			const T datum = dist(rng);
			data.push_back(datum);

			ddda.addDatum(datum);
		}

		#ifdef OPENMPCD_DEBUG
			REQUIRE_THROWS_AS(
				ddda.hasBlockVariance(ddda.getMaximumBlockID() + 1),
				OpenMPCD::InvalidArgumentException);
			REQUIRE_THROWS_AS(
				ddda.getBlockVariance(ddda.getMaximumBlockID() + 1),
				OpenMPCD::InvalidArgumentException);
		#endif

		for(std::size_t block = 0; block < ddda.getMaximumBlockID(); ++block)
		{
			const std::size_t blockSize = std::size_t(1) << block;

			if(dataSize / blockSize < 2)
			{
				REQUIRE_FALSE(ddda.hasBlockVariance(block));
				REQUIRE_THROWS_AS(
					ddda.getBlockVariance(block),
					OpenMPCD::InvalidCallException);
			}
			else
			{
				OpenMPCD::OnTheFlyStatistics<T> stat;
				OpenMPCD::OnTheFlyStatistics<T> tmp;
				for(std::size_t d = 0; d < dataSize; ++d)
				{
					tmp.addDatum(data[d]);

					if(tmp.getSampleSize() == blockSize)
					{
						stat.addDatum(tmp.getSampleMean());
						tmp = OpenMPCD::OnTheFlyStatistics<T>();
					}
				}

				const T expected = stat.getSampleVariance();

				REQUIRE(ddda.hasBlockVariance(block));
				REQUIRE(ddda.getBlockVariance(block) == Approx(expected));
			}
		}
	}
}
SCENARIO(
	"`OpenMPCD::OnTheFlyStatisticsDDDA::hasBlockVariance`, "
		"`OpenMPCD::OnTheFlyStatisticsDDDA::getBlockVariance`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_hasBlockVariance_getBlockVariance<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}


template<typename T>
void test_getBlockStandardDeviation()
{
	static const std::size_t dataSizes[] =
		{
			1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		};

	boost::mt11213b rng;
	boost::random::normal_distribution<T> dist(0, 1);
	for(std::size_t ds = 0; ds < sizeof(dataSizes)/sizeof(dataSizes[0]); ++ds)
	{
		const std::size_t dataSize = dataSizes[ds];

		OpenMPCD::OnTheFlyStatisticsDDDA<T> ddda;

		for(std::size_t i = 0; i < dataSize; ++i)
			ddda.addDatum(dist(rng));

		#ifdef OPENMPCD_DEBUG
			REQUIRE_THROWS_AS(
				ddda.getBlockStandardDeviation(ddda.getMaximumBlockID() + 1),
				OpenMPCD::InvalidArgumentException);
		#endif

		for(std::size_t block = 0; block < ddda.getMaximumBlockID(); ++block)
		{
			if(!ddda.hasBlockVariance(block))
			{
				REQUIRE_THROWS_AS(
					ddda.getBlockStandardDeviation(block),
					OpenMPCD::InvalidCallException);

				continue;
			}

			const T expected = sqrt(ddda.getBlockVariance(block));

			REQUIRE(ddda.getBlockStandardDeviation(block) == expected);
		}
	}
}
SCENARIO(
	"`OpenMPCD::OnTheFlyStatisticsDDDA::getBlockStandardDeviation`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_getBlockStandardDeviation<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}


template<typename T>
void test_getSampleStandardDeviation()
{
	static const std::size_t dataSizes[] =
		{
			1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		};

	boost::mt11213b rng;
	boost::random::normal_distribution<T> dist(0, 1);
	for(std::size_t ds = 0; ds < sizeof(dataSizes)/sizeof(dataSizes[0]); ++ds)
	{
		const std::size_t dataSize = dataSizes[ds];

		OpenMPCD::OnTheFlyStatisticsDDDA<T> ddda;

		for(std::size_t i = 0; i < dataSize; ++i)
			ddda.addDatum(dist(rng));

		if(!ddda.hasBlockVariance(0))
		{
			REQUIRE_THROWS_AS(
				ddda.getBlockStandardDeviation(0),
				OpenMPCD::InvalidCallException);

			continue;
		}

		const T expected = ddda.getBlockStandardDeviation(0);

		REQUIRE(ddda.getSampleStandardDeviation() == expected);
	}
}
SCENARIO(
	"`OpenMPCD::OnTheFlyStatisticsDDDA::getSampleStandardDeviation`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_getSampleStandardDeviation<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}


template<typename T>
void test_getBlockStandardErrorOfTheMean()
{
	static const std::size_t dataSizes[] =
		{
			1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		};

	boost::mt11213b rng;
	boost::random::normal_distribution<T> dist(0, 1);
	for(std::size_t ds = 0; ds < sizeof(dataSizes)/sizeof(dataSizes[0]); ++ds)
	{
		const std::size_t dataSize = dataSizes[ds];

		OpenMPCD::OnTheFlyStatisticsDDDA<T> ddda;
		std::vector<T> data;
		data.reserve(dataSize);

		for(std::size_t i = 0; i < dataSize; ++i)
		{
			const T datum = dist(rng);
			data.push_back(datum);

			ddda.addDatum(datum);
		}

		#ifdef OPENMPCD_DEBUG
			REQUIRE_THROWS_AS(
				ddda.getBlockStandardErrorOfTheMean(
					ddda.getMaximumBlockID() + 1),
				OpenMPCD::InvalidArgumentException);
		#endif

		for(std::size_t block = 0; block < ddda.getMaximumBlockID(); ++block)
		{
			const std::size_t blockSize = std::size_t(1) << block;

			if(ddda.hasBlockVariance(block))
			{
				OpenMPCD::OnTheFlyStatistics<T> stat;
				OpenMPCD::OnTheFlyStatistics<T> tmp;
				for(std::size_t d = 0; d < dataSize; ++d)
				{
					tmp.addDatum(data[d]);

					if(tmp.getSampleSize() == blockSize)
					{
						stat.addDatum(tmp.getSampleMean());
						tmp = OpenMPCD::OnTheFlyStatistics<T>();
					}
				}

				const T expected = stat.getStandardErrorOfTheMean();

				REQUIRE(
					ddda.getBlockStandardErrorOfTheMean(block)
					==
					Approx(expected));
			}
			else
			{
				REQUIRE_THROWS_AS(
					ddda.getBlockStandardErrorOfTheMean(block),
					OpenMPCD::InvalidCallException);
			}
		}
	}
}
SCENARIO(
	"`OpenMPCD::OnTheFlyStatisticsDDDA::getBlockStandardErrorOfTheMean`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_getBlockStandardErrorOfTheMean<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}



template<typename T>
void test_getEstimatedStandardDeviationOfBlockStandardErrorOfTheMean()
{
	static const std::size_t dataSizes[] =
		{
			1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		};

	boost::mt11213b rng;
	boost::random::normal_distribution<T> dist(0, 1);
	for(std::size_t ds = 0; ds < sizeof(dataSizes)/sizeof(dataSizes[0]); ++ds)
	{
		const std::size_t dataSize = dataSizes[ds];

		OpenMPCD::OnTheFlyStatisticsDDDA<T> ddda;
		std::vector<T> data;
		data.reserve(dataSize);

		for(std::size_t i = 0; i < dataSize; ++i)
		{
			const T datum = dist(rng);
			data.push_back(datum);

			ddda.addDatum(datum);
		}

		#ifdef OPENMPCD_DEBUG
			REQUIRE_THROWS_AS(
				ddda.getEstimatedStandardDeviationOfBlockStandardErrorOfTheMean(
					ddda.getMaximumBlockID() + 1),
				OpenMPCD::InvalidArgumentException);
		#endif

		for(std::size_t block = 0; block < ddda.getMaximumBlockID(); ++block)
		{
			const std::size_t blockSize = std::size_t(1) << block;

			if(ddda.hasBlockVariance(block))
			{
				OpenMPCD::OnTheFlyStatistics<T> stat;
				OpenMPCD::OnTheFlyStatistics<T> tmp;
				for(std::size_t d = 0; d < dataSize; ++d)
				{
					tmp.addDatum(data[d]);

					if(tmp.getSampleSize() == blockSize)
					{
						stat.addDatum(tmp.getSampleMean());
						tmp = OpenMPCD::OnTheFlyStatistics<T>();
					}
				}

				const T sem = ddda.getBlockStandardErrorOfTheMean(block);
				const std::size_t reducedSampleSize = dataSize / blockSize;
				const T expected = sem / sqrt(2 * reducedSampleSize);

				REQUIRE(
					ddda.getEstimatedStandardDeviationOfBlockStandardErrorOfTheMean(block)
					==
					Approx(expected));
			}
			else
			{
				REQUIRE_THROWS_AS(
					ddda.getEstimatedStandardDeviationOfBlockStandardErrorOfTheMean(
						block),
					OpenMPCD::InvalidCallException);
			}
		}
	}
}
SCENARIO(
	"`OpenMPCD::OnTheFlyStatisticsDDDA::"
		"getEstimatedStandardDeviationOfBlockStandardErrorOfTheMean`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_getEstimatedStandardDeviationOfBlockStandardErrorOfTheMean<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}



template<typename T>
void test_optimal_standard_error()
{
	static const std::size_t dataSizes[] =
		{
			0,
			1,
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		};

	boost::mt11213b rng;
	boost::random::normal_distribution<T> dist(0, 1);
	for(std::size_t ds = 0; ds < sizeof(dataSizes)/sizeof(dataSizes[0]); ++ds)
	{
		const std::size_t dataSize = dataSizes[ds];

		OpenMPCD::OnTheFlyStatisticsDDDA<T> ddda;

		if(dataSize < 2)
		{
			REQUIRE_THROWS_AS(
				ddda.getOptimalBlockIDForStandardErrorOfTheMean(),
				OpenMPCD::InvalidCallException);
			REQUIRE_THROWS_AS(
				ddda.optimalStandardErrorOfTheMeanEstimateIsReliable(),
				OpenMPCD::InvalidCallException);
			REQUIRE_THROWS_AS(
				ddda.getOptimalStandardErrorOfTheMean(),
				OpenMPCD::InvalidCallException);

			continue;
		}

		std::vector<T> data;
		data.reserve(dataSize);

		OpenMPCD::OnTheFlyStatistics<T> stat;
		for(std::size_t i = 0; i < dataSize; ++i)
		{
			const T datum = dist(rng);
			data.push_back(datum);

			ddda.addDatum(datum);
			stat.addDatum(datum);
		}


		const std::size_t optimalBlockID =
			ddda.getOptimalBlockIDForStandardErrorOfTheMean();
		const std::size_t optimalBlockSize = std::size_t(1) << optimalBlockID;

		const T rawSE = stat.getStandardErrorOfTheMean();

		std::vector<OpenMPCD::OnTheFlyStatistics<T> > blocks;
		for(std::size_t block = 0; block < ddda.getMaximumBlockID(); ++block)
		{
			const std::size_t blockSize = std::size_t(1) << block;

			OpenMPCD::OnTheFlyStatistics<T> stat;
			OpenMPCD::OnTheFlyStatistics<T> tmp;
			for(std::size_t d = 0; d < dataSize; ++d)
			{
				tmp.addDatum(data[d]);

				if(tmp.getSampleSize() == blockSize)
				{
					stat.addDatum(tmp.getSampleMean());
					tmp = OpenMPCD::OnTheFlyStatistics<T>();
				}
			}
			blocks.push_back(stat);
		}



		const T expectedSE = blocks[optimalBlockID].getStandardErrorOfTheMean();

		REQUIRE(ddda.getOptimalStandardErrorOfTheMean() == Approx(expectedSE));

		if(ddda.optimalStandardErrorOfTheMeanEstimateIsReliable())
		{
			REQUIRE(optimalBlockSize < dataSize / 50.0);
		}
		else
		{
			REQUIRE(optimalBlockSize >= dataSize / 50.0);
		}

		std::vector<bool> criteria;
		const std::size_t maxBlockID = ddda.getMaximumBlockID();
		for(std::size_t blockID = 0; blockID < maxBlockID + 1; ++blockID)
		{
			if(!ddda.hasBlockVariance(blockID))
			{
				criteria.push_back(false);
				continue;
			}

			const T currentSE = blocks[blockID].getStandardErrorOfTheMean();
			const T quotient = currentSE / rawSE;
			const bool criterion =
				pow(2, blockID * 3) > 2 * dataSize * pow(quotient, 4);

			criteria.push_back(criterion);
		}

		REQUIRE(criteria.size() == maxBlockID + 1);
		for(std::size_t blockID = 0; blockID < maxBlockID + 1; ++blockID)
		{
			if(blockID < optimalBlockID - 1)
				continue;

			if(blockID == optimalBlockID - 1)
			{
				REQUIRE_FALSE(criteria[blockID]);
				continue;
			}

			if(blockID == ddda.getMaximumBlockID() - 1)
			{
				if(!ddda.hasBlockVariance(ddda.getMaximumBlockID()))
				{
					if(optimalBlockID == blockID)
						continue;
				}
			}

			if(blockID == ddda.getMaximumBlockID())
			{
				if(optimalBlockID == ddda.getMaximumBlockID())
				{
					continue;
				}
				else
				{
					if(!ddda.hasBlockVariance(ddda.getMaximumBlockID()))
						continue;
				}
			}

			REQUIRE(criteria[blockID]);
		}
	}
}
SCENARIO(
		"`OpenMPCD::OnTheFlyStatisticsDDDA::"
			"getOptimalBlockIDForStandardErrorOfTheMean`, "
		"`OpenMPCD::OnTheFlyStatisticsDDDA::"
			"optimalStandardErrorOfTheMeanEstimateIsReliable`, "
		"`OpenMPCD::OnTheFlyStatisticsDDDA::"
			"getOptimalStandardErrorOfTheMean`"
		,
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_optimal_standard_error<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}



template<typename T>
void test_getOptimalBlockIDForStandardErrorOfTheMean_zero_variance()
{
	static const std::size_t dataSizes[] =
		{
			2, 3,
			4, 5,
			7, 8, 9,
			15, 16, 17,
			31, 32, 33,
			63, 64, 65, 100
		};

	boost::mt11213b rng;
	boost::random::normal_distribution<T> dist(0, 1);
	for(std::size_t ds = 0; ds < sizeof(dataSizes)/sizeof(dataSizes[0]); ++ds)
	{
		const std::size_t dataSize = dataSizes[ds];

		OpenMPCD::OnTheFlyStatisticsDDDA<T> ddda;

		const T datum = dist(rng);
		for(std::size_t i = 0; i < dataSize; ++i)
			ddda.addDatum(datum);

		REQUIRE(ddda.getOptimalBlockIDForStandardErrorOfTheMean() == 0);
	}
}
SCENARIO(
	"`OpenMPCD::OnTheFlyStatisticsDDDA::"
		"getOptimalBlockIDForStandardErrorOfTheMean` with zero variance",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_getOptimalBlockIDForStandardErrorOfTheMean_zero_variance<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}


template<typename T>
void test_serializeToString_unserializeFromString()
{
	class Helper
	{
		public:
		static bool approximatelyEqual(
			OpenMPCD::OnTheFlyStatisticsDDDA<T>& lhs,
			OpenMPCD::OnTheFlyStatisticsDDDA<T>& rhs)
		{
			if(lhs.getSampleSize() != rhs.getSampleSize())
				return false;

			if(lhs.getSampleSize() == 0)
				return true;

			if(lhs.getSampleMean() != Approx(rhs.getSampleMean()))
				return false;

			if(lhs.getMaximumBlockID() != rhs.getMaximumBlockID())
				return false;

			for(std::size_t i = 0; i < lhs.getMaximumBlockID() + 1; ++i)
			{
				if(lhs.hasBlockVariance(i) != rhs.hasBlockVariance(i))
					return false;
				if(!lhs.hasBlockVariance(i))
					continue;

				if(lhs.getBlockVariance(i) != Approx(rhs.getBlockVariance(i)))
					return false;
			}

			return true;
		}
	};
	OpenMPCD::OnTheFlyStatisticsDDDA<T> stat;

	REQUIRE_THROWS_AS(
		stat.unserializeFromString(""),
		OpenMPCD::InvalidArgumentException);
	REQUIRE_THROWS_AS(
		stat.unserializeFromString("foo"),
		OpenMPCD::InvalidArgumentException);
	REQUIRE_THROWS_AS(
		stat.unserializeFromString("123|0"),
		OpenMPCD::InvalidArgumentException);
	REQUIRE_THROWS_AS(
		stat.unserializeFromString("1;0;0;0"),
		OpenMPCD::InvalidArgumentException);
	REQUIRE_THROWS_AS(
		stat.unserializeFromString("1|-1"),
		OpenMPCD::InvalidArgumentException);
	REQUIRE_THROWS_AS(
		stat.unserializeFromString("-1|0"),
		OpenMPCD::InvalidArgumentException);
	REQUIRE_THROWS_AS(
		stat.unserializeFromString("foo|0"),
		OpenMPCD::InvalidArgumentException);


	OpenMPCD::OnTheFlyStatisticsDDDA<T> empty;

	REQUIRE(Helper::approximatelyEqual(stat, empty));

	stat.unserializeFromString("1|0");
	REQUIRE(Helper::approximatelyEqual(stat, empty));

	stat.unserializeFromString(stat.serializeToString());
	REQUIRE(Helper::approximatelyEqual(stat, empty));

	{
		OpenMPCD::OnTheFlyStatisticsDDDA<T> unserialized;
		unserialized.unserializeFromString(stat.serializeToString());
		REQUIRE(Helper::approximatelyEqual(stat, empty));
	}


	boost::mt11213b rng;
	boost::random::normal_distribution<T> dist(0, 1);
	boost::random::uniform_int_distribution<std::size_t> range(1, 5);
	for(std::size_t i = 0; i < 50; ++i)
	{
		const std::size_t maxJ = range(rng);
		for(std::size_t j = 0; j < maxJ; ++j)
			stat.addDatum(dist(rng));

		OpenMPCD::OnTheFlyStatisticsDDDA<T> unserialized;
		if(maxJ == 1)
			unserialized.addDatum(0); //try with a non-empty instance

		for(std::size_t j = 0; j < 2; ++j)
		{
			unserialized.unserializeFromString(stat.serializeToString());
			REQUIRE(Helper::approximatelyEqual(stat, unserialized));
		}

		for(std::size_t j = 0; j < 50; ++j)
		{
			//test that the waiting samples are (un)serialized correctly
			const T datum = dist(rng);
			stat.addDatum(datum);
			unserialized.addDatum(datum);

			REQUIRE(Helper::approximatelyEqual(stat, unserialized));
		}
	}

	{
		OpenMPCD::OnTheFlyStatisticsDDDA<T> stat;
		OpenMPCD::OnTheFlyStatisticsDDDA<T> unserialized;
		stat.addDatum(1);
		stat.addDatum(2);
		stat.addDatum(3);
		stat.addDatum(4);
		stat.addDatum(5);

		OpenMPCD::OnTheFlyStatistics<T> block1;
		OpenMPCD::OnTheFlyStatistics<T> block2;
		OpenMPCD::OnTheFlyStatistics<T> block3;
		block1.addDatum(1);
		block1.addDatum(2);
		block1.addDatum(3);
		block1.addDatum(4);
		block1.addDatum(5);
		block2.addDatum((1 + 2) / 2.0);
		block2.addDatum((3 + 4) / 2.0);
		block3.addDatum(((1 + 2) / 2.0) + ((3 + 4) / 2.0));

		std::string myState;
		myState += "1|" "3";
		myState += "|" + block1.serializeToString();
		myState += "|" + block2.serializeToString();
		myState += "|" + block3.serializeToString();
		myState += "|" "5";
		myState += "|";
		myState += "|";

		unserialized.unserializeFromString(myState);
		REQUIRE(Helper::approximatelyEqual(stat, unserialized));
	}
}
SCENARIO(
	"`OpenMPCD::OnTheFlyStatisticsDDDA::serializeToString`, "
		"`OpenMPCD::OnTheFlyStatisticsDDDA::unserializeFromString`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_serializeToString_unserializeFromString<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}
