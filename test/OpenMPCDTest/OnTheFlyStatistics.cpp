/**
 * @file
 * Tests functionality of `OpenMPCD::OnTheFlyStatistics`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/OnTheFlyStatistics.hpp>

#include <boost/math/distributions/chi_squared.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/uniform_real_distribution.hpp>

template<typename T>
static void test_OnTheFlyStatistics_do(const T* const data, const std::size_t N)
{
	OpenMPCD::OnTheFlyStatistics<T> stat;

	REQUIRE(stat.getSampleSize() == 0);
	REQUIRE(stat.getSampleMean() == T());
	REQUIRE(stat.getSampleVariance() == T());
	REQUIRE(stat.getSampleStandardDeviation() == sqrt(T()));

	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			stat.getStandardErrorOfTheMean(), OpenMPCD::InvalidCallException);
	#endif

	T mean = T();
	T varianceHelper = T();
	for(std::size_t i = 0; i < N; ++i)
	{
		stat.addDatum(data[i]);

		const T delta = data[i] - mean;

		mean += delta / (i + 1);
		varianceHelper += delta * (data[i] - mean);

		T variance = 0;
		if(stat.getSampleSize() >= 2)
			variance = varianceHelper / (stat.getSampleSize() - 1);

		REQUIRE(stat.getSampleSize() == i + 1);
		REQUIRE(stat.getSampleMean() == mean);
		REQUIRE(stat.getSampleVariance() == variance);
		REQUIRE(stat.getSampleStandardDeviation() == T(sqrt(variance)));
		REQUIRE(
			stat.getStandardErrorOfTheMean()
			==
			T(stat.getSampleStandardDeviation() / sqrt(stat.getSampleSize())));
	}
}
template<typename T>
static void test_OnTheFlyStatistics()
{
	static const std::size_t sampleSizes[] = {1, 10, 100};

	static const std::size_t ssi_num =
		sizeof(sampleSizes)/sizeof(sampleSizes[0]);
	static const std::size_t maxSampleSize = sampleSizes[ssi_num - 1];

	T data_scalar[maxSampleSize];

	boost::mt11213b rng;
	boost::random::uniform_real_distribution<T> dist(-1e8, 1e8);
	for(std::size_t i = 0; i < maxSampleSize; ++i)
	{
		data_scalar[i] = dist(rng);
	}

	for(std::size_t ssi = 0; ssi < ssi_num; ++ssi)
	{
		test_OnTheFlyStatistics_do(data_scalar, sampleSizes[ssi]);
	}
}

template<typename T>
static void test_OnTheFlyStatistics_BesselsCorrection()
{
	static const T sample[] = {-2, 0, 2};
	static const std::size_t sampleSize = sizeof(sample) / sizeof(sample[0]);

	static const T mean = 0;
	static const T unbiasedSampleVariance = 4;

	OpenMPCD::OnTheFlyStatistics<T> stat;
	for(std::size_t i = 0; i < sampleSize; ++i)
		stat.addDatum(sample[i]);

	REQUIRE(stat.getSampleMean() == 0);
	REQUIRE(stat.getSampleVariance() == unbiasedSampleVariance);
	REQUIRE(stat.getSampleStandardDeviation() == sqrt(unbiasedSampleVariance));
	REQUIRE(
		Approx(stat.getStandardErrorOfTheMean())
		==
		T(sqrt(unbiasedSampleVariance / sampleSize)));
}

template<typename T>
static void test_OnTheFlyStatistics_probabilistic()
{
	static const unsigned int sampleSize = 10000;
	static const T distMean = 0;
	static const T distStandardDeviation = 1;
	static const T significanceLevel = 0.005;

	OpenMPCD::OnTheFlyStatistics<T> stat;
	boost::mt11213b rng;
	boost::random::normal_distribution<T> dist(distMean, distStandardDeviation);

	for(unsigned int i = 0; i < sampleSize; ++i)
		stat.addDatum(dist(rng));

	REQUIRE(stat.getSampleSize() == sampleSize);

	const T sampleMean = stat.getSampleMean();
	const T sampleVariance = stat.getSampleVariance();

	{
		//test mean
		const T t_statistic =
			(sampleMean - distMean) / sqrt(sampleVariance / double(sampleSize));

		boost::math::students_t studentTDist(sampleSize - 1);
		const T tmp =
			boost::math::cdf(
				boost::math::complement(studentTDist, fabs(t_statistic)));

		REQUIRE(tmp > significanceLevel / 2.0);
	}

	{
		//test variance
		const T distVariance = distStandardDeviation * distStandardDeviation;
		const T statistic = (sampleSize - 1) * sampleVariance / distVariance;

		using namespace boost::math;

		chi_squared_distribution<T> chiSquaredDist(sampleSize - 1);

		const T upperThreshold =
			quantile(complement(chiSquaredDist, significanceLevel / 2.0));
		const T lowerThreshold =
			quantile(chiSquaredDist, significanceLevel / 2.0);

		REQUIRE(statistic < upperThreshold);
		REQUIRE(statistic > lowerThreshold);
	}
}

template<typename T>
static void test_OnTheFlyStatistics_serializeToString_unserializeFromString()
{
	OpenMPCD::OnTheFlyStatistics<T> stat;

	REQUIRE_THROWS_AS(
		stat.unserializeFromString(""),
		OpenMPCD::InvalidArgumentException);
	REQUIRE_THROWS_AS(
		stat.unserializeFromString("foo"),
		OpenMPCD::InvalidArgumentException);
	REQUIRE_THROWS_AS(
		stat.unserializeFromString("123;0;0.0;0.0"),
		OpenMPCD::InvalidArgumentException);

	REQUIRE_THROWS_AS(
		stat.unserializeFromString("1;0;1.0;0.0"),
		OpenMPCD::InvalidArgumentException);
	REQUIRE_THROWS_AS(
		stat.unserializeFromString("1;0;0.0;1.0"),
		OpenMPCD::InvalidArgumentException);
	REQUIRE_THROWS_AS(
		stat.unserializeFromString("1;-1;0.0;0.0"),
		OpenMPCD::InvalidArgumentException);
	REQUIRE_THROWS_AS(
		stat.unserializeFromString("1;2;0.0;-1.0"),
		OpenMPCD::InvalidArgumentException);
	REQUIRE_THROWS_AS(
		stat.unserializeFromString("1;0;0.0;0.0;0.0"),
		OpenMPCD::InvalidArgumentException);
	REQUIRE_THROWS_AS(
		stat.unserializeFromString("1;0;0.0;0.0;"),
		OpenMPCD::InvalidArgumentException);
	REQUIRE_THROWS_AS(
		stat.unserializeFromString("1;0;;0.0"),
		OpenMPCD::InvalidArgumentException);
	REQUIRE_THROWS_AS(
		stat.unserializeFromString("-1;0;0;0"),
		OpenMPCD::InvalidArgumentException);
	REQUIRE_THROWS_AS(
		stat.unserializeFromString("foo;0;0;0"),
		OpenMPCD::InvalidArgumentException);


	REQUIRE(stat.getSampleSize() == 0);
	REQUIRE(stat.getSampleMean() == 0);
	REQUIRE(stat.getSampleVariance() == 0);

	stat.unserializeFromString("1;0;0;0");
	REQUIRE(stat.getSampleSize() == 0);
	REQUIRE(stat.getSampleMean() == 0);
	REQUIRE(stat.getSampleVariance() == 0);

	stat.unserializeFromString(stat.serializeToString());
	REQUIRE(stat.getSampleSize() == 0);
	REQUIRE(stat.getSampleMean() == 0);
	REQUIRE(stat.getSampleVariance() == 0);

	{
		OpenMPCD::OnTheFlyStatistics<T> unserialized;
		unserialized.unserializeFromString(stat.serializeToString());
		REQUIRE(unserialized.getSampleSize() == 0);
		REQUIRE(unserialized.getSampleMean() == 0);
		REQUIRE(unserialized.getSampleVariance() == 0);
	}


	boost::mt11213b rng;
	boost::random::normal_distribution<T> dist(0, 1);
	boost::random::uniform_int_distribution<std::size_t> range(1, 5);
	for(std::size_t i = 0; i < 50; ++i)
	{
		const std::size_t maxJ = range(rng);
		for(std::size_t j = 0; j < maxJ; ++j)
			stat.addDatum(dist(rng));

		OpenMPCD::OnTheFlyStatistics<T> unserialized;
		if(maxJ == 1)
			unserialized.addDatum(0);

		unserialized.unserializeFromString(stat.serializeToString());

		REQUIRE(stat.getSampleSize() == Approx(unserialized.getSampleSize()));
		REQUIRE(stat.getSampleMean() == Approx(unserialized.getSampleMean()));
		REQUIRE(
			stat.getSampleVariance()
			==
			Approx(unserialized.getSampleVariance()));
	}

	{
		OpenMPCD::OnTheFlyStatistics<T> stat;
		OpenMPCD::OnTheFlyStatistics<T> unserialized;
		stat.addDatum(1);
		stat.addDatum(2);
		stat.addDatum(3);
		stat.addDatum(4);
		stat.addDatum(5);

		unserialized.unserializeFromString("1;5;3.0;10.0");
		REQUIRE(stat.getSampleSize() == unserialized.getSampleSize());
		REQUIRE(stat.getSampleMean() == unserialized.getSampleMean());
		REQUIRE(stat.getSampleVariance() == unserialized.getSampleVariance());
	}
}

SCENARIO(
	"`OpenMPCD::OnTheFlyStatistics`",
	"")
{
	test_OnTheFlyStatistics<float>();
	test_OnTheFlyStatistics_BesselsCorrection<float>();
	test_OnTheFlyStatistics_probabilistic<float>();
	test_OnTheFlyStatistics_serializeToString_unserializeFromString<float>();

	test_OnTheFlyStatistics<double>();
	test_OnTheFlyStatistics_BesselsCorrection<double>();
	test_OnTheFlyStatistics_probabilistic<double>();
	test_OnTheFlyStatistics_serializeToString_unserializeFromString<double>();
}

