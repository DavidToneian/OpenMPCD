/**
 * @file
 * Tests `OpenMPCD::PairPotentials::WeeksChandlerAndersen`.
 */

#include <OpenMPCDTest/include_catch.hpp>
#include <OpenMPCDTest/include_libtaylor.hpp>

#include <OpenMPCD/PairPotentials/WeeksChandlerAndersen_DistanceOffset.hpp>

#include <boost/preprocessor/seq/for_each.hpp>


template<typename T>
static void test_constructor()
{
	using namespace OpenMPCD::PairPotentials;

	REQUIRE_NOTHROW(WeeksChandlerAndersen_DistanceOffset<T>(0, 0, 0));
	REQUIRE_NOTHROW(WeeksChandlerAndersen_DistanceOffset<T>(0, 2, 3));
	REQUIRE_NOTHROW(WeeksChandlerAndersen_DistanceOffset<T>(1, 0, 3));
	REQUIRE_NOTHROW(WeeksChandlerAndersen_DistanceOffset<T>(1, 2, 0));

	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			(WeeksChandlerAndersen_DistanceOffset<T>(-1, 2, 3)),
			OpenMPCD::AssertionException);
		REQUIRE_THROWS_AS(
			(WeeksChandlerAndersen_DistanceOffset<T>(1, -2, 3)),
			OpenMPCD::AssertionException);
		REQUIRE_THROWS_AS(
			(WeeksChandlerAndersen_DistanceOffset<T>(1, 2, -3)),
			OpenMPCD::AssertionException);
	#endif
}
SCENARIO(
	"`OpenMPCD::PairPotentials::WeeksChandlerAndersen_DistanceOffset::"
	"WeeksChandlerAndersen_DistanceOffset`",
	"")
{
	#define SEQ_DATATYPE (float)(double)(long double)(OpenMPCD::FP)
	#define CALLTEST(_r, _data, T) \
		test_constructor<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef SEQ_DATATYPE
	#undef CALLTEST
}

template<typename T>
static void test_force(const T epsilon, const T sigma, const T d)
{
	using namespace OpenMPCD;
	using namespace OpenMPCD::PairPotentials;

	typedef taylor<T, 3, 1> Taylor;

	const WeeksChandlerAndersen_DistanceOffset<T>
		interaction(epsilon, sigma, d);
	const WeeksChandlerAndersen_DistanceOffset<Taylor>
		interaction_taylor(epsilon, sigma, d);

	for(unsigned int i = 0; i < 100; ++i)
	{
		const Vector3D<T> R(-0.1 + i, i*0.1, 2*i);
		const Vector3D<Taylor> R_taylor(
			Taylor(R.getX(), 0), Taylor(R.getY(), 1), Taylor(R.getZ(), 2));

		const T r = R.getMagnitude();
		if(r - d <= 0)
		{
			#ifdef OPENMPCD_DEBUG
				REQUIRE_THROWS_AS(
					interaction.force(R),
					OpenMPCD::AssertionException);
			#endif

			continue;
		}

		const Vector3D<T> result = interaction.force(R);
		const Taylor result_taylor = - interaction_taylor.potential(R_taylor);

		REQUIRE(result.getX() == Approx(result_taylor[1]));
		REQUIRE(result.getY() == Approx(result_taylor[2]));
		REQUIRE(result.getZ() == Approx(result_taylor[3]));
	}

	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			interaction.force(Vector3D<T>(0, 0, 0)),
			OpenMPCD::AssertionException);
	#endif
}
template<typename T>
static void test_force()
{
	static const T epsilon[] = {0, 0.1, 0.5, 1, 5, 10};
	static const T sigma[]   = {0, 0.1, 0.5, 1, 5, 10};
	static const T d[]       = {0, 0.1, 0.5, 1, 5, 10};

	for(std::size_t e = 0; e < sizeof(epsilon)/sizeof(epsilon[0]); ++e)
	{
		for(std::size_t s = 0; s < sizeof(sigma)/sizeof(sigma[0]); ++s)
		{
			for(std::size_t D = 0; D < sizeof(d)/sizeof(d[0]); ++D)
				test_force<T>(epsilon[e], sigma[s], d[D]);
		}
	}
}
SCENARIO(
	"`OpenMPCD::PairPotentials::WeeksChandlerAndersen_DistanceOffset::force`",
	"")
{
	#define SEQ_DATATYPE (float)(double)(long double)(OpenMPCD::FP)
	#define CALLTEST(_r, _data, T) \
		test_force<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef SEQ_DATATYPE
	#undef CALLTEST
}


template<typename T>
static void test_potential(const T epsilon, const T sigma, const T d)
{
	using namespace OpenMPCD;
	using namespace OpenMPCD::PairPotentials;

	const WeeksChandlerAndersen_DistanceOffset<T>
		interaction(epsilon, sigma, d);

	for(unsigned int i = 0; i < 100; ++i)
	{
		const Vector3D<T> R(-0.1 + i, i*0.1, 2*i);
		const T r = R.getMagnitude();

		if(r - d <= 0)
		{
			#ifdef OPENMPCD_DEBUG
				REQUIRE_THROWS_AS(
					interaction.potential(R),
					OpenMPCD::AssertionException);
			#endif

			continue;
		}

		const T cutoff = pow(2, 1.0/6) * sigma;

		T expected;

		if(r - d > cutoff)
		{
			expected = 0;
		}
		else
		{
			const T frac6 = pow(sigma/(r - d), 6);
			const T frac12 = frac6 * frac6;

			expected = 4 * epsilon * (frac12 - frac6 + 1.0/4);
		}

		REQUIRE(expected == Approx(interaction.potential(R)));
	}

	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			interaction.potential(Vector3D<T>(0, 0, 0)),
			OpenMPCD::AssertionException);
	#endif
}
template<typename T>
static void test_potential()
{
	static const T epsilon[] = {0, 0.1, 0.5, 1, 5, 10};
	static const T sigma[]   = {0, 0.1, 0.5, 1, 5, 10};
	static const T d[]       = {0, 0.1, 0.5, 1, 5, 10};

	for(std::size_t e = 0; e < sizeof(epsilon)/sizeof(epsilon[0]); ++e)
	{
		for(std::size_t s = 0; s < sizeof(sigma)/sizeof(sigma[0]); ++s)
		{
			for(std::size_t D = 0; D < sizeof(d)/sizeof(d[0]); ++D)
				test_potential<T>(epsilon[e], sigma[s], d[D]);
		}
	}
}
SCENARIO(
	"`OpenMPCD::PairPotentials::"
	"WeeksChandlerAndersen_DistanceOffset::potential`",
	"")
{
	#define SEQ_DATATYPE (float)(double)(long double)(OpenMPCD::FP)
	#define CALLTEST(_r, _data, T) \
		test_potential<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef SEQ_DATATYPE
	#undef CALLTEST
}



template<typename T>
static void test_forceOnR1DueToR2(const T epsilon, const T sigma, const T d)
{
	using namespace OpenMPCD;
	using namespace OpenMPCD::PairPotentials;

	const WeeksChandlerAndersen_DistanceOffset<T>
		interaction(epsilon, sigma, d);

	for(unsigned int i = 0; i < 100; ++i)
	{
		const Vector3D<T> r1(-0.1 + i, i*0.1, 2*i);
		const Vector3D<T> r2(-0.2 + 2 * i, i*0.1 - 3, 4*i);

		const T r = (r1-r2).getMagnitude();
		if(r - d <= 0)
		{
			#ifdef OPENMPCD_DEBUG
				REQUIRE_THROWS_AS(
					interaction.forceOnR1DueToR2(r1, r2),
					OpenMPCD::AssertionException);
			#endif

			continue;
		}

		const Vector3D<T> result = interaction.forceOnR1DueToR2(r1, r2);
		const Vector3D<T> expected = interaction.force(r1 - r2);

		REQUIRE(result == expected);
	}
}
template<typename T>
static void test_forceOnR1DueToR2()
{
	static const T epsilon[] = {0, 0.1, 0.5, 1, 5, 10};
	static const T sigma[]   = {0, 0.1, 0.5, 1, 5, 10};
	static const T d[]       = {0, 0.1, 0.5, 1, 5, 10};

	for(std::size_t e = 0; e < sizeof(epsilon)/sizeof(epsilon[0]); ++e)
	{
		for(std::size_t s = 0; s < sizeof(sigma)/sizeof(sigma[0]); ++s)
		{
			for(std::size_t D = 0; D < sizeof(d)/sizeof(d[0]); ++D)
				test_forceOnR1DueToR2<T>(epsilon[e], sigma[s], d[D]);
		}
	}
}
SCENARIO(
	"`OpenMPCD::PairPotentials::"
	"WeeksChandlerAndersen_DistanceOffset::forceOnR1DueToR2`",
	"")
{
	#define SEQ_DATATYPE (float)(double)(long double)(OpenMPCD::FP)
	#define CALLTEST(_r, _data, T) \
		test_forceOnR1DueToR2<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef SEQ_DATATYPE
	#undef CALLTEST
}


template<typename T>
static void test_get_parameters()
{
	using namespace OpenMPCD;
	using namespace OpenMPCD::PairPotentials;

	const T epsilon = 2;
	const T sigma = 10;
	const T D = 3;
	const WeeksChandlerAndersen_DistanceOffset<T> interaction(
		epsilon, sigma, D);

	REQUIRE(interaction.getEpsilon() == epsilon);
	REQUIRE(interaction.getSigma() == sigma);
	REQUIRE(interaction.getD() == D);
}
SCENARIO(
	"`OpenMPCD::PairPotentials::WeeksChandlerAndersen_DistanceOffset::"
		"getEpsilon`, "
	"`OpenMPCD::PairPotentials::WeeksChandlerAndersen_DistanceOffset::"
		"getSigma`, "
	"`OpenMPCD::PairPotentials::WeeksChandlerAndersen_DistanceOffset::"
		"getD`",
	"")
{
	#define SEQ_DATATYPE (float)(double)(long double)(OpenMPCD::FP)
	#define CALLTEST(_r, _data, T) \
		test_get_parameters<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef SEQ_DATATYPE
	#undef CALLTEST
}
