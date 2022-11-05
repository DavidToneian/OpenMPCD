/**
 * @file
 * Tests functionality in `OpenMPCD::Utility::MathematicalFunctions`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/Utility/MathematicalFunctions.hpp>

#include <OpenMPCD/Types.hpp>
#include <OpenMPCD/Utility/MathematicalConstants.hpp>

#include <boost/preprocessor/seq/for_each.hpp>


#define TEST_DATATYPES (float)(double)(long double)(OpenMPCD::FP)

static const long double values[] =
{
	-123.456L,
	-1.23L,
	-1.0L,
	-0.5L,
	-0.123L,
	0.0L,
	0.123L,
	0.5L,
	1.0L,
	1.23L,
	123.456L
};
static const std::size_t valueCount = sizeof(values)/sizeof(values[0]);



template<typename T>
static void test_acos();
template<> void test_acos<float>()
{
	typedef float T;

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		if(x < -1 || x > 1)
			continue;

		REQUIRE(
			OpenMPCD::Utility::MathematicalFunctions::acos(x)
			==
			::acosf(x));
	}
}
template<> void test_acos<double>()
{
	typedef double T;

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		if(x < -1 || x > 1)
			continue;

		REQUIRE(
			OpenMPCD::Utility::MathematicalFunctions::acos(x)
			==
			::acos(x));
	}
}
template<> void test_acos<long double>()
{
	typedef long double T;

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		if(x < -1 || x > 1)
			continue;

		REQUIRE(
			OpenMPCD::Utility::MathematicalFunctions::acos(x)
			==
			::acosl(x));
	}
}
SCENARIO(
	"`OpenMPCD::Utility::MathematicalFunctions::acos`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_acos<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}



template<typename T>
static void test_cos();
template<> void test_cos<float>()
{
	typedef float T;

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		REQUIRE(
			OpenMPCD::Utility::MathematicalFunctions::cos(x)
			==
			::cosf(x));
	}
}
template<> void test_cos<double>()
{
	typedef double T;

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		REQUIRE(
			OpenMPCD::Utility::MathematicalFunctions::cos(x)
			==
			::cos(x));
	}
}
template<> void test_cos<long double>()
{
	typedef long double T;

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		REQUIRE(
			OpenMPCD::Utility::MathematicalFunctions::cos(x)
			==
			::cosl(x));
	}
}
SCENARIO(
	"`OpenMPCD::Utility::MathematicalFunctions::cos`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_cos<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}



template<typename T>
static void test_cospi()
{
	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		const T expected =
			OpenMPCD::Utility::MathematicalFunctions::cos(
				x * OpenMPCD::Utility::MathematicalConstants::pi<T>());

		REQUIRE(
			OpenMPCD::Utility::MathematicalFunctions::cospi(x)
			==
			expected);
	}
}
SCENARIO(
	"`OpenMPCD::Utility::MathematicalFunctions::cospi`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_cospi<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}



template<typename T>
static void test_sin();
template<> void test_sin<float>()
{
	typedef float T;

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		REQUIRE(
			OpenMPCD::Utility::MathematicalFunctions::sin(x)
			==
			::sinf(x));
	}
}
template<> void test_sin<double>()
{
	typedef double T;

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		REQUIRE(
			OpenMPCD::Utility::MathematicalFunctions::sin(x)
			==
			::sin(x));
	}
}
template<> void test_sin<long double>()
{
	typedef long double T;

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		REQUIRE(
			OpenMPCD::Utility::MathematicalFunctions::sin(x)
			==
			::sinl(x));
	}
}
SCENARIO(
	"`OpenMPCD::Utility::MathematicalFunctions::sin`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_sin<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}



template<typename T>
static void test_sinpi()
{
	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		const T expected =
			OpenMPCD::Utility::MathematicalFunctions::sin(
				x * OpenMPCD::Utility::MathematicalConstants::pi<T>());

		REQUIRE(
			OpenMPCD::Utility::MathematicalFunctions::sinpi(x)
			==
			expected);
	}
}
SCENARIO(
	"`OpenMPCD::Utility::MathematicalFunctions::sinpi`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_sinpi<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}



template<typename T>
static void test_sincos()
{
	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		const T expectedSin =
			OpenMPCD::Utility::MathematicalFunctions::sin(x);
		const T expectedCos =
			OpenMPCD::Utility::MathematicalFunctions::cos(x);

		T s;
		T c;

		#ifdef OPENMPCD_DEBUG_ASSERT
			T* const null = NULL;
			REQUIRE_THROWS_AS(
				OpenMPCD::Utility::MathematicalFunctions::sincos(x, null, &c),
				OpenMPCD::NULLPointerException);
			REQUIRE_THROWS_AS(
				OpenMPCD::Utility::MathematicalFunctions::sincos(x, &s, null),
				OpenMPCD::NULLPointerException);
		#endif

		OpenMPCD::Utility::MathematicalFunctions::sincos(x, &s, &c);

		REQUIRE(s == expectedSin);
		REQUIRE(c == expectedCos);
	}
}
SCENARIO(
	"`OpenMPCD::Utility::MathematicalFunctions::sincos`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_sincos<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}


template<typename T>
static void test_sincospi()
{
	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		const T expectedSin =
			OpenMPCD::Utility::MathematicalFunctions::sinpi(x);
		const T expectedCos =
			OpenMPCD::Utility::MathematicalFunctions::cospi(x);

		T s;
		T c;

		#ifdef OPENMPCD_DEBUG_ASSERT
			T* const null = NULL;
			REQUIRE_THROWS_AS(
				OpenMPCD::Utility::MathematicalFunctions::sincos(x, null, &c),
				OpenMPCD::NULLPointerException);
			REQUIRE_THROWS_AS(
				OpenMPCD::Utility::MathematicalFunctions::sincos(x, &s, null),
				OpenMPCD::NULLPointerException);
		#endif

		OpenMPCD::Utility::MathematicalFunctions::sincospi(x, &s, &c);

		REQUIRE(s == expectedSin);
		REQUIRE(c == expectedCos);
	}
}
SCENARIO(
	"`OpenMPCD::Utility::MathematicalFunctions::sincospi`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_sincospi<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}



template<typename T>
static void test_sqrt();
template<> void test_sqrt<float>()
{
	typedef float T;

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		if(x < 0)
			continue;

		REQUIRE(
			OpenMPCD::Utility::MathematicalFunctions::sqrt(x)
			==
			::sqrtf(x));
	}
}
template<> void test_sqrt<double>()
{
	typedef double T;

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		if(x < 0)
			continue;

		REQUIRE(
			OpenMPCD::Utility::MathematicalFunctions::sqrt(x)
			==
			::sqrt(x));
	}
}
template<> void test_sqrt<long double>()
{
	typedef long double T;

	for(std::size_t i = 0; i < valueCount; ++i)
	{
		const T x = values[i];

		if(x < 0)
			continue;

		REQUIRE(
			OpenMPCD::Utility::MathematicalFunctions::sqrt(x)
			==
			::sqrtl(x));
	}
}
SCENARIO(
	"`OpenMPCD::Utility::MathematicalFunctions::sqrt`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_sqrt<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}
