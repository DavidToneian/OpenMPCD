/**
 * @file
 * Tests functionality in `OpenMPCD::Utility::MathematicalConstants`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/Utility/MathematicalConstants.hpp>

#include <OpenMPCD/Types.hpp>

#include <boost/math/constants/constants.hpp>
#include <boost/preprocessor/seq/for_each.hpp>


#define TEST_DATATYPES (float)(double)(long double)(OpenMPCD::FP)

template<typename T>
static void test_pi()
{
	REQUIRE(
		OpenMPCD::Utility::MathematicalConstants::pi<T>()
		==
		boost::math::constants::pi<T>());
}
SCENARIO(
	"`OpenMPCD::Utility::MathematicalConstants::pi`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_pi<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}
