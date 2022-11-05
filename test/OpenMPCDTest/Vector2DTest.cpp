/**
 * @file
 * Tests `OpenMPCD::Vector2D`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/Vector2D.hpp>

#include <OpenMPCD/Utility/MathematicalFunctions.hpp>

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/scoped_ptr.hpp>

#include <vector>


#ifndef TEST_DATATYPES
#define TEST_DATATYPES (float)(double)(long double)(OpenMPCD::FP)
#endif

#ifndef TEST_DATATYPES_HIGHPRECISION
#define TEST_DATATYPES_HIGHPRECISION (double)(long double)(OpenMPCD::FP)
#endif


#ifndef OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION
#define OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION
#endif

#ifndef OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION_ARG
#define OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION_ARG
#endif

#ifndef OPENMPCDTEST_VECTOR2DTEST_SCENARIO_SUFFIX
#define OPENMPCDTEST_VECTOR2DTEST_SCENARIO_SUFFIX
#endif

#ifndef OPENMPCDTEST_VECTOR2DTEST_SCENARIO_TAG
#define OPENMPCDTEST_VECTOR2DTEST_SCENARIO_TAG ""
#endif

#ifndef REQ
#define REQ REQUIRE
#endif

#ifndef REQ_FALSE
#define REQ_FALSE REQUIRE_FALSE
#endif

#ifndef OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTIONS_CALL
#define OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTIONS_CALL(funcReal, funcComplex) \
	funcReal(); funcComplex();
#endif


static void _nullFunction(){}


template<typename T> static
OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION
void test_getters_real(OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION_ARG)
{
	typedef OpenMPCD::Vector2D<T> V;

	const V defaultInitialized;

	const V zero(0.0, 0.0);

	const V vec1(1.0, 2.5);
	const V vec2(-2.0, 99.9);

	REQ(defaultInitialized.getX() == 0);
	REQ(defaultInitialized.getY() == 0);

	REQ(zero.getX() == 0);
	REQ(zero.getY() == 0);

	REQ(vec1.getX() == 1);
	REQ(vec1.getY() == 2.5);

	REQ(vec2.getX() == -2);
	REQ(vec2.getY() == T(99.9));
}
template<typename T> static
void test_getters_complex(OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION_ARG)
{
	typedef std::complex<T> C;
	typedef OpenMPCD::Vector2D<C> V;

	const V zero(0.0, 0.0);

	const V vec1(C(1.0, -1.2), C(2.5, 33.0));
	const V vec2(-2.0, 99.9);

	REQ(zero.getX() == C(0, 0));
	REQ(zero.getY() == C(0, 0));

	REQ(vec1.getX() == C(1, -1.2));
	REQ(vec1.getY() == C(2.5, 33));

	REQ(vec2.getX() == C(-2, 0));
	REQ(vec2.getY() == C(99.9, 0));
}
SCENARIO(
	"`OpenMPCD::Vector2D::getX`, "
	"`OpenMPCD::Vector2D::getY`"
	OPENMPCDTEST_VECTOR2DTEST_SCENARIO_SUFFIX,
	OPENMPCDTEST_VECTOR2DTEST_SCENARIO_TAG)
{
	#define CALLTEST(_r, _data, T) \
		OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTIONS_CALL( \
			test_getters_real<T>, test_getters_complex<T>);

	BOOST_PP_SEQ_FOR_EACH(\
			CALLTEST, \
			_,
			TEST_DATATYPES)

	#undef CALLTEST
}


template<typename T> static
OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION
void test_dot_real(OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION_ARG)
{
	typedef OpenMPCD::Vector2D<T> V;

	const V zero(0.0, 0.0);

	const V vec1(1.0, 2.5);
	const V vec2(-2.0, 99.9);

	const T vec1_dot_vec2 =
		vec1.getX() * vec2.getX() + vec1.getY() * vec2.getY();

	REQ(zero.dot(zero) == 0);

	REQ(vec1.dot(zero) == 0);
	REQ(zero.dot(vec1) == 0);

	REQ(vec1.dot(vec2) == vec1_dot_vec2);
	REQ(vec2.dot(vec1) == vec1_dot_vec2);
}
template<typename T> static
void test_dot_complex(OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION_ARG)
{
	typedef std::complex<T> C;
	typedef OpenMPCD::Vector2D<C> V;

	const V zero(0.0, 0.0);

	const V vec1(C(1.0, -1.2), C(2.5, 33.0));
	const V vec2(-2.0, 99.9);

	const C vec1_dot_vec2 =
		std::conj(vec1.getX()) * vec2.getX()
		+ std::conj(vec1.getY()) * vec2.getY();

	REQ(zero.dot(zero) == C(0, 0));

	REQ(vec1.dot(zero) == C(0, 0));
	REQ(zero.dot(vec1) == C(0, 0));

	REQ(vec1.dot(vec2) == vec1_dot_vec2);
	REQ(vec2.dot(vec1) == std::conj(vec1_dot_vec2));
}
SCENARIO(
	"`OpenMPCD::Vector2D::dot`"
	OPENMPCDTEST_VECTOR2DTEST_SCENARIO_SUFFIX,
	OPENMPCDTEST_VECTOR2DTEST_SCENARIO_TAG)
{
	#define CALLTEST(_r, _data, T) \
		OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTIONS_CALL( \
			test_dot_real<T>, test_dot_complex<T>);

	BOOST_PP_SEQ_FOR_EACH(\
			CALLTEST, \
			_,
			TEST_DATATYPES)

	#undef CALLTEST
}


template<typename T> static
OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION
void test_getMagnitudeSquared_getMagnitude_real(
	OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION_ARG)
{
	typedef OpenMPCD::Vector2D<T> V;

	namespace MF = OpenMPCD::Utility::MathematicalFunctions;

	const V zero(0.0, 0.0);

	const V vec1(1.0, 2.5);
	const V vec2(-2.0, 99.9);

	REQ(zero.getMagnitudeSquared() == 0);
	REQ(zero.getMagnitude() == 0);

	REQ(vec1.getMagnitudeSquared() == vec1.dot(vec1));
	REQ(vec1.getMagnitude() == MF::sqrt(vec1.getMagnitudeSquared()));

	REQ(vec2.getMagnitudeSquared() == vec2.dot(vec2));
	REQ(vec2.getMagnitude() == MF::sqrt(vec2.getMagnitudeSquared()));
}
template<typename T> static
void test_getMagnitudeSquared_getMagnitude_complex(
	OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION_ARG)
{
	typedef std::complex<T> C;
	typedef OpenMPCD::Vector2D<C> V;

	namespace MF = OpenMPCD::Utility::MathematicalFunctions;

	const V zero(0.0, 0.0);

	const V vec1(C(1.0, -1.2), C(2.5, 33.0));
	const V vec2(-2.0, 99.9);

	const C vec1_dot_vec2 =
		std::conj(vec1.getX()) * vec2.getX()
		+ std::conj(vec1.getY()) * vec2.getY();

	REQ(zero.getMagnitudeSquared() == 0);
	REQ(zero.getMagnitude() == 0);

	REQ(vec1.getMagnitudeSquared() == vec1.dot(vec1));
	REQ(vec1.getMagnitude() == MF::sqrt(vec1.getMagnitudeSquared()));

	REQ(vec2.getMagnitudeSquared() == vec2.dot(vec2));
	REQ(vec2.getMagnitude() == MF::sqrt(vec2.getMagnitudeSquared()));
}
SCENARIO(
	"`OpenMPCD::Vector2D::getMagnitudeSquared`, "
	"`OpenMPCD::Vector2D::getMagnitude`"
	OPENMPCDTEST_VECTOR2DTEST_SCENARIO_SUFFIX,
	OPENMPCDTEST_VECTOR2DTEST_SCENARIO_TAG)
{
	#define CALLTEST(_r, _data, T) \
		OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTIONS_CALL( \
			test_getMagnitudeSquared_getMagnitude_real<T>, \
			test_getMagnitudeSquared_getMagnitude_complex<T>);

	BOOST_PP_SEQ_FOR_EACH(\
			CALLTEST, \
			_,
			TEST_DATATYPES)

	#undef CALLTEST
}


template<typename T> static
OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION
void test_getCosineOfAngle_getAngle_real(
	OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION_ARG)
{
	const std::size_t count = 100;

	typedef OpenMPCD::Vector2D<T> V;

	namespace MF = OpenMPCD::Utility::MathematicalFunctions;


	V vectors[count];

	for(std::size_t i = 0; i < count; ++i)
		vectors[i] = V(i + 0.01, -T(i) / 2.0 + 0.01);

	for(std::size_t i = 0; i < count; ++i)
	{
		for(std::size_t j = 0; j < count; ++j)
		{
			namespace MF = OpenMPCD::Utility::MathematicalFunctions;

			T expected = vectors[i].dot(vectors[j]);
			expected /= MF::sqrt(
				vectors[i].getMagnitudeSquared() *
				vectors[j].getMagnitudeSquared());


			const T acosAngle = vectors[i].getCosineOfAngle(vectors[j]);

			REQ(acosAngle == expected);
			REQ(MF::acos(acosAngle) == vectors[i].getAngle(vectors[j]));
		}
	}
}
SCENARIO(
	"`OpenMPCD::Vector2D::getCosineOfAngle`, "
	"`OpenMPCD::Vector2D::getAngle`"
	OPENMPCDTEST_VECTOR2DTEST_SCENARIO_SUFFIX,
	OPENMPCDTEST_VECTOR2DTEST_SCENARIO_TAG)
{
	#define CALLTEST(_r, _data, T) \
		OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTIONS_CALL( \
			test_getCosineOfAngle_getAngle_real<T>, _nullFunction);

	BOOST_PP_SEQ_FOR_EACH(\
			CALLTEST, \
			_,
			TEST_DATATYPES_HIGHPRECISION)

	#undef CALLTEST
}



template<typename T> static
OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION
void test_equality_inequality_operators_real(
	OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION_ARG)
{
	typedef OpenMPCD::Vector2D<T> V;

	const V zero(0.0, 0.0);

	const V vec1(1.0, 2.5);
	const V vec2(1.0, 1.5);
	const V vec3(-2.0, 99.9);

	REQ(zero == zero);
	REQ_FALSE(zero != zero);
	REQ_FALSE(zero == vec1);
	REQ(zero != vec1);
	REQ_FALSE(zero == vec2);
	REQ(zero != vec2);
	REQ_FALSE(zero == vec3);
	REQ(zero != vec3);

	REQ_FALSE(vec1 == zero);
	REQ(vec1 != zero);
	REQ(vec1 == vec1);
	REQ_FALSE(vec1 != vec1);
	REQ_FALSE(vec1 == vec2);
	REQ(vec1 != vec2);
	REQ_FALSE(vec1 == vec3);
	REQ(vec1 != vec3);

	REQ_FALSE(vec2 == zero);
	REQ(vec2 != zero);
	REQ_FALSE(vec2 == vec1);
	REQ(vec2 != vec1);
	REQ(vec2 == vec2);
	REQ_FALSE(vec2 != vec2);
	REQ_FALSE(vec2 == vec3);
	REQ(vec2 != vec3);

	REQ_FALSE(vec3 == zero);
	REQ(vec3 != zero);
	REQ_FALSE(vec3 == vec1);
	REQ(vec3 != vec1);
	REQ_FALSE(vec3 == vec2);
	REQ(vec3 != vec2);
	REQ(vec3 == vec3);
	REQ_FALSE(vec3 != vec3);
}
template<typename T> static
void test_equality_inequality_operators_complex(
	OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTION_ARG)
{
	typedef std::complex<T> C;
	typedef OpenMPCD::Vector2D<C> V;

	const V zero(0.0, 0.0);

	const V vec1(1.0, 2.5);
	const V vec2(1.0, 1.5);
	const V vec3(1.0, C(1.5, 0.1));

	REQ(zero == zero);
	REQ_FALSE(zero != zero);
	REQ_FALSE(zero == vec1);
	REQ(zero != vec1);
	REQ_FALSE(zero == vec2);
	REQ(zero != vec2);
	REQ_FALSE(zero == vec3);
	REQ(zero != vec3);

	REQ_FALSE(vec1 == zero);
	REQ(vec1 != zero);
	REQ(vec1 == vec1);
	REQ_FALSE(vec1 != vec1);
	REQ_FALSE(vec1 == vec2);
	REQ(vec1 != vec2);
	REQ_FALSE(vec1 == vec3);
	REQ(vec1 != vec3);

	REQ_FALSE(vec2 == zero);
	REQ(vec2 != zero);
	REQ_FALSE(vec2 == vec1);
	REQ(vec2 != vec1);
	REQ(vec2 == vec2);
	REQ_FALSE(vec2 != vec2);
	REQ_FALSE(vec2 == vec3);
	REQ(vec2 != vec3);

	REQ_FALSE(vec3 == zero);
	REQ(vec3 != zero);
	REQ_FALSE(vec3 == vec1);
	REQ(vec3 != vec1);
	REQ_FALSE(vec3 == vec2);
	REQ(vec3 != vec2);
	REQ(vec3 == vec3);
	REQ_FALSE(vec3 != vec3);
}
SCENARIO(
	"`OpenMPCD::Vector2D::operator==`, "
	"`OpenMPCD::Vector2D::operator!=`"
	OPENMPCDTEST_VECTOR2DTEST_SCENARIO_SUFFIX,
	OPENMPCDTEST_VECTOR2DTEST_SCENARIO_TAG)
{
	#define CALLTEST(_r, _data, T) \
		OPENMPCDTEST_VECTOR2DTEST_TESTFUNCTIONS_CALL( \
			test_equality_inequality_operators_real<T>, \
			test_equality_inequality_operators_complex<T>);

	BOOST_PP_SEQ_FOR_EACH(\
			CALLTEST, \
			_,
			TEST_DATATYPES)

	#undef CALLTEST
}

