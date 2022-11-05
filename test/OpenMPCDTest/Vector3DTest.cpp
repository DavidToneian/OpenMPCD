/**
 * @file
 * Tests `OpenMPCD::Vector3D`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/Vector3D.hpp>

#include <OpenMPCD/Utility/MathematicalFunctions.hpp>

#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/scoped_ptr.hpp>


#define TEST_DATATYPES (float)(double)(long double)(OpenMPCD::FP)


template<typename T> static void dot_real()
{
	using OpenMPCD::Vector3D;

	const Vector3D<T> zero(0.0, 0.0, 0.0);

	const Vector3D<T> vec1(1.0, 2.5, -42.42);
	const Vector3D<T> vec2(-2.0, 99.9, 123.456);

	const T vec1_dot_vec2 = vec1.getX() * vec2.getX() + vec1.getY() * vec2.getY() + vec1.getZ() * vec2.getZ();

	REQUIRE(zero.dot(zero) == 0);

	REQUIRE(vec1.dot(zero) == 0);
	REQUIRE(zero.dot(vec1) == 0);

	REQUIRE(vec1.dot(vec2) == vec1_dot_vec2);
	REQUIRE(vec2.dot(vec1) == vec1_dot_vec2);
}

template<typename T> static void dot_complex()
{
	using OpenMPCD::Vector3D;

	const Vector3D<std::complex<T> > zero(0.0, 0.0, 0.0);

	const Vector3D<std::complex<T> > vec1(
			std::complex<T>(1.0, -1.2), std::complex<T>(2.5, 33.0), std::complex<T>(-42.42, 22.2233));
	const Vector3D<std::complex<T> > vec2(-2.0, 99.9, 123.456);

	const std::complex<T> vec1_dot_vec2 =
		std::conj(vec1.getX()) * vec2.getX()
		+ std::conj(vec1.getY()) * vec2.getY()
		+ std::conj(vec1.getZ()) * vec2.getZ();

	REQUIRE(zero.dot(zero) == std::complex<T>(0, 0));

	REQUIRE(vec1.dot(zero) == std::complex<T>(0, 0));
	REQUIRE(zero.dot(vec1) == std::complex<T>(0, 0));

	REQUIRE(vec1.dot(vec2) == vec1_dot_vec2);
	REQUIRE(vec2.dot(vec1) == std::conj(vec1_dot_vec2));
}

SCENARIO(
	"`OpenMPCD::Vector3D::dot`",
	"")
{
	dot_real<float>();
	dot_real<double>();
	dot_real<long double>();

	dot_complex<float>();
	dot_complex<double>();
	dot_complex<long double>();
}



template<typename T> static void test_getRandomUnitVector()
{
	static const std::size_t count = 100;

	boost::scoped_ptr<OpenMPCD::Vector3D<T> > vectors[count];

	OpenMPCD::RNG rng;

	for(std::size_t i = 0; i < count; ++i)
	{
		const OpenMPCD::Vector3D<T> random =
			OpenMPCD::Vector3D<T>::getRandomUnitVector(rng);

		REQUIRE(random.getMagnitudeSquared() == Approx(1));

		vectors[i].reset(new OpenMPCD::Vector3D<T>(random));

		for(std::size_t j = 0; j < i; ++j)
			REQUIRE(*vectors[j].get() != random);
	}
}
SCENARIO(
	"`OpenMPCD::Vector3D::getRandomUnitVector`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_getRandomUnitVector<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}



template<typename T> static void test_getUnitVectorFromRandom01()
{
	static const T X_1 = 0.01234;
	static const T X_2 = 0.56789;

	const OpenMPCD::Vector3D<T> result =
		OpenMPCD::Vector3D<T>::getUnitVectorFromRandom01(X_1, X_2);
	REQUIRE(result.getMagnitudeSquared() == Approx(1));


	const T z         = 2 * X_1 - 1;
	const T phiOverPi =	2 * X_2;

	const T root = OpenMPCD::Utility::MathematicalFunctions::sqrt(1 - z*z);

	T x;
	T y;
	OpenMPCD::Utility::MathematicalFunctions::sincospi(phiOverPi, &y, &x);

	REQUIRE(result.getX() == x * root);
	REQUIRE(result.getY() == y * root);
	REQUIRE(result.getZ() == z);
}
SCENARIO(
	"`OpenMPCD::Vector3D::getUnitVectorFromRandom01`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_getUnitVectorFromRandom01<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		TEST_DATATYPES)

	#undef CALLTEST
}



template<typename T> static void equality_inequality_operators()
{
	using OpenMPCD::Vector3D;

	const Vector3D<T> zero(0.0, 0.0, 0.0);

	const Vector3D<T> vec1(1.0, 2.5, -42.42);
	const Vector3D<T> vec2(1.0, 2.5, 0);
	const Vector3D<T> vec3(-2.0, 99.9, 123.456);

	REQUIRE(zero == zero);
	REQUIRE_FALSE(zero != zero);
	REQUIRE_FALSE(zero == vec1);
	REQUIRE(zero != vec1);
	REQUIRE_FALSE(zero == vec2);
	REQUIRE(zero != vec2);
	REQUIRE_FALSE(zero == vec3);
	REQUIRE(zero != vec3);

	REQUIRE_FALSE(vec1 == zero);
	REQUIRE(vec1 != zero);
	REQUIRE(vec1 == vec1);
	REQUIRE_FALSE(vec1 != vec1);
	REQUIRE_FALSE(vec1 == vec2);
	REQUIRE(vec1 != vec2);
	REQUIRE_FALSE(vec1 == vec3);
	REQUIRE(vec1 != vec3);

	REQUIRE_FALSE(vec2 == zero);
	REQUIRE(vec2 != zero);
	REQUIRE_FALSE(vec2 == vec1);
	REQUIRE(vec2 != vec1);
	REQUIRE(vec2 == vec2);
	REQUIRE_FALSE(vec2 != vec2);
	REQUIRE_FALSE(vec2 == vec3);
	REQUIRE(vec2 != vec3);

	REQUIRE_FALSE(vec3 == zero);
	REQUIRE(vec3 != zero);
	REQUIRE_FALSE(vec3 == vec1);
	REQUIRE(vec3 != vec1);
	REQUIRE_FALSE(vec3 == vec2);
	REQUIRE(vec3 != vec2);
	REQUIRE(vec3 == vec3);
	REQUIRE_FALSE(vec3 != vec3);
}

SCENARIO(
	"`OpenMPCD::Vector3D::operator==`, "
	"`OpenMPCD::Vector3D::operator!=`",
	"")
{
	equality_inequality_operators<float>();
	equality_inequality_operators<double>();
	equality_inequality_operators<long double>();
}
