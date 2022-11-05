/**
 * @file
 * Tests `OpenMPCD::Scalar`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/Scalar.hpp>


template<typename T> static void getRealPart_real()
{
	namespace Scalar = OpenMPCD::Scalar;

	const T zero = 0.0;
	const T negative = -1.5;
	const T positive = 4.2;

	REQUIRE(Scalar::getRealPart(zero) == zero);
	REQUIRE(Scalar::getRealPart(negative) == negative);
	REQUIRE(Scalar::getRealPart(positive) == positive);
}

template<typename T> static void getRealPart_complex()
{
	namespace Scalar = OpenMPCD::Scalar;

	const std::complex<T> zero(0.0, 0.0);

	const std::complex<T> negativeReal(-1.5, 0.0);
	const std::complex<T> positiveReal(4.2, 0.0);

	const std::complex<T> negativeImag(0.0, -2.3);
	const std::complex<T> positiveImag(0.0, 9.99);

	const std::complex<T> mixed(-12.34, 56.78);


	REQUIRE(Scalar::getRealPart(zero) == zero.real());
	REQUIRE(Scalar::getRealPart(negativeReal) == negativeReal.real());
	REQUIRE(Scalar::getRealPart(positiveReal) == positiveReal.real());
	REQUIRE(Scalar::getRealPart(negativeImag) == negativeImag.real());
	REQUIRE(Scalar::getRealPart(positiveImag) == positiveImag.real());
	REQUIRE(Scalar::getRealPart(mixed) == mixed.real());
}

SCENARIO(
	"`OpenMPCD::Scalar::getRealPart`",
	"")
{
	getRealPart_real<float>();
	getRealPart_real<double>();
	getRealPart_real<long double>();

	getRealPart_complex<float>();
	getRealPart_complex<double>();
	getRealPart_complex<long double>();
}


template<typename T> static void isZero_real()
{
	namespace Scalar = OpenMPCD::Scalar;

	const T zero = 0.0;
	const T negative = -1.5;
	const T positive = 4.2;

	REQUIRE(Scalar::isZero(zero));
	REQUIRE(!Scalar::isZero(negative));
	REQUIRE(!Scalar::isZero(positive));
}

template<typename T> static void isZero_complex()
{
	namespace Scalar = OpenMPCD::Scalar;

	const std::complex<T> zero(0.0, 0.0);

	const std::complex<T> negativeReal(-1.5, 0.0);
	const std::complex<T> positiveReal(4.2, 0.0);

	const std::complex<T> negativeImag(0.0, -2.3);
	const std::complex<T> positiveImag(0.0, 9.99);

	const std::complex<T> mixed(-12.34, 56.78);


	REQUIRE(Scalar::isZero(zero));
	REQUIRE(!Scalar::isZero(negativeReal));
	REQUIRE(!Scalar::isZero(positiveReal));
	REQUIRE(!Scalar::isZero(negativeImag));
	REQUIRE(!Scalar::isZero(positiveImag));
	REQUIRE(!Scalar::isZero(mixed));
}

SCENARIO(
	"`OpenMPCD::Scalar::isZero`",
	"")
{
	isZero_real<float>();
	isZero_real<double>();
	isZero_real<long double>();

	isZero_complex<float>();
	isZero_complex<double>();
	isZero_complex<long double>();
}
