/**
 * @file
 * Tests `OpenMPCD::PairPotentials::AdditivePair`.
 */

#include <OpenMPCDTest/include_catch.hpp>
#include <OpenMPCDTest/include_libtaylor.hpp>

#include <OpenMPCD/PairPotentials/AdditivePair.hpp>

#include <OpenMPCD/PairPotentials/WeeksChandlerAndersen.hpp>

#include <boost/preprocessor/seq/for_each.hpp>


template<typename T>
static void test_constructor()
{
	using namespace OpenMPCD::PairPotentials;

	#ifdef OPENMPCD_DEBUG
		const WeeksChandlerAndersen<T> wca(1.0, 1.0);

		REQUIRE_THROWS_AS(
			(AdditivePair<T>(&wca, NULL)),
			OpenMPCD::NULLPointerException);
		REQUIRE_THROWS_AS(
			(AdditivePair<T>(NULL, &wca)),
			OpenMPCD::NULLPointerException);
		REQUIRE_THROWS_AS(
			(AdditivePair<T>(NULL, NULL)),
			OpenMPCD::NULLPointerException);
	#endif
}
SCENARIO(
	"`OpenMPCD::PairPotentials::AdditivePair::AdditivePair`",
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
static void test_force()
{
	using namespace OpenMPCD;
	using namespace OpenMPCD::PairPotentials;

	typedef taylor<T, 3, 1> Taylor;

	const WeeksChandlerAndersen<T> wca1(1.0, 0.5);
	const WeeksChandlerAndersen<T> wca2(0.5, 1.5);
	const WeeksChandlerAndersen<Taylor> wca1_taylor(1.0, 0.5);
	const WeeksChandlerAndersen<Taylor> wca2_taylor(0.5, 1.5);

	const AdditivePair<T> interaction(&wca1, &wca2);
	const AdditivePair<Taylor> interaction_taylor(&wca1_taylor, &wca2_taylor);

	for(unsigned int i = 0; i < 100; ++i)
	{
		const Vector3D<T> R(-0.1 + i, i*0.1, 2*i);
		const Vector3D<Taylor> R_taylor(
			Taylor(R.getX(), 0), Taylor(R.getY(), 1), Taylor(R.getZ(), 2));

		const Vector3D<T> result = interaction.force(R);
		const Taylor result_taylor = - interaction_taylor.potential(R_taylor);

		REQUIRE(result.getX() == Approx(result_taylor[1]));
		REQUIRE(result.getY() == Approx(result_taylor[2]));
		REQUIRE(result.getZ() == Approx(result_taylor[3]));

		REQUIRE(result == wca1.force(R) + wca2.force(R));
	}
}
SCENARIO(
	"`OpenMPCD::PairPotentials::AdditivePair::force`",
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
static void test_potential()
{
	using namespace OpenMPCD;
	using namespace OpenMPCD::PairPotentials;

	const WeeksChandlerAndersen<T> wca1(1.0, 0.5);
	const WeeksChandlerAndersen<T> wca2(0.5, 1.5);
	const AdditivePair<T> interaction(&wca1, &wca2);

	for(unsigned int i = 0; i < 100; ++i)
	{
		const Vector3D<T> R(-0.1 + i, i*0.1, 2*i);

		REQUIRE(
			interaction.potential(R) == wca1.potential(R) + wca2.potential(R));
	}
}
SCENARIO(
	"`OpenMPCD::PairPotentials::AdditivePair::potential`",
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
static void test_forceOnR1DueToR2()
{
	using namespace OpenMPCD;
	using namespace OpenMPCD::PairPotentials;

	const WeeksChandlerAndersen<T> wca1(1.0, 0.5);
	const WeeksChandlerAndersen<T> wca2(0.5, 1.5);
	const AdditivePair<T> interaction(&wca1, &wca2);

	for(unsigned int i = 0; i < 100; ++i)
	{
		const Vector3D<T> r1(-0.1 + i, i*0.1, 2*i);
		const Vector3D<T> r2(-0.2 + 2 * i, i*0.1 - 3, 4*i);

		const Vector3D<T> result = interaction.forceOnR1DueToR2(r1, r2);
		const Vector3D<T> expected = interaction.force(r1 - r2);

		REQUIRE(result == expected);

		REQUIRE(
			result
			==
			wca1.forceOnR1DueToR2(r1, r2) + wca2.forceOnR1DueToR2(r1, r2));
	}
}
SCENARIO(
	"`OpenMPCD::PairPotentials::AdditivePair::forceOnR1DueToR2`",
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
