/**
 * @file
 * Tests
 * `OpenMPCD::PairPotentials::MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles`.
 */

#include <OpenMPCDTest/include_catch.hpp>
#include <OpenMPCDTest/include_libtaylor.hpp>

#include <OpenMPCD/PairPotentials/MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles.hpp>

#include <boost/preprocessor/seq/for_each.hpp>


#define SEQ_TEST_TYPES (double)(long double)(OpenMPCD::FP)


template<typename T>
static void test_force()
{
	using namespace OpenMPCD;
	using namespace OpenMPCD::PairPotentials;

	typedef taylor<T, 3, 1> Taylor;

	const T prefactor = 20;
	Vector3D<T> orientation(1, 1, 1);
	orientation.normalize();
	Vector3D<Taylor> orientation_taylor(1, 1, 1);
	orientation_taylor.normalize();
	const MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles<T>
		interaction(prefactor, orientation);
	const MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles<Taylor>
		interaction_taylor(prefactor, orientation_taylor);

	for(unsigned int i = 0; i < 100; ++i)
	{
		const Vector3D<T> r(-0.1 + i, i*0.1, 2*i);
		const Vector3D<Taylor> r_taylor(
			Taylor(r.getX(), 0), Taylor(r.getY(), 1), Taylor(r.getZ(), 2));

		const Vector3D<T> result = interaction.force(r);
		const Taylor result_taylor = - interaction_taylor.potential(r_taylor);

		REQUIRE(result.getX() == Approx(result_taylor[1]));
		REQUIRE(result.getY() == Approx(result_taylor[2]));
		REQUIRE(result.getZ() == Approx(result_taylor[3]));
	}
}
SCENARIO(
	"`OpenMPCD::PairPotentials::"
	"MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles::"
	"force`",
	"")
{
	#define SEQ_DATATYPE SEQ_TEST_TYPES
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

	const T prefactor = 20;
	Vector3D<T> orientation(1, 1, 1);
	orientation.normalize();
	const MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles<T>
		interaction(prefactor, orientation);

	for(unsigned int i = 0; i < 100; ++i)
	{
		const OpenMPCD::Vector3D<T> r(-0.1 + i, i*0.1, 2*i);

		const T R = r.getMagnitude();
		const T dot = orientation.dot(r) / R;
		const T expected = - prefactor * pow(R, -3) * (3 * dot * dot - 1);
		REQUIRE(expected == Approx(interaction.potential(r)));
	}
}
SCENARIO(
	"`OpenMPCD::PairPotentials::"
	"MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles::"
	"potential`",
	"")
{
	#define SEQ_DATATYPE SEQ_TEST_TYPES
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

	const T prefactor = 20;
	Vector3D<T> orientation(1, 1, 1);
	orientation.normalize();
	const MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles<T>
		interaction(prefactor, orientation);

	for(unsigned int i = 0; i < 100; ++i)
	{
		const Vector3D<T> r1(-0.1 + i, i*0.1, 2*i);
		const Vector3D<T> r2(-0.2 + 2 * i, i*0.1 - 3, 4*i);

		const Vector3D<T> result = interaction.forceOnR1DueToR2(r1, r2);
		const Vector3D<T> expected = interaction.force(r1 - r2);

		REQUIRE(result == expected);
	}
}
SCENARIO(
	"`OpenMPCD::PairPotentials::"
	"MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles::"
	"forceOnR1DueToR2`",
	"")
{
	#define SEQ_DATATYPE SEQ_TEST_TYPES
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
static void test_getPrefactor()
{
	using namespace OpenMPCD;
	using namespace OpenMPCD::PairPotentials;

	const T prefactor = 20;
	Vector3D<T> orientation(1, 1, 1);
	orientation.normalize();
	const MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles<T>
		interaction(prefactor, orientation);

	REQUIRE(interaction.getPrefactor() == prefactor);
}
SCENARIO(
	"`OpenMPCD::PairPotentials::"
	"MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles::"
	"getPrefactor`",
	"")
{
	#define SEQ_DATATYPE SEQ_TEST_TYPES
	#define CALLTEST(_r, _data, T) \
		test_getPrefactor<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef SEQ_DATATYPE
	#undef CALLTEST
}


template<typename T>
static void test_getOrientation()
{
	using namespace OpenMPCD;
	using namespace OpenMPCD::PairPotentials;

	const T prefactor = 20;
	Vector3D<T> orientation(1, 1, 1);
	orientation.normalize();
	const MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles<T>
		interaction(prefactor, orientation);

	REQUIRE(interaction.getDipoleOrientation() == orientation);
}
SCENARIO(
	"`OpenMPCD::PairPotentials::"
	"MagneticDipoleDipoleInteraction_ConstantIdenticalDipoles::"
	"test_getOrientation`",
	"")
{
	#define SEQ_DATATYPE SEQ_TEST_TYPES
	#define CALLTEST(_r, _data, T) \
		test_getOrientation<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef SEQ_DATATYPE
	#undef CALLTEST
}
