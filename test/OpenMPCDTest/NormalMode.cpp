/**
 * @file
 * Tests functionality of `OpenMPCD::NormalMode`.
 */

#include <OpenMPCDTest/include_catch.hpp>

#include <OpenMPCD/NormalMode.hpp>

#include <boost/math/constants/constants.hpp>
#include <boost/preprocessor/seq/for_each.hpp>

#include <vector>

#define SEQ_DATATYPE (float)(double)(OpenMPCD::FP)


static Catch::Detail::Approx Approx(const float value)
{
        return
                Catch::Detail::Approx(value).
                epsilon(std::numeric_limits<float>::epsilon() * 625);
}
static Catch::Detail::Approx Approx(const double value)
{
        return Catch::Detail::Approx(value);
}

template<typename T>
static void test_computeNormalCoordinate()
{
	static const unsigned int totalVectorCount = 50;

	static const T shifts[] = {0.0, -0.5};
	static const std::size_t shiftsCount = sizeof(shifts)/sizeof(shifts[0]);

	typedef OpenMPCD::Vector3D<T> V;

	std::vector<V> vectors;
	for(unsigned int i = 0; i < totalVectorCount; ++i)
		vectors.push_back(V(0.1 * i, -0.33 * i, 0.5 + i));

	using OpenMPCD::NormalMode::computeNormalCoordinate;

	#ifdef OPENMPCD_DEBUG
		REQUIRE_THROWS_AS(
			computeNormalCoordinate<T>(0, 0, 1),
			OpenMPCD::NULLPointerException);

		REQUIRE_THROWS_AS(
			computeNormalCoordinate(0, &vectors[0], 0),
			OpenMPCD::InvalidArgumentException);

		REQUIRE_THROWS_AS(
			computeNormalCoordinate(2, &vectors[0], 1),
			OpenMPCD::InvalidArgumentException);
	#endif

	const T pi = boost::math::constants::pi<T>();
	for(unsigned int N = 1; N <= totalVectorCount; ++N)
	{
		for(std::size_t shiftsIdx = 0; shiftsIdx < shiftsCount; ++shiftsIdx)
		{
			const T shift = shifts[shiftsIdx];
			for(unsigned int i = 0; i <= N; ++i)
			{
				V expected(0, 0, 0);
				for(unsigned int n = 1; n <= N; ++n)
				{
					const T cosArgument = T(i) * T(n + shift) * pi / T(N);
					expected += cos(cosArgument) * vectors[n - 1];
				}
				expected /= N;

				const V result =
					computeNormalCoordinate(i, &vectors[0], N, shift);

				REQUIRE(expected.getX() == Approx(result.getX()));
				REQUIRE(expected.getY() == Approx(result.getY()));
				REQUIRE(expected.getZ() == Approx(result.getZ()));
			}
		}

		{ //for i == 0, S == 0:
			V expected(0, 0, 0);
			for(unsigned int n = 1; n <= N; ++n)
				expected += vectors[n - 1];
			expected /= N;

			const V result = computeNormalCoordinate(0, &vectors[0], N);

			REQUIRE(expected.getX() == Approx(result.getX()));
			REQUIRE(expected.getY() == Approx(result.getY()));
			REQUIRE(expected.getZ() == Approx(result.getZ()));
		}

		{ //for i == N, S == 0:
			V expected(0, 0, 0);
			for(unsigned int n = 1; n <= N; ++n)
			{
				const int sign = n % 2 == 1 ? -1 : 1;
				expected += sign * vectors[n - 1];
			}
			expected /= N;

			const V result = computeNormalCoordinate(N, &vectors[0], N);

			REQUIRE(expected.getX() == Approx(result.getX()));
			REQUIRE(expected.getY() == Approx(result.getY()));
			REQUIRE(expected.getZ() == Approx(result.getZ()));
		}

		{ //for i == N, S == -0.5:
			const V expected(0, 0, 0);

			const V result =
				computeNormalCoordinate(N, &vectors[0], N, T(-0.5));

			REQUIRE(expected.getX() == Approx(result.getX()));
			REQUIRE(expected.getY() == Approx(result.getY()));
			REQUIRE(expected.getZ() == Approx(result.getZ()));
		}
	}

	vectors.clear();
	vectors.push_back(V(0, 1, 2));
	vectors.push_back(V(0.5, 1.5, 2.5));
	vectors.push_back(V(-0.5, -1.5, -2.5));

	{
		const std::size_t N = 1;
		const unsigned int i = 0;
		const V expected(0, 1, 2);

		const V result = computeNormalCoordinate(i, &vectors[0], N);
		REQUIRE(expected.getX() == Approx(result.getX()));
		REQUIRE(expected.getY() == Approx(result.getY()));
		REQUIRE(expected.getZ() == Approx(result.getZ()));
	}
	{
		const std::size_t N = 1;
		const unsigned int i = 1;
		const V expected(0, -1, -2);

		const V result = computeNormalCoordinate(i, &vectors[0], N);
		REQUIRE(expected.getX() == Approx(result.getX()));
		REQUIRE(expected.getY() == Approx(result.getY()));
		REQUIRE(expected.getZ() == Approx(result.getZ()));
	}

	{
		const std::size_t N = 2;
		const unsigned int i = 0;
		const V expected(1.0/4, 5.0/4, 9.0/4);

		const V result = computeNormalCoordinate(i, &vectors[0], N);
		REQUIRE(expected.getX() == Approx(result.getX()));
		REQUIRE(expected.getY() == Approx(result.getY()));
		REQUIRE(expected.getZ() == Approx(result.getZ()));
	}
	{
		const std::size_t N = 2;
		const unsigned int i = 1;
		const V expected(-1.0/4, -3.0/4, -5.0/4);

		const V result = computeNormalCoordinate(i, &vectors[0], N);
		REQUIRE(expected.getX() == Approx(result.getX()));
		REQUIRE(expected.getY() == Approx(result.getY()));
		REQUIRE(expected.getZ() == Approx(result.getZ()));
	}
	{
		const std::size_t N = 2;
		const unsigned int i = 2;
		const V expected(1.0/4, 1.0/4, 1.0/4);

		const V result = computeNormalCoordinate(i, &vectors[0], N);
		REQUIRE(expected.getX() == Approx(result.getX()));
		REQUIRE(expected.getY() == Approx(result.getY()));
		REQUIRE(expected.getZ() == Approx(result.getZ()));
	}

	{
		const std::size_t N = 3;
		const unsigned int i = 0;
		const V expected(0, 1.0/3, 2.0/3);

		const V result = computeNormalCoordinate(i, &vectors[0], N);
		REQUIRE(expected.getX() == Approx(result.getX()));
		REQUIRE(expected.getY() == Approx(result.getY()));
		REQUIRE(expected.getZ() == Approx(result.getZ()));
	}
	{
		const std::size_t N = 3;
		const unsigned int i = 1;
		const V expected(1.0/12, 5.0/12, 3.0/4);

		const V result = computeNormalCoordinate(i, &vectors[0], N);
		REQUIRE(expected.getX() == Approx(result.getX()));
		REQUIRE(expected.getY() == Approx(result.getY()));
		REQUIRE(expected.getZ() == Approx(result.getZ()));
	}
	{
		const std::size_t N = 3;
		const unsigned int i = 2;
		const V expected(-1.0/4, -11.0/12, -19.0/12);

		const V result = computeNormalCoordinate(i, &vectors[0], N);
		REQUIRE(expected.getX() == Approx(result.getX()));
		REQUIRE(expected.getY() == Approx(result.getY()));
		REQUIRE(expected.getZ() == Approx(result.getZ()));
	}
	{
		const std::size_t N = 3;
		const unsigned int i = 3;
		const V expected(1.0/3, 2.0/3, 1);

		const V result = computeNormalCoordinate(i, &vectors[0], N);
		REQUIRE(expected.getX() == Approx(result.getX()));
		REQUIRE(expected.getY() == Approx(result.getY()));
		REQUIRE(expected.getZ() == Approx(result.getZ()));
	}


	{
		const std::size_t N = 3;
		const unsigned int i = 0;
		const V expected(0, 1.0/3, 2.0/3);

		const V result = computeNormalCoordinate(i, &vectors[0], N, T(-0.5));
		REQUIRE(expected.getX() == Approx(result.getX()));
		REQUIRE(expected.getY() == Approx(result.getY()));
		REQUIRE(expected.getZ() == Approx(result.getZ()));
	}
	{
		const std::size_t N = 3;
		const unsigned int i = 1;
		const V expected(
			0.14433756729740646, 0.7216878364870323,1.2990381056766582);

		const V result = computeNormalCoordinate(i, &vectors[0], N, T(-0.5));
		REQUIRE(expected.getX() == Approx(result.getX()));
		REQUIRE(expected.getY() == Approx(result.getY()));
		REQUIRE(expected.getZ() == Approx(result.getZ()));
	}
	{
		const std::size_t N = 3;
		const unsigned int i = 2;
		const V expected(-0.25, -0.5833333333333334, -0.9166666666666666);

		const V result = computeNormalCoordinate(i, &vectors[0], N, T(-0.5));
		REQUIRE(expected.getX() == Approx(result.getX()));
		REQUIRE(expected.getY() == Approx(result.getY()));
		REQUIRE(expected.getZ() == Approx(result.getZ()));
	}
	{
		const std::size_t N = 3;
		const unsigned int i = 3;
		const V expected(0, 0, 0);

		const V result = computeNormalCoordinate(i, &vectors[0], N, T(-0.5));
		REQUIRE(expected.getX() == Approx(result.getX()));
		REQUIRE(expected.getY() == Approx(result.getY()));
		REQUIRE(expected.getZ() == Approx(result.getZ()));
	}
}
SCENARIO(
	"`OpenMPCD::NormalMode::computeNormalCoordinate`",
	"")
{
	#define CALLTEST(_r, _data, T) \
		test_computeNormalCoordinate<T>();

	BOOST_PP_SEQ_FOR_EACH(\
		CALLTEST, \
		_,
		SEQ_DATATYPE)

	#undef CALLTEST
}
